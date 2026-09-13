"""
LidarAE: a teljes autoencoder (graf encoder + ResNet decoder).

Forras: lidm/models/autoencoder_topo.py (TopoAutoencoder), a topologiai
        loss reszek NELKUL.

Ami kimaradt az eredetibol es miert:
  - scipy.spatial.Delaunay import        -> a topo filtraciohoz kellett
  - topologylayer / dionysus importok    -> a perzisztens homologiahoz
  - setup_filtration()                   -> 4x4-es Delaunay racs
  - _reduce_to_scalar_field()            -> a feature map 16 skalarra butitasa
  - a training_step topo-blokkja         -> .cpu() hivas minden lepesben (!)

Ezzel eltunt a TopologyLayer + Dionysus + libboost + scipy fuggoseg.
Marad: torch, numpy, lightning.
"""

import torch
import torch.nn.functional as F

try:
    import pytorch_lightning as pl
except ImportError:
    import lightning as pl

from .encoder import Encoder
from .decoder import Decoder
from .distributions import DiagonalGaussianDistribution


class LidarAE(pl.LightningModule):
    """
    Range image autoencoder graf-alapu encoderrel.

    Ket uzemmod:
      kl_weight = 0    -> determinisztikus AE (mint az eredeti TopoAutoencoder)
      kl_weight > 0    -> VAE (double_z=True kell a ddconfig-ban!)

    A VAE-nek RL-ben van egy konkret elonye: a KL-tag magatol N(0,1) kore
    huzza a latenst, igy nem kell kezzel skalazni (tanh * latent_scale).
    """

    def __init__(self,
                 ddconfig,
                 embed_dim=16,
                 image_key="image",
                 learning_rate=1e-4,
                 kl_weight=0.0,
                 ckpt_path=None,
                 monitor="val/rec_loss",
                 **kwargs
                 ):
        super().__init__()
        self.image_key = image_key
        self.learning_rate = learning_rate
        self.kl_weight = kl_weight
        self.monitor = monitor

        # VAE eseten az encodernek 2*z csatornat kell adnia
        self.use_vae = kl_weight > 0
        if self.use_vae:
            ddconfig = dict(ddconfig)
            ddconfig['double_z'] = True

        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu", weights_only=False)
        if "state_dict" in sd:
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    # ------------------------------------------------------------------
    # Az RL EZT a ket metodust fogja hasznalni (fokent az elsot)
    # ------------------------------------------------------------------

    def encode(self, x, sample=False):
        """
        Range image -> latens.

        x      : (B, 1, 64, 1024) float, [-1, 1]
        return: VAE eseten posterior.mode() (a mean), kulonben z
                shape: (B, z_channels, 16, 128)

        RL-ben: sample=False (determinisztikus, a mean kell, nem minta).
        """
        h, _, _ = self.encoder(x)
        if self.use_vae:
            posterior = DiagonalGaussianDistribution(h)
            return posterior.sample() if sample else posterior.mode()
        return h

    def decode(self, z):
        """Latens -> range image. RL futaskor NEM kell."""
        return self.decoder(z)

    def forward(self, x):
        """
        Teljes kor. Tanitashoz.
        return: x_rec, posterior (vagy None, ha nem VAE)
        """
        h, _, _ = self.encoder(x)
        if self.use_vae:
            posterior = DiagonalGaussianDistribution(h)
            z = posterior.sample()
        else:
            posterior, z = None, h
        return self.decoder(z), posterior

    # ------------------------------------------------------------------
    # Tanitas
    # ------------------------------------------------------------------

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[:, None]
        return x

    def _shared_step(self, batch, stage):
        inputs = self.get_input(batch, self.image_key)
        x_rec, posterior = self(inputs)

        # L1 rekonstrukcio (az eredeti TopoAutoencoder is ezt hasznalta)
        loss_rec = F.l1_loss(inputs, x_rec)
        loss = loss_rec

        self.log(f"{stage}/rec_loss", loss_rec, prog_bar=True)

        if posterior is not None:
            loss_kl = posterior.kl().mean()
            loss = loss + self.kl_weight * loss_kl
            self.log(f"{stage}/kl_loss", loss_kl, prog_bar=True)

        self.log(f"{stage}/total_loss", loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.learning_rate, betas=(0.5, 0.9)
        )

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key).to(self.device)
        xrec, _ = self(x)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log
