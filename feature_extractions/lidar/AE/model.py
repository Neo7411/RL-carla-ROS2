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


class LidarAE(pl.LightningModule):
    """
    Range image autoencoder graf-alapu encoderrel.

    Determinisztikus AE (mint az eredeti TopoAutoencoder): egy kep -> egy
    konkret latens, nem eloszlas. A ddconfig double_z-je ezert vegig False,
    az encoder z_channels csatornat ad.
    """

    def __init__(self,
                 ddconfig,
                 embed_dim=16,
                 image_key="image",
                 learning_rate=1e-4,
                 ckpt_path=None,
                 monitor="val/rec_loss",
                 **kwargs
                 ):
        super().__init__()
        self.image_key = image_key
        self.learning_rate = learning_rate
        self.monitor = monitor

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

    def encode(self, x):
        """
        Range image -> latens.

        x      : (B, 1, 64, 1024) float, [-1, 1]
        return: z, shape (B, z_channels, 16, 128)

        VESZTESEGMENTES: a teljes feature map jon vissza, pont ez megy a
        decode()-ba is. A dimenzio-csokkentes (RL obs-hoz) NEM itt tortenik
        - lasd a lenti megjegyzest.
        """
        h, _, _ = self.encoder(x)
        return h

    def decode(self, z):
        """Latens -> range image. RL futaskor NEM kell."""
        return self.decoder(z)

    # ------------------------------------------------------------------
    # MEGJEGYZES az RL-integraciohoz (meg NINCS megoldva, ott intezzuk)
    #
    # Az encode() (B, z_channels, 16, 128) = 32768 szamot ad mintankent,
    # korlatlan tartomanyban. Ez tudatos: az AE dolga a veszteseg nelkuli
    # kodolas, es a decoder is ezt a teljes feature mapet varja.
    #
    # Az RL obs-ba viszont igy nem tehato be kozvetlenul, ket ok miatt:
    #
    #   1. MERET: a kamera latent 64 dim, ez 32768 - 512-szeres tulsuly.
    #      Osszefuzve elnyomna a waypointokat, a sebesseget, mindent.
    #
    #   2. TARTOMANY: az encoder vegen csak egy nn.Conv1d all, aktivacio
    #      NELKUL (encoder.py), tehat nincs felso korlat. Meresben a
    #      jelenlegi ckpt -1.42 .. +2.96 kozott mozog, de ez a tanitas
    #      soran szabadon elszallhat. A gym.spaces.Box fix low/high-t var,
    #      es az SB3 a hataron kivuli erteket nem vagja le.
    #
    # Ezt a ket dolgot az RL oldalan kell kezelni (train_rl.py), nem itt:
    # az AE maradjon veszteseg nelkuli. Merve, ha valaha kell:
    #   - pooling (16 csat. x 4 azimut szektor = 64 dim) -> -1.84..3.81
    #   - utana tanh * 4 -> garantalt -4..4 (extrem bemenetre is tart)
    # ------------------------------------------------------------------

    def forward(self, x):
        """
        Teljes kor. Tanitashoz.
        return: x_rec
        """
        return self.decoder(self.encode(x))

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
        x_rec = self(inputs)

        # L1 rekonstrukcio (az eredeti TopoAutoencoder is ezt hasznalta)
        loss = F.l1_loss(inputs, x_rec)

        self.log(f"{stage}/rec_loss", loss, prog_bar=True)
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
        xrec = self(x)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log
