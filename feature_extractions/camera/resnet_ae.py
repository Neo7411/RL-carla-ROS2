"""
ResNetAE - a farrell236/ResNetAE TensorFlow implementacio PyTorch atirata.

Forras: https://github.com/farrell236/ResNetAE

AZ EREDETIHEZ KEPEST EGYETLEN VALTOZTATAS: az eredeti negyzetes kepet kovetel
(`assert input_shape[0] == input_shape[1]`), mi viszont 160x80-as kepekkel
dolgozunk. Ezert a latent terkep merete kulon van szamolva magassagra es
szelessegre:

    (80, 160) --4 szint felezes-->  (5, 10)

A halo logikaja valtozatlan.

FELEPITES

  ENCODER                               DECODER
    input_conv: 3 -> 8 csatorna           input_conv: z_dim -> max_filters
    4 szint, szintenkent:                 4 szint, szintenkent:
      n_ResidualBlock db ResBlock           ConvTranspose (duplazas)
      strided conv (felezes)                n_ResidualBlock db ResBlock
      + multi-res skip a vegere             + multi-res skip a z_top-bol
    output_conv -> z_dim                  output_conv -> 3 csatorna

MULTI-RESOLUTION SKIP

Ez az architektura lenyege. Minden szint kimenete kap egy sajat "rovidzarat"
egyenesen a bottleneck ele, kulonbozo lepteku konvolucioval (ks = 2^(n-i)).
Az encoder vegen ezek OSSZEADODNAK:

    x = x + skip_0 + skip_1 + skip_2 + skip_3

Igy a finom reszletek (korai szintek, nagy felbontas) is eljutnak a
latentig, nem csak a legmelyebb, legdurvabb jellemzok.

FONTOS: ez NEM U-Net skip. A U-Net az encoderbol a DECODERBE vezet at
informaciot, megkerulve a latentet. Itt minden skip a bottleneck ELOTT
egyesul, tehat minden informacio atmegy a latent vektoron - ami az RL
felhasznalashoz kotelezo.

HASZNALAT
    from camera.resnet_ae import ResNetAE
    model = ResNetAE(latent_dim=256)
"""

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Az eredeti ResidualBlock: ket Conv-BN-LeakyReLU, majd x + F(x).

    A csatornaszam nem valtozik (az eredetiben sem), ezert a skip agon
    nincs szukseg alakitasra.
    """

    def __init__(self, channels, kernel_size=3, stride=1):
        super().__init__()
        pad = kernel_size // 2       # 'same' padding
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, stride, pad),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size, stride, pad),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return x + self.block(x)


class ResNetEncoder(nn.Module):
    """
    (3, H, W) -> (z_dim, H/16, W/16)

    Szintenkent: n_ResidualBlock darab ResBlock, majd felezes. A szintek
    csatornaszama 8, 16, 32, 64 (azaz 2^(i+3)), a felezes utan 2^(i+4).
    """

    def __init__(self, in_channels=3, n_ResidualBlock=8, n_levels=4,
                 z_dim=10, bUseMultiResSkips=True):
        super().__init__()
        self.n_levels = n_levels
        self.bUseMultiResSkips = bUseMultiResSkips
        self.max_filters = 2 ** (n_levels + 3)

        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.res_blk_list = nn.ModuleList()
        self.conv_list = nn.ModuleList()
        self.multi_res_skip_list = nn.ModuleList()

        for i in range(n_levels):
            n_filters_1 = 2 ** (i + 3)
            n_filters_2 = 2 ** (i + 4)
            ks = 2 ** (n_levels - i)

            self.res_blk_list.append(nn.Sequential(
                *[ResidualBlock(n_filters_1) for _ in range(n_ResidualBlock)]
            ))

            self.conv_list.append(nn.Sequential(
                nn.Conv2d(n_filters_1, n_filters_2, 2, 2, 0),
                nn.BatchNorm2d(n_filters_2),
                nn.LeakyReLU(0.2, inplace=True),
            ))

            if bUseMultiResSkips:
                # ks lepteku konvolucio: ez viszi le a szint kimenetet
                # egyenesen a bottleneck felbontasara.
                self.multi_res_skip_list.append(nn.Sequential(
                    nn.Conv2d(n_filters_1, self.max_filters, ks, ks, 0),
                    nn.BatchNorm2d(self.max_filters),
                    nn.LeakyReLU(0.2, inplace=True),
                ))

        self.output_conv = nn.Conv2d(self.max_filters, z_dim, 3, 1, 1)

    def forward(self, x):
        x = self.input_conv(x)

        skips = []
        for i in range(self.n_levels):
            x = self.res_blk_list[i](x)
            if self.bUseMultiResSkips:
                skips.append(self.multi_res_skip_list[i](x))
            x = self.conv_list[i](x)

        if self.bUseMultiResSkips:
            # Minden szint rovidzara ideadodik a legmelyebb jellemzokhoz.
            x = sum([x] + skips)

        return self.output_conv(x)


class ResNetDecoder(nn.Module):
    """
    (z_dim, H/16, W/16) -> (out_channels, H, W)

    Az encoder tukorkepe: szintenkent ConvTranspose duplazas, majd
    ResBlock-ok. A multi-res skipek itt a BEMENETBOL (z_top) agaznak el,
    minden szinthez sajat leptekkel.
    """

    def __init__(self, n_ResidualBlock=8, n_levels=4, z_dim=10,
                 output_channels=3, bUseMultiResSkips=True):
        super().__init__()
        self.n_levels = n_levels
        self.bUseMultiResSkips = bUseMultiResSkips
        self.max_filters = 2 ** (n_levels + 3)

        self.input_conv = nn.Sequential(
            nn.Conv2d(z_dim, self.max_filters, 3, 1, 1),
            nn.BatchNorm2d(self.max_filters),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.res_blk_list = nn.ModuleList()
        self.conv_list = nn.ModuleList()
        self.multi_res_skip_list = nn.ModuleList()

        for i in range(n_levels):
            n_filters = 2 ** (n_levels - i + 2)
            ks = 2 ** (i + 1)
            in_ch = self.max_filters if i == 0 else 2 ** (n_levels - i + 3)

            self.conv_list.append(nn.Sequential(
                nn.ConvTranspose2d(in_ch, n_filters, 2, 2, 0),
                nn.BatchNorm2d(n_filters),
                nn.LeakyReLU(0.2, inplace=True),
            ))

            self.res_blk_list.append(nn.Sequential(
                *[ResidualBlock(n_filters) for _ in range(n_ResidualBlock)]
            ))

            if bUseMultiResSkips:
                self.multi_res_skip_list.append(nn.Sequential(
                    nn.ConvTranspose2d(self.max_filters, n_filters, ks, ks, 0),
                    nn.BatchNorm2d(n_filters),
                    nn.LeakyReLU(0.2, inplace=True),
                ))

        self.output_conv = nn.Conv2d(2 ** 3, output_channels, 3, 1, 1)

    def forward(self, z):
        z = z_top = self.input_conv(z)

        for i in range(self.n_levels):
            z = self.conv_list[i](z)
            z = self.res_blk_list[i](z)
            if self.bUseMultiResSkips:
                z = z + self.multi_res_skip_list[i](z_top)

        return self.output_conv(z)


class ResNetAE(L.LightningModule):
    """
    Teljes autoencoder: kep -> latent vektor -> kep.

    img_size: (magassag, szelesseg). Az eredeti implementacio negyzetes
              kepet kovetelt, ez a valtozat teglalapot is elfogad.
    latent_dim: a bottleneck vektor hossza (az eredetiben bottleneck_dim).
    z_dim: a konvolucios latent terkep csatornaszama a bottleneck ELOTT.

    Z_DIM = 64 ES NEM 10 (az eredeti alapertek):

    Az eredeti 256x256-os kepekkel dolgozott, ott a latent terkep 16x16 volt,
    tehat a bottleneck ele 16*16*10 = 2560 ertek erkezett. A mi 80x160-as
    kepeinknel a terkep csak 5x10, vagyis z_dim=10 mellett 5*10*10 = 500
    ertek maradna - ebbol kellene 256 dimenziot kinyerni.

    Az 500 -> 256 szoritas maga lenne a szuk keresztmetszet, nem a latent:
    a halo mar azelott elvesztene az informaciot, hogy a bottleneckhez erne.
    z_dim=64 mellett 5*10*64 = 3200 ertek all rendelkezesre, ami aranyaiban
    az eredetihez hasonlo.
    """

    def __init__(self, latent_dim: int = 256, img_size=(80, 160),
                 n_ResidualBlock: int = 8, n_levels: int = 4,
                 z_dim: int = 64, lr: float = 1e-3,
                 latent_scale: float = 4.0, bUseMultiResSkips: bool = True):
        super().__init__()
        self.save_hyperparameters()

        # A latent terkep merete a n_levels felezes utan. Az eredeti
        # implementacio egyetlen szamot hasznalt (negyzetes kep), itt
        # magassag es szelesseg kulon.
        h, w = img_size
        self.lat_h = h // (2 ** n_levels)
        self.lat_w = w // (2 ** n_levels)
        self.z_dim = z_dim
        flat_dim = self.lat_h * self.lat_w * z_dim

        self.encoder = ResNetEncoder(3, n_ResidualBlock, n_levels, z_dim,
                                     bUseMultiResSkips)
        self.decoder = ResNetDecoder(n_ResidualBlock, n_levels, z_dim, 3,
                                     bUseMultiResSkips)

        self.fc1 = nn.Linear(flat_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, flat_dim)

    def encode(self, x):
        h = self.encoder(x).flatten(1)
        # tanh + skala: az eredetiben nincs, de a latent az RL observation
        # resze lesz, es ott korlatozni kell, kulonben elnyomja a tobbi jelet
        # (ugyanaz az indoklas, mint a camera_ae.py-ban).
        return self.hparams.latent_scale * torch.tanh(self.fc1(h))

    def decode(self, z):
        z = self.fc2(z).view(-1, self.z_dim, self.lat_h, self.lat_w)
        # Sigmoid: a bemenet [0,1], a kimenet is oda keruljon.
        return torch.sigmoid(self.decoder(z))

    def forward(self, x):
        return self.decode(self.encode(x))

    # ------------------------------------------------------------------

    def _shared_step(self, batch, stage):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        loss = F.mse_loss(self(x), x)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False,
                 on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


if __name__ == "__main__":
    # Onteszt: alakok es egy tanito lepes 160x80-as kepekkel.
    model = ResNetAE(latent_dim=256)
    x = torch.rand(2, 3, 80, 160)

    z = model.encode(x)
    rec = model(x)

    print(f"bemenet      : {tuple(x.shape)}")
    print(f"latent terkep: ({model.z_dim}, {model.lat_h}, {model.lat_w})")
    print(f"latens       : {tuple(z.shape)}  [{z.min():.2f}, {z.max():.2f}]")
    print(f"rekonstrukcio: {tuple(rec.shape)}  [{rec.min():.2f}, {rec.max():.2f}]")
    assert rec.shape == x.shape, "ALAK ELTERES!"

    # Tanulas-ellenorzes: ha a loss nem csokken, a jel elhal valahol a halon,
    # es a modell hasznalhatatlan.
    opt = model.configure_optimizers()
    first = last = None
    for i in range(30):
        loss = F.mse_loss(model(x), x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i == 0:
            first = loss.item()
        last = loss.item()

    with torch.no_grad():
        rec = model(x)

    n = sum(p.numel() for p in model.parameters())
    print(f"tanito lepes : loss {first:.4f} -> {last:.4f}")
    print(f"kimenet most : [{rec.min():.3f}, {rec.max():.3f}]")
    print(f"parameterek  : {n / 1e6:.1f}M")
    assert last < first * 0.9, "NEM TANUL - a loss nem csokkent erdemben!"
    print("OK")
