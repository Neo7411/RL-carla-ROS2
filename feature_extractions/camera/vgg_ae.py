"""
VGG-stilusu autoencoder - a jzenn/Image-AutoEncoder architekturaja.

Forras: https://github.com/jzenn/Image-AutoEncoder

Az encoder a VGG-19 elso negy blokkjat koveti (a `relu4_1` retegig), a
dekoder ennek tukorkepe.

KET ELTERES AZ EREDETITOL, MINDKETTO AZ OSSZEHASONLITHATOSAG MIATT:

1. BOTTLENECK. Az eredeti NEM szukit latent vektorra: az encoder kimenete egy
   (512, H/8, W/8) terkep marad, amit a dekoder kozvetlenul visszaepit. Ez
   stilusatvitelhez (style transfer) valo, ahol a terbeli szerkezet kell.
   Nekunk viszont EGY latent VEKTOR kell az RL observationhoz, ezert az
   encoder utan jon egy fc1 (terkep -> 256), a dekoder ele pedig egy fc2.
   Igy ugyanaz a szuk keresztmetszet, mint a masik ket modellnel.

2. LOSS. Az eredeti harom tagot kombinal: feature loss (a rekonstrukcio
   ujrakodolasan), per-pixel loss es TV regularizer. Mi sima MSE-t
   hasznalunk, mint a masik ket modellnel - kulonben nem a halokat
   hasonlitanank ossze, hanem a loss fuggvenyeket.

AMI VALTOZATLAN:

- ReflectionPad2d a konvoluciok elott. A sima nulla-padding sotet keretet
  rajzol a kep szelere (a halo "latja" a nullakat); a tukrozott padding a
  kep sajat tartalmat hajtja vissza, ezert a szel is termeszetes marad.
- MaxPool az encoderben, UpsamplingNearest a dekoderben.
- A csatornaszamok es a blokkfelepites: 64 -> 128 -> 256 -> 512.

HASZNALAT
    from camera.vgg_ae import VGGAE
    model = VGGAE(latent_dim=256)
"""

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_ch, out_ch):
    """ReflectionPad + Conv3x3 + ReLU - az eredeti alapegysege."""
    return [
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(in_ch, out_ch, 3, 1, 0),
        nn.ReLU(inplace=True),
    ]


class Encoder(nn.Module):
    """
    VGG-19 elso negy blokkja a relu4_1-ig.

        (3,   80, 160)
        (64,  80, 160)   1. blokk
        (64,  40,  80)   maxpool
        (128, 40,  80)   2. blokk
        (128, 20,  40)   maxpool
        (256, 20,  40)   3. blokk
        (256, 10,  20)   maxpool
        (512, 10,  20)   4. blokk (r41)
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # 1. blokk - az elso 1x1 konvolucio az eredetiben is ott van
            nn.Conv2d(3, 3, 1, 1, 0),
            *conv_block(3, 64),
            *conv_block(64, 64),
            nn.MaxPool2d(2, 2),

            # 2. blokk
            *conv_block(64, 128),
            *conv_block(128, 128),
            nn.MaxPool2d(2, 2),

            # 3. blokk
            *conv_block(128, 256),
            *conv_block(256, 256),
            *conv_block(256, 256),
            *conv_block(256, 256),
            nn.MaxPool2d(2, 2),

            # 4. blokk - itt all meg az eredeti (r41)
            *conv_block(256, 512),
        )

    def forward(self, x):
        return self.net(x)

    def load_pretrained(self):
        """
        ImageNet-en tanult VGG-19 sulyok betoltese.

        MIERT KELL: 13 konvolucios reteg egymas utan, normalizalas nelkul.
        Veletlen inicializaciobol a jel szorasa retegrol retegre csokken, es
        a vegen a dekoder kimenete gyakorlatilag konstans - a Sigmoid mindent
        0.5-re visz, a gradiens elhal, a halo NEM TANUL. (Meresse: veletlen
        indulassal a loss 20 lepes utan sem mozdult.)

        Az eredeti repo ezert tolt be elotanitott sulyokat. A torchvision
        VGG-19 `features` modulja ugyanabban a sorrendben tartalmazza a
        konvoluciokat, mint a mi halonk, ezert sorban parba allithatok.
        A padding mas (mi ReflectionPad-et hasznalunk kulon retegkent), de a
        SULYOK ettol fuggetlenul ervenyesek - csak a kep szelen ter el a
        viselkedes.
        """
        from torchvision.models import VGG19_Weights, vgg19

        pretrained = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        src = [m for m in pretrained if isinstance(m, nn.Conv2d)]
        # A sajat halonk elso konvolucioja egy 1x1-es "szinkevero", ami a
        # VGG-ben nincs meg - azt kihagyjuk a parositasbol.
        dst = [m for m in self.net if isinstance(m, nn.Conv2d)][1:]

        with torch.no_grad():
            for d, s in zip(dst, src):
                d.weight.copy_(s.weight)
                d.bias.copy_(s.bias)
        return len(dst)


class Decoder(nn.Module):
    """Az encoder tukorkepe: UpsamplingNearest + konvoluciok."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # 1. blokk
            *conv_block(512, 256),
            nn.UpsamplingNearest2d(scale_factor=2),

            # 2. blokk
            *conv_block(256, 256),
            *conv_block(256, 256),
            *conv_block(256, 256),
            *conv_block(256, 128),
            nn.UpsamplingNearest2d(scale_factor=2),

            # 3. blokk
            *conv_block(128, 128),
            *conv_block(128, 64),
            nn.UpsamplingNearest2d(scale_factor=2),

            # 4. blokk - a vegen NINCS ReLU, mert a Sigmoid jon utana
            *conv_block(64, 64),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, 3, 1, 0),
        )

        # KAIMING INICIALIZALAS. A dekoder 9 konvolucios rétege ReLU-kkal
        # valtakozik, normalizalas nelkul. A PyTorch alapertelmezett
        # inicializalasa nem ReLU-ra van szabva, es a jel szorasa retegrol
        # retegre csokken: meressel 0.29 -> 0.014 (20x csillapitas), amitol a
        # Sigmoid mindent 0.5-re visz, es a halo NEM TANUL.
        #
        # A Kaiming-normal (He et al. 2015) pont ezt kompenzalja: a sulyokat
        # 2/fan_in szorassal huzza, ahol a 2-es szorzo a ReLU feleles
        # levagasat ellensulyozza. Az encoder ezt nem igenyli, mert oda
        # elotanitott VGG-19 sulyok kerulnek.
        for m in self.net:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in",
                                        nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class VGGAE(L.LightningModule):
    """
    (3, 80, 160) -> latent_dim -> (3, 80, 160)

    img_size: (magassag, szelesseg). Harom maxpool van, tehat a terkep
              merete a nyolcada a bemenetnek: 80x160 -> 10x20.
    bottleneck_channels: ennyire vagjuk le az 512 csatornat a lapitas elott.

    FIGYELEM - A TANULASI RATA: az alapertek 1e-4, NEM 1e-3 mint a masik ket
    modellnel. Ez a halo 22 konvolucios reteg normalizalas nelkul; 1e-3
    mellett meressel elszall (a loss 0.09-rol 0.33-ra ugrik, a Sigmoid 0/1-re
    telitodik, es ott is ragad). Ha atallitod, ellenorizd a tanulasi gorbet.
    """

    def __init__(self, latent_dim: int = 256, img_size=(80, 160),
                 bottleneck_channels: int = 32, lr: float = 1e-4,
                 latent_scale: float = 4.0, pretrained: bool = True,
                 freeze_encoder: bool = False):
        super().__init__()
        self.save_hyperparameters()

        h, w = img_size
        self.map_h, self.map_w = h // 8, w // 8      # 3 maxpool
        self.bc = bottleneck_channels
        flat_dim = self.bc * self.map_h * self.map_w

        self.encoder = Encoder()
        self.decoder = Decoder()

        # Elotanitott sulyok: enelkul a halo nem tanul (lasd load_pretrained).
        if pretrained:
            self.encoder.load_pretrained()

        # Fagyasztott encoder: csak a dekoder es a bottleneck tanul. Az
        # eredeti repo get_encoder_decoder_model()-je is igy csinalja.
        # Gyorsabb, de a jellemzok ImageNet-re vannak hangolva, nem CARLA
        # kepekre - ezert alapbol finomhangolunk (freeze_encoder=False).
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # 512 csatorna 10x20-on = 102 400 ertek. Ezt 1x1 konvolucioval
        # visszuk le bottleneck_channels-re, mert egy 102 400 -> 256 Linear
        # egymagaban 26M parametert jelentene - tobbet, mint az egesz halo.
        self.to_bottleneck = nn.Conv2d(512, self.bc, 1)
        self.from_bottleneck = nn.Conv2d(self.bc, 512, 1)

        self.fc1 = nn.Linear(flat_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, flat_dim)

    def encode(self, x):
        h = self.to_bottleneck(self.encoder(x)).flatten(1)
        # tanh + skala: a latent az RL observation resze lesz, ott korlatozni
        # kell (ugyanaz az indoklas, mint a camera_ae.py-ban).
        return self.hparams.latent_scale * torch.tanh(self.fc1(h))

    def decode(self, z):
        h = self.fc2(z).view(-1, self.bc, self.map_h, self.map_w)
        h = self.from_bottleneck(h)
        # Sigmoid: a bemenet [0,1], a kimenet is oda keruljon.
        return torch.sigmoid(self.decoder(h))

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
        # Csak a tanithato parametereket adjuk at: fagyasztott encoder mellett
        # az Adam kulonben hibat dobna a gradiens nelkuli tenzorokra.
        params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.Adam(params, lr=self.hparams.lr)


if __name__ == "__main__":
    model = VGGAE(latent_dim=256)
    x = torch.rand(2, 3, 80, 160)

    z = model.encode(x)
    rec = model(x)

    print(f"bemenet      : {tuple(x.shape)}")
    print(f"encoder (r41): (512, {model.map_h}, {model.map_w})")
    print(f"bottleneck   : ({model.bc}, {model.map_h}, {model.map_w})")
    print(f"latens       : {tuple(z.shape)}  [{z.min():.2f}, {z.max():.2f}]")
    print(f"rekonstrukcio: {tuple(rec.shape)}  [{rec.min():.2f}, {rec.max():.2f}]")
    assert rec.shape == x.shape, "ALAK ELTERES!"

    # Tanulas-ellenorzes: 30 lepes ugyanazon a par kepen. Ha a loss nem
    # csokken, a jel elhal valahol a halon (ez tortent elotanitott sulyok
    # nelkul), es a modell hasznalhatatlan.
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
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"tanito lepes : loss {first:.4f} -> {last:.4f}")
    print(f"kimenet most : [{rec.min():.3f}, {rec.max():.3f}]")
    print(f"parameterek  : {n / 1e6:.1f}M ({train / 1e6:.1f}M tanithato)")
    assert last < first * 0.9, "NEM TANUL - a loss nem csokkent erdemben!"
    print("OK")
