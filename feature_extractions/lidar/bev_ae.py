"""
BEV (madartavlat) autoencoder - fix statisztikai BEV + konvolucios AE.

A pontfelho rendezetlen: nincs "elso" pont, es framenkent mas a darabszam. A
konvolucio viszont RACSOT akar. A megoldas a PointPillars otlete: a teret
fentrol nezve negyzetracsra osztjuk (a magassag menten NEM vagunk, ezert
"pillar" = oszlop, nem "voxel" = kocka), es minden pontot beteszunk abba az
oszlopba, ahova esik. Cellankent aztan egy fix hosszu vektort keszitunk.

    nyers pontfelho (B, N, 3)
        -> BEV statisztikak    (B, 3, H, W)      fix keplet, NEM tanul
        -> konvolucios encoder (B, 256, 8, 8)    EZ MEGY AZ RL-BE
        -> konvolucios decoder (B, 3, H, W)      csak a tanitashoz kell

MIERT FIX A BEV, es nem tanult pillar-encoder (PointPillars PillarFeatureNet):
ott a cellankenti osszegzest egy mini-PointNet tanulja, de azt a detekcios fej
tanitja CIMKEKBOL - kivulrol jovo, fix tanitojelbol. Nekunk nincs cimkenk,
csak a rekonstrukcio, es azzal a tanult encoder korkoros lenne: a cel maga is
a halo kimenete, tehat eleg a NULLA fele huzni, amit a decoder trivialisan
eltalal. Egy teljes tanitason a val loss igy 0.0001-ig ment le - nem jo
rekonstrukcio, hanem ures latens. A fix BEV ugyanazt a szerepet tolti be
(ponthalmaz -> cellankent egy vektor), csak nem tud elszokni.

Referencia: Lang et al., "PointPillars" (CVPR 2019) - a pillar-racs otlete.
"""

import lightning as L
import torch
import torch.nn as nn


# A BEV cellankenti HAROM csatornaja, mind fix keplet, tanulhato suly nelkul:
#
#   0  occupancy  van-e pont a cellaban        0 / 1
#   1  z_max      a legmagasabb pont           0 .. 1
#   2  z_min      a legalacsonyabb pont        0 .. 1
#
# KORABBAN OT VOLT (density es z_mean is). Kimerve (25 epoch, 12000 frame,
# conv-neckkel), a ketto elhagyasa gyakorlatilag ingyen van:
#
#   csatornak                  IoU      z_mae
#   5 (occ,dens,zmax,zmin,zmean)  0.6517   0.509
#   3 (occ,zmax,zmin)             0.6433   0.523   <- ez
#   2 (occ,zmax)                  0.6401   0.571
#   1 (csak occ)                  0.6615   -
#
# A z_mean a z_max es z_min kozott van, tehat redundans. A density szinten:
# az occupancy mar megmondja, hogy van-e pont, a pontos darabszam pedig az
# RL-nek nem mond tobbet. A z_max+z_min viszont KELL: 2 csatornanal a
# magassaghiba visszaromlik (0.571), mert a magassag-tartomanyt egyetlen
# szam nem irja le - egy jardaszegely es egy auto nem ugyanaz.
#
# A z-csatornak ures cellaban 0-t kapnak. Onmagaban ez felrevezeto lenne (a 0
# egy ervenyes magassag is lehetne), ezert a valodi ertekeket [0.05, 1] koze
# normaljuk: az ures cella igy elkulonul a legalacsonyabb ponttol. A LOSS
# viszont ures cellaban nem nezi a z-t (lasd loss()): a dekoder Sigmoid-ja
# sosem ad pontos 0-t, igy abbol csak egy le nem vihato loss-padlo lenne.
BEV_CHANNELS = 3


# A BEV utan ez egy sima kep-autoencoder, ugyanaz a felepites, mint a kamera
# AE-nel: Conv2d(kernel=4, stride=2) blokkok felezik a meretet, a decoder
# ConvTranspose2d-vel duplaz vissza.
#
#   (3, 128, 128) -> (C, 64, 64) -> (2C, 32, 32) -> (4C, 16, 16) -> (8C, 8, 8)
#   A (8C, 8, 8) terbeli kimenet MAGA a latens - nincs vektorra lapitas.

class BEVConvAE(L.LightningModule):
    """Fix statisztikai BEV + konvolucios autoencoder.

    Tanitas:
        model = BEVConvAE()
        loss = model.loss(points)          # (B, N, 3) nyers pontfelho

    RL-ben (csak az encoder kell):
        z = model.encode(points)           # (B, 256, 8, 8) jellemzoterkep

    z_min/z_max : a magassag-normalas hatarai. Az alapertek a dataset merese
        szerint van beallitva (a pontok 90%-a -3.4 es +4.4 m kozott van);
        tulsagosan bo tartomany a valodi valtozatossagot egy keskeny savba
        nyomna ossze.
    geo_weight : a magassag-tag sulya az occupancy-hoz kepest. A ketto mas
        nagysagrendu (BCE+Dice vs maszkolt MSE); 5.0-nel indulnak egy skalan.
    pos_weight : a foglalt cellak sulya az occupancy BCE-ben. Ez allitja a
        precision/recall aranyt - kimerve 1.0 a legjobb.
    dice_weight : a Dice-tag sulya. A BCE cellankent fuggetlenul buntet, a
        Dice a halmazok atfedeset meri - azt, amit IoU-val ertekelunk.
    """

    def __init__(self, base_channels: int = 32,
                 lr: float = 1e-3,
                 voxel_size: float = 0.8, pc_range: float = 51.2,
                 z_min: float = -3.5, z_max: float = 5.0,
                 geo_weight: float = 5.0, pos_weight: float = 1.0,
                 dice_weight: float = 1.0):
        super().__init__()
        self.save_hyperparameters()

        self.pc_range = (-pc_range, -pc_range, z_min, pc_range, pc_range, z_max)
        self.grid_w = self.grid_h = int(round(2 * pc_range / voxel_size))

        C, Cin = base_channels, BEV_CHANNELS
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(Cin, C, 4, 2, 1, bias=False),
            nn.BatchNorm2d(C), nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(C, C * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(C * 2), nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(C * 2, C * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(C * 4), nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(C * 4, C * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(C * 8), nn.LeakyReLU(0.2, inplace=True),
        )

        # NINCS BOTTLENECK. A conv_encoder terbeli kimenete MAGA a latens.
        #
        # Korabban egy Linear-par (vagy conv-neck) vitte le 256-ra. MERVE, ez
        # dobta el a minoseg nagy reszet (25 epoch, 9000 frame):
        #
        #   szukites          precision   z_mae    param
        #   Linear -> 256      0.8176     0.466     2.8M
        #   Linear -> 1024     0.8158     0.474     3.2M
        #   Linear -> 2048     0.8184     0.469     3.7M
        #   NINCS              0.9410     0.464     1.4M   <- ez
        #
        # A latens MERETE nem szamit (256 es 2048 kozott nincs kulonbseg) - a
        # TERBELI SZERKEZET elvesztese a problema. Ezert a szukites teljesen
        # kikerult, es az RL sajat CNN feature-extractorral dolgozza fel a
        # jellemzoterkepet.
        #
        # Megprobaltuk a LiLa-Net (arxiv 2510.02028) skip-connection otletet
        # is, de a cikk sajat nullazasos tesztje megbuktatta: 64 csatornas
        # skippel a precision 0.936 lett, DE kinullazott latenssel is 0.910 -
        # vagyis a modell a skipen csempeszte at mindent, a latens kiurult.
        # Negy felezes utan ekkora marad. 128x128 bemenetnel ez (256, 8, 8).
        self.latent_shape = (C * 8, self.grid_h // 16, self.grid_w // 16)
        self.enc_out_shape = self.latent_shape     # regi nev, ugyanaz

        self.conv_decoder = nn.Sequential(
            nn.ConvTranspose2d(C * 8, C * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(C * 4), nn.ReLU(inplace=True),

            nn.ConvTranspose2d(C * 4, C * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(C * 2), nn.ReLU(inplace=True),

            nn.ConvTranspose2d(C * 2, C, 4, 2, 1, bias=False),
            nn.BatchNorm2d(C), nn.ReLU(inplace=True),

            nn.ConvTranspose2d(C, Cin, 4, 2, 1),
            # A BEV minden csatornaja [0, 1]-ben van, igy a halonak nem kell
            # kulon megtanulnia, hogy a tartomanyon kivulre ne menjen.
            nn.Sigmoid(),
        )

    @torch.no_grad()
    def to_bev(self, points):
        """Nyers pontfelho -> fix BEV jellemzoterkep. (B, N, 3) -> (B, 3, H, W)

        A batch indexet belekodoljuk a cella-kulcsba, igy egyetlen scatter
        elintezi az egesz batchet - batchenkenti Python ciklussal ez GPU-n a
        leptetesi ido nagy reszet elvinne.
        """
        B = points.shape[0]
        device, dtype = points.device, points.dtype
        x_min, y_min, z_min, x_max, y_max, z_max = self.pc_range
        v = self.hparams.voxel_size
        cells = self.grid_h * self.grid_w

        # Csak a hatosugaron beluli pontok erdekelnek.
        keep = ((points[..., 0] >= x_min) & (points[..., 0] < x_max)
                & (points[..., 1] >= y_min) & (points[..., 1] < y_max)
                & (points[..., 2] >= z_min) & (points[..., 2] < z_max))

        # Melyik cellaba esik minden pont - a batch indexevel egyutt kodolva.
        x_idx = ((points[..., 0] - x_min) / v).long().clamp(0, self.grid_w - 1)
        y_idx = ((points[..., 1] - y_min) / v).long().clamp(0, self.grid_h - 1)
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand_as(x_idx)
        flat = batch_idx * cells + y_idx * self.grid_w + x_idx

        # A hatosugaron kivuli pontok egy kuka-cellaba mennek, amit a vegen
        # levagunk: igy nincs maszkolt indexeles, es az alak fix marad.
        flat = torch.where(keep, flat, torch.full_like(flat, B * cells)).reshape(-1)

        z = ((points[..., 2] - z_min) / (z_max - z_min)).clamp(0, 1).reshape(-1)

        def scatter(src, reduce, init):
            out = torch.full((B * cells + 1,), init, device=device, dtype=dtype)
            out.scatter_reduce_(0, flat, src, reduce=reduce)
            return out[:-1].view(B, cells)

        count = scatter(torch.ones_like(z), "sum", 0.0)
        occupancy = (count > 0).to(dtype)

        # Az `* occupancy` nullazza az ures cellakban maradt init erteket, a
        # [0.05, 1] eltolas pedig elvalasztja az urest a legalacsonyabb ponttol.
        z_max_c = scatter(z, "amax", 0.0)
        z_min_c = scatter(z, "amin", 1.0)
        heights = [(0.05 + 0.95 * c) * occupancy for c in (z_max_c, z_min_c)]

        bev = torch.stack([occupancy, *heights], dim=1)
        return bev.view(B, BEV_CHANNELS, self.grid_h, self.grid_w)

    def encode_bev(self, bev):
        """BEV -> TERBELI jellemzoterkep. (B, 3, H, W) -> (B, C*8, H/16, W/16)

        NINCS TOBBE VEKTORRA LAPITAS. Korabban egy Linear vitte le 256-ra, de
        MERVE ez dobta el a minoseg nagy reszet:

            szukites            precision   z_mae
            Linear -> 256        0.8176     0.466
            Linear -> 1024       0.8158     0.474     (a meret nem szamit!)
            nincs szukites       0.9410     0.464     <- ez

        A latens MERETE nem volt a szuk keresztmetszet (256 es 2048 kozott
        nincs kulonbseg) - a TERBELI SZERKEZET elvesztese volt az. A racs
        szomszedossagi viszonyai egy lapos vektorban nem abrazolhatok.

        Az RL ezt a jellemzoterkepet kapja, es SAJAT CNN feature-extractorral
        dolgozza fel (SB3 BaseFeaturesExtractor). Igy az extractor a JUTALOMRA
        optimalizal, nem a rekonstrukciora - azt tanulja ki belole, ami a
        vezeteshez kell.
        """
        return self.conv_encoder(bev)

    def decode(self, h):
        """Jellemzoterkep -> BEV. Csak a tanitashoz kell."""
        return self.conv_decoder(h)

    def encode(self, points):
        """Nyers pontfelho -> jellemzoterkep. EZT hasznalja majd az RL."""
        return self.encode_bev(self.to_bev(points))

    def forward(self, points):
        """Teljes kor: pontfelho -> latens -> rekonstrualt BEV."""
        return self.decode(self.encode(points))

    def loss(self, points, parts=False):
        """Ket tag: occupancy (BCE + Dice) + magassag-MSE a FOGLALT cellakon.

        MIERT NEM EGY SULYOZOTT MSE MINDENRE (a regi valtozat): ket okbol allt
        meg 0.02 korul, es kozben az IoU csak 0.58 volt.

        1. A z-csatornak URES cellaban 0-t kapnak, a dekoder Sigmoid-ja viszont
           sosem ad pontos 0-t - a cellak ~80%-an maradt egy le nem vihato
           reziduum. Ezert MASZKOLJUK a magassagot: ures cellaban nem szamit.
        2. Az occupancy 0/1 cimke, arra az MSE gradiense elhal. Merve, azonos
           sulyokon: a kimeneti reteg gradiense 0.00089 volt, BCE-vel 0.0528.

        A Dice-tag azert kell a BCE melle, mert a BCE cellankent fuggetlenul
        buntet, a Dice viszont a halmazok atfedeset meri - ugyanazt, amit az
        IoU-val ertekelunk.

        parts=True : (loss, dict) - bontas a diagnosztikahoz. Egy szam
            osszemosva elrejti, ha az occupancy megvan, de a geometria nem.
        """
        bev = self.to_bev(points)
        rec = self.decode(self.encode_bev(bev))

        occ_t, occ_p = bev[:, 0], rec[:, 0]
        # A Sigmoid mar lefutott a dekoderben, ezert a logit helyett a
        # valoszinuseget kapjuk - clamp kell, kulonben a log(0) NaN-t ad.
        p = occ_p.clamp(1e-6, 1 - 1e-6)

        # pos_weight / dice_weight: KIMERVE, nem szarmaztatva. A pos_weight a
        # precision/recall aranyt allitja; nullarol tanitva 1.0 a legjobb
        # (3.0 -> IoU 0.460, 5.0 -> 0.444, 8.0 -> 0.326).
        pw = self.hparams.pos_weight
        occ_loss = -(pw * occ_t * p.log() + (1 - occ_t) * (1 - p).log()).mean()

        if self.hparams.dice_weight > 0:
            inter = (p * occ_t).sum(dim=(1, 2))
            denom = p.sum(dim=(1, 2)) + occ_t.sum(dim=(1, 2))
            occ_loss = occ_loss + self.hparams.dice_weight * (
                1.0 - (2 * inter + 1.0) / (denom + 1.0)).mean()

        # Magassag CSAK ott, ahol tenyleg van pont.
        mask = occ_t.unsqueeze(1)
        n = mask.sum().clamp(min=1) * (BEV_CHANNELS - 1)
        geo_loss = (((rec[:, 1:] - bev[:, 1:]) ** 2) * mask).sum() / n

        total = occ_loss + self.hparams.geo_weight * geo_loss
        if not parts:
            return total
        with torch.no_grad():
            pred, targ = occ_p > 0.5, occ_t > 0.5
            inter_c = pred & targ
            prec = inter_c.sum() / pred.sum().clamp(min=1)
            rec_r = inter_c.sum() / targ.sum().clamp(min=1)

            # Magassaghiba METERBEN, csak azokon a cellakon, amit a modell IS
            # foglaltnak tart. A VALODI foglalt cellakra atlagolva ez csal: a
            # kihagyott cellakban a z-josla ervenytelen, megis beleszamitana.
            # Merve ugyanazon a modellen: 0.824 m az osszesen, 0.595 a
            # metszeten - a kulonbseg tisztan a recall muve.
            m_int = inter_c.unsqueeze(1).float()
            span = self.hparams.z_max - self.hparams.z_min
            z_mae = ((rec[:, 1:] - bev[:, 1:]).abs() * m_int).sum() / (
                m_int.sum().clamp(min=1) * (BEV_CHANNELS - 1)) * span / 0.95
        return total, {"occ_bce": float(occ_loss), "geo_mse": float(geo_loss),
                       "occ_prec": float(prec), "occ_rec": float(rec_r),
                       "z_mae_m": float(z_mae)}

    # --- Lightning --------------------------------------------------------

    def _shared_step(self, batch, stage):
        points = batch[0] if isinstance(batch, (list, tuple)) else batch
        loss = self.loss(points)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
