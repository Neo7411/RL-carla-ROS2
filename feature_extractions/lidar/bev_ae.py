"""
BEV (madartavlat) autoencoder - PointPillars pillar-encoder + konvolucios AE.

A PointPillars ket reszbol all: egy pillar-encoderbol, ami a nyers pontfelhot
2D "pszeudo-kepbe" teriti, es egy detektalo fejbol. A detektalo fej cimkeket
igenyel, ami nekunk nincs - ezert csak az ELSO reszt vesszuk at, es a
pszeudo-kepre teszunk egy konvolucios autoencodert. Igy a halo cimke nelkul,
onmagat rekonstrualva tanul, pont mint a kamera AE.

    nyers pontfelho (B, N, 3)
        -> pillar voxelizacio       (B, P, K, 3)     nincs CUDA op, sima torch
        -> PillarFeatureNet         (B, P, Cp)       mini-PointNet oszloponkent
        -> scatter                  (B, Cp, H, W)    a pszeudo-kep
        -> konvolucios encoder      (B, latent_dim)  EZ MEGY AZ RL-BE
        -> konvolucios decoder      (B, Cp, H, W)    csak a tanitashoz kell

A pillar-racs itt szandekosan kicsi (128x128 a CARLA 50 m-es hatosugarara,
0.8 m-es cella), mert a KITTI-meretu 496x432-es racs ekkora latenshez
felesleges, es lassu is.

Referencia: Lang et al., "PointPillars" (CVPR 2019) - a pillar-encoder resze.
"""

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. VOXELIZACIO ES SCATTER
# =============================================================================
#
# Mindketto allapotmentes (nincs tanulhato suly), ezert fuggveny, nem osztaly.
#
# A pontfelho rendezetlen: nincs "elso" pont, es framenkent mas a darabszam. A
# konvolucio viszont RACSOT akar. A pillar-voxelizacio ezt hidalja at: a teret
# fentrol nezve negyzetracsra osztjuk (a magassag mentén NEM vagunk, ezert
# "pillar" = oszlop, nem "voxel" = kocka), es minden pontot beteszunk abba az
# oszlopba, ahova esik.
#
# A hivatalos implementacio erre CUDA kernelt hasznal. Az alabbi tiszta torch:
# lassabb, de barhol fut, es forditani sem kell.
# =============================================================================

@torch.no_grad()
def voxelize(points, pc_range, voxel_size, grid_w, grid_h, max_points, max_pillars):
    """Nyers pontokat fix meretu pillar-racsba rendez.

    points : (B, N, 3) nyers pontfelho

    Vissza:
        pillars    : (B, P, K, 3) oszloponkent csoportositott pontok
        coords     : (B, P, 2) az oszlop (y, x) racskoordinataja
        num_points : (B, P) hany valodi pont van az oszlopban
        mask       : (B, P) True, ahol az oszlop nem ures
    """
    B, _, C = points.shape
    device = points.device
    x_min, y_min, z_min, x_max, y_max, z_max = pc_range
    vx, vy = voxel_size

    pillars = points.new_zeros((B, max_pillars, max_points, C))
    coords = torch.zeros((B, max_pillars, 2), dtype=torch.long, device=device)
    num_points = torch.zeros((B, max_pillars), dtype=torch.long, device=device)
    mask = torch.zeros((B, max_pillars), dtype=torch.bool, device=device)

    for b in range(B):
        pts = points[b]

        # Csak a hatosugaron beluli pontok erdekelnek.
        keep = ((pts[:, 0] >= x_min) & (pts[:, 0] < x_max)
                & (pts[:, 1] >= y_min) & (pts[:, 1] < y_max)
                & (pts[:, 2] >= z_min) & (pts[:, 2] < z_max))
        pts = pts[keep]
        if pts.numel() == 0:
            continue

        # Melyik cellaba esik minden pont.
        x_idx = ((pts[:, 0] - x_min) / vx).long().clamp(0, grid_w - 1)
        y_idx = ((pts[:, 1] - y_min) / vy).long().clamp(0, grid_h - 1)
        flat = y_idx * grid_w + x_idx

        # A nem-ures cellak, es hogy melyik pont melyikbe tartozik.
        uniq, inverse = torch.unique(flat, return_inverse=True)
        n = min(uniq.numel(), max_pillars)

        # Az oszlopon BELULI sorszam: ez adja a K tengely helyet.
        order = torch.argsort(inverse, stable=True)
        inv_sorted = inverse[order]
        counts = torch.bincount(inv_sorted, minlength=uniq.numel())
        group_start = torch.cumsum(counts, 0) - counts
        rank = torch.arange(inv_sorted.numel(), device=device) - group_start[inv_sorted]

        sel = (inv_sorted < n) & (rank < max_points)
        pillars[b, inv_sorted[sel], rank[sel]] = pts[order[sel]]
        num_points[b, :n] = counts[:n].clamp(max=max_points)
        mask[b, :n] = True
        coords[b, :n, 0] = uniq[:n] // grid_w   # y
        coords[b, :n, 1] = uniq[:n] % grid_w    # x

    return pillars, coords, num_points, mask


def scatter_to_bev(pillar_features, coords, mask, grid_h, grid_w):
    """Az oszlopvektorokat visszateriti a (H, W) meretu ures BEV vasznara.

    (B, P, C) -> (B, C, H, W)
    """
    B, P, C = pillar_features.shape
    canvas = pillar_features.new_zeros((B, C, grid_h * grid_w))
    for b in range(B):
        valid = mask[b]
        if not valid.any():
            continue
        flat = coords[b, valid, 0] * grid_w + coords[b, valid, 1]
        canvas[b, :, flat] = pillar_features[b, valid].t()
    return canvas.view(B, C, grid_h, grid_w)


# =============================================================================
# 2. PILLAR FEATURE NET - mini-PointNet oszloponkent
# =============================================================================
#
# Minden oszlopban K pont van, de EGY vektort akarunk belole. A PointNet
# receptje: kozos sulyu linearis reteg minden pontra kulon, majd max-pool a
# pontok mentén. A max-pool azert jo, mert SORRENDFUGGETLEN - barhogy
# keverjuk a pontokat az oszlopon belul, ugyanaz jon ki. Pont ezt akarjuk,
# hiszen a pontfelhonek nincs termeszetes sorrendje.
# =============================================================================

class PillarFeatureNet(nn.Module):
    """Csoportositott pontok -> oszloponkent egy tanult vektor.

    Minden pontot kibovitunk a paper szerinti extra jellemzokkel:
        x, y, z              - a nyers koordinata
        xc, yc, zc           - elteres az oszlop pontjainak atlagatol
        xp, yp               - elteres az oszlop kozeppontjatol
    Az elso az abszolut helyzetet mondja meg, a masik ketto a lokalis alakot -
    ezek nelkul a halo nem tudna megkulonboztetni egy fal es egy autoteto
    pontjait.
    """

    def __init__(self, out_channels=64, voxel_size=(0.8, 0.8), pc_range=None):
        super().__init__()
        self.vx, self.vy = voxel_size
        self.x_offset = self.vx / 2 + pc_range[0]
        self.y_offset = self.vy / 2 + pc_range[1]

        # 3 nyers + 3 sulypont-elteres + 2 cellakozep-elteres = 8 csatorna.
        self.linear = nn.Linear(8, out_channels, bias=False)
        self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        self.out_channels = out_channels

    def forward(self, pillars, coords, num_points):
        K = pillars.shape[2]
        idx = torch.arange(K, device=pillars.device).view(1, 1, K)
        valid = (idx < num_points.unsqueeze(-1)).unsqueeze(-1).to(pillars.dtype)

        # Elteres az oszlop sajat sulypontjatol.
        denom = num_points.clamp(min=1).to(pillars.dtype).view(*num_points.shape, 1, 1)
        mean = (pillars * valid).sum(dim=2, keepdim=True) / denom
        f_cluster = pillars - mean

        # Elteres a cella geometriai kozeppontjatol.
        x_center = coords[..., 1].to(pillars.dtype) * self.vx + self.x_offset
        y_center = coords[..., 0].to(pillars.dtype) * self.vy + self.y_offset
        f_center = torch.stack([pillars[..., 0] - x_center.unsqueeze(-1),
                                pillars[..., 1] - y_center.unsqueeze(-1)], dim=-1)

        x = torch.cat([pillars, f_cluster, f_center], dim=-1) * valid

        x = self.linear(x)
        B, P, K, C = x.shape
        x = F.relu(self.norm(x.reshape(-1, C)).reshape(B, P, K, C))

        # A kitolto (nem letezo) pontokat nullazzuk, kulonben megnyerhetnek a
        # max-poolt, es kitalalt jellemzot adnanak.
        return (x * valid).max(dim=2)[0]


# =============================================================================
# 3. A MODELL
# =============================================================================
#
# A pillar-frontend utan ez egy sima kep-autoencoder, ugyanaz a felepites,
# mint a kamera AE-nel: Conv2d(kernel=4, stride=2) blokkok felezik a meretet,
# a decoder ConvTranspose2d-vel duplaz vissza.
#
#   (64, 128, 128) -> (C, 64, 64) -> (2C, 32, 32) -> (4C, 16, 16) -> (8C, 8, 8)
#   flatten (8C*8*8) -> Linear -> latent_dim
# =============================================================================

class BEVConvAE(L.LightningModule):
    """Pillar-encoder + konvolucios autoencoder a BEV pszeudo-kepen.

    A rekonstrukcios loss a PSZEUDO-KEPRE megy, nem a nyers pontokra. Ez
    szandekos: a pillar-encoder egyutt tanul a tobbivel, de a cel egy olyan
    latens, amibol a BEV-jellemzoterkep visszaepitheto.

    Tanitas:
        model = BEVConvAE(latent_dim=128)
        loss = model.loss(points)          # (B, N, 3) nyers pontfelho

    RL-ben (csak az encoder kell):
        z = model.encode(points)           # (B, latent_dim)
    """

    def __init__(self, latent_dim: int = 128, base_channels: int = 32,
                 pillar_channels: int = 64, lr: float = 1e-3,
                 latent_scale: float = 4.0, voxel_size: float = 0.8,
                 pc_range: float = 51.2, z_min: float = -5.0, z_max: float = 9.0,
                 max_points_per_pillar: int = 32, max_pillars: int = 6000):
        super().__init__()
        self.save_hyperparameters()

        self.pc_range = (-pc_range, -pc_range, z_min, pc_range, pc_range, z_max)
        self.voxel_size = (voxel_size, voxel_size)
        self.grid_w = int(round(2 * pc_range / voxel_size))
        self.grid_h = self.grid_w

        self.pfn = PillarFeatureNet(pillar_channels, self.voxel_size, self.pc_range)

        C, Cin = base_channels, pillar_channels
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

        # Negy felezes utan ekkora marad. 128x128 bemenetnel ez 8x8.
        self.enc_out_shape = (C * 8, self.grid_h // 16, self.grid_w // 16)
        flat_dim = C * 8 * (self.grid_h // 16) * (self.grid_w // 16)

        self.fc_encode = nn.Linear(flat_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, flat_dim)

        self.conv_decoder = nn.Sequential(
            nn.ConvTranspose2d(C * 8, C * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(C * 4), nn.ReLU(inplace=True),

            nn.ConvTranspose2d(C * 4, C * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(C * 2), nn.ReLU(inplace=True),

            nn.ConvTranspose2d(C * 2, C, 4, 2, 1, bias=False),
            nn.BatchNorm2d(C), nn.ReLU(inplace=True),

            nn.ConvTranspose2d(C, Cin, 4, 2, 1),
        )

    def to_bev(self, points):
        """Nyers pontfelho -> BEV pszeudo-kep. (B, N, 3) -> (B, Cp, H, W)"""
        pillars, coords, num_points, mask = voxelize(
            points, self.pc_range, self.voxel_size, self.grid_w, self.grid_h,
            self.hparams.max_points_per_pillar, self.hparams.max_pillars)
        return scatter_to_bev(self.pfn(pillars, coords, num_points), coords,
                              mask, self.grid_h, self.grid_w)

    def encode_bev(self, bev):
        """Pszeudo-kep -> latens. (B, Cp, H, W) -> (B, latent_dim)"""
        z = self.fc_encode(self.conv_encoder(bev).flatten(1))
        # tanh korlatozza a latenst [-scale, scale] koze, mert az RL
        # observation-space-nek veges hatarai vannak.
        return torch.tanh(z) * self.hparams.latent_scale

    def decode(self, z):
        """Latens -> pszeudo-kep. Csak a tanitashoz kell."""
        h = self.fc_decode(z / self.hparams.latent_scale)
        return self.conv_decoder(h.view(-1, *self.enc_out_shape))

    def encode(self, points):
        """Nyers pontfelho -> latens. EZT hasznalja majd az RL."""
        return self.encode_bev(self.to_bev(points))

    def forward(self, points):
        """Teljes kor: pontfelho -> latens -> rekonstrualt pszeudo-kep."""
        return self.decode(self.encode(points))

    def loss(self, points):
        """MSE a pszeudo-kep es a rekonstrukcioja kozott.

        A pszeudo-kepet csak egyszer szamoljuk ki - a voxelizacio a lassu resz.

        A CEL `.detach()`-elve van, es ez NEM reszletkerdes.
        A `bev` a PillarFeatureNet kimenete, tehat maga is TANULHATO. Ha a cel
        is a gradiensen lognne, a halonak nem kellene megtanulnia
        rekonstrualni: eleg lenne, ha a `pfn` a pszeudo-kepet a NULLA fele
        huzza, mert a nulla celt a decoder trivialisan eltalalja. Az MSE
        raadasul negyzetes, tehat a cel 10%-os zsugoritasa onmagaban ~19%
        loss-csokkenest ad ingyen.

        Merve is ez tortent: 80 lepes alatt a loss 2.63 -> 0.94 esett, mikozben
        a cel atlagos abszolut erteke 0.387 -> 0.338, a szorasa 1.566 -> 1.412
        zsugorodott. Egy teljes tanitason a val loss 0.0001-ig ment le, ami nem
        jo rekonstrukcio, hanem ures latens - pont az, ami az RL-nek
        hasznalhatatlan.

        A `.detach()`-csel a cel egy lepesen belul FIX, tehat nem tud
        elszokni. A `pfn` tovabbra is tanul, csak az encoder feloli agon
        keresztul - ugyanugy, ahogy az eredeti PointPillarsban is a detekcios
        fej tanitja, nem a rekonstrukcio.
        """
        bev = self.to_bev(points)
        return F.mse_loss(self.decode(self.encode_bev(bev)), bev.detach())

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


# =============================================================================
# 4. ONTESZT
# =============================================================================

def test_inference():
    torch.manual_seed(0)

    B, N = 2, 12000
    # A CARLA lidar 50 m-es hatosugaru, a pontok nagyjabol korben szorodnak.
    points = torch.empty(B, N, 3)
    points[..., 0].uniform_(-50, 50)
    points[..., 1].uniform_(-50, 50)
    points[..., 2].uniform_(-5, 8)

    model = BEVConvAE(latent_dim=128).eval()

    with torch.no_grad():
        bev = model.to_bev(points)
        z = model.encode_bev(bev)
        rec = model.decode(z)

    print(f"  bemenet      : {tuple(points.shape)}")
    print(f"  pszeudo-kep  : {tuple(bev.shape)}")
    print(f"  kitoltottseg : {100 * (bev.abs().sum(1) > 0).float().mean():.1f}% cella")
    print(f"  latens       : {tuple(z.shape)}  = {z.shape[1]} ertek / minta")
    print(f"  rekonstrukcio: {tuple(rec.shape)}")
    assert rec.shape == bev.shape, "ALAK ELTERES!"

    # Egy tanito lepes, hogy a backward is ellenorizve legyen.
    model.train()
    opt = model.configure_optimizers()
    loss = model.loss(points)
    opt.zero_grad()
    loss.backward()
    opt.step()

    n = sum(p.numel() for p in model.parameters())
    print(f"  tanito lepes : OK, loss = {float(loss):.4f}")
    print(f"  parameterek  : {n / 1e6:.1f}M")
    print("OK")


if __name__ == "__main__":
    test_inference()
