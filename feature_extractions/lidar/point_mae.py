"""
Point-MAE - maszkolt autoencoder nyers pontfelhore.

A kamera AE ugy tanul, hogy visszaepiti a sajat bemenetet. Pontfelhonel ez
nem mukodik jol: a pontfelho rendezetlen es egyenetlen surusegu, egy sima
"epitsd vissza az egeszet" feladatot a halo trivialisan megold (eleg
atlagolnia). A Point-MAE trukkje, hogy a pontfelho nagy reszet ELTAKARJA, es
csak a lathato reszbol kell kitalalni a hianyzot - ez mar valodi geometriai
megertest igenyel.

    nyers pontfelho (B, N, 3)
        -> patch-ekre bontas    (B, G, K, 3)   FPS kozeppontok + kNN szomszedok
        -> maszkolas            a patch-ek 60%-a eltunik
        -> Transformer encoder  csak a LATHATO patch-eken fut
        -> Transformer decoder  a hianyzokat epiti vissza
        -> Chamfer loss         a rekonstrualt es a valodi pontok tavolsaga

RL-ben a maszkolas nincs: `encode()` az OSSZES patch-et bekodolja, es a
tokenekbol egy latens vektort pool-oz.

Referencia: Pang et al., "Masked Autoencoders for Point Cloud Self-supervised
Learning" (ECCV 2022). Az FPS / kNN / Chamfer itt tiszta PyTorch, nincs
forditando CUDA kiterjesztes.
"""

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. MINTAVETELI SEGEDFUGGVENYEK
# =============================================================================
#
# Ezek allapotmentesek (nincs tanulhato sulyuk), ezert fuggvenyek, nem
# nn.Module osztalyok.
# =============================================================================

def index_points(points, idx):
    """Pontok kigyujtese index szerint. (B, N, C) + (B, S[, K]) -> (B, S[, K], C)"""
    B = points.shape[0]
    batch_idx = torch.arange(B, device=points.device)
    batch_idx = batch_idx.view([B] + [1] * (idx.dim() - 1)).expand_as(idx)
    return points[batch_idx, idx]


def farthest_point_sample(xyz, npoint):
    """Legtavolabbi pont mintavetel (FPS): egyenletesen szorja a kozeppontokat.

    MIERT NEM VELETLEN MINTAVETEL: a lidar pontsurusege erosen egyenetlen -
    a szenzor kozeleben sokszorosa a tavoli reszekenek. Veletlen mintaval a
    kozeppontok az auto ore tomorulnenek, es a tavoli resz lefedetlen maradna.
    Az FPS mindig azt a pontot valasztja, ami a legtavolabb van az eddig
    kivalasztottaktol, igy a lefedettseg egyenletes lesz.

    xyz : (B, N, 3)  ->  (B, npoint) indexek
    """
    B, N, _ = xyz.shape
    device = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_idx = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_idx, farthest].view(B, 1, 3)
        # Minden pontnal azt tartjuk nyilvan, milyen kozel van a LEGKOZELEBBI
        # eddig kivalasztott kozepponthoz; a kovetkezo valasztott a maximum.
        distance = torch.minimum(distance, ((xyz - centroid) ** 2).sum(-1))
        farthest = distance.argmax(dim=-1)
    return centroids


def knn_point(k, xyz, new_xyz):
    """A new_xyz minden pontjahoz a k legkozelebbi xyz-beli pont indexe."""
    return torch.cdist(new_xyz, xyz).topk(k, dim=-1, largest=False)[1]


def chamfer_distance(pred, gt):
    """Szimmetrikus legkozelebbi-szomszed tavolsag - a CUDA Chamfer helyett.

    MIERT NEM MSE: az MSE osszeparositja az i-edik josolt pontot az i-edik
    valodival, de a pontfelhoben NINCS sorrend - ugyanaz az alak sokfele
    sorrendben leirhato. A Chamfer minden josolt pontot a hozza legkozelebbi
    valodihoz meri (es forditva is), igy sorrendfuggetlen.

    pred (B, N, 3), gt (B, M, 3) -> skalar
    """
    dist = torch.cdist(pred, gt) ** 2
    return dist.min(dim=2)[0].mean() + dist.min(dim=1)[0].mean()


# =============================================================================
# 2. TRANSFORMER BLOKK
# =============================================================================

class Block(nn.Module):
    """Pre-norm Transformer blokk: figyelem + MLP, mindketto reziduallal.

    A nn.MultiheadAttention beepitett - nem kell sajat QKV-projekciot irni.
    """

    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim), nn.Dropout(drop),
        )

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


# =============================================================================
# 3. A MODELL
# =============================================================================

class PointMAE(L.LightningModule):
    """Maszkolt autoencoder pontfelhore.

    Tanitas:
        model = PointMAE(latent_dim=128)
        loss = model.loss(points)          # (B, N, 3) nyers pontfelho

    RL-ben (csak az encoder kell, maszkolas nelkul):
        z = model.encode(points)           # (B, latent_dim)

    num_group  : hany patch (G)
    group_size : pont patch-enkent (K)
    embed_dim  : a Transformer token szelessege
    mask_ratio : a patch-ek mekkora reszet takarjuk el tanitas kozben
    latent_dim : a vegso latens merete - EZ megy az RL observationbe
    """

    def __init__(self, latent_dim: int = 128, num_group: int = 64,
                 group_size: int = 32, embed_dim: int = 192,
                 encoder_depth: int = 6, decoder_depth: int = 2,
                 num_heads: int = 6, mask_ratio: float = 0.6,
                 lr: float = 1e-3, latent_scale: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.save_hyperparameters()

        # --- patch beagyazas: mini-PointNet, (K, 3) patch -> egy token ------
        # Ugyanaz a recept, mint a pillar-encodernel: pontonkenti konvolucio,
        # majd max-pool a pontok mentén (sorrendfuggetlen).
        self.patch_embed = nn.ModuleDict({
            "first": nn.Sequential(nn.Conv1d(3, 128, 1), nn.BatchNorm1d(128),
                                   nn.ReLU(inplace=True), nn.Conv1d(128, 256, 1)),
            "second": nn.Sequential(nn.Conv1d(512, 512, 1), nn.BatchNorm1d(512),
                                    nn.ReLU(inplace=True), nn.Conv1d(512, embed_dim, 1)),
        })

        # --- poziciokodolas: a patch KOZEPPONTJA ---------------------------
        # A patch-eket a sajat kozeppontjukhoz kepest normalizaljuk, igy a
        # token csak a LOKALIS alakot hordozza. Hogy a halo tudja, hol van az
        # a patch a terben, a kozeppontot kulon kodoljuk be.
        self.pos_embed = nn.Sequential(nn.Linear(3, 128), nn.GELU(),
                                       nn.Linear(128, embed_dim))
        self.decoder_pos_embed = nn.Sequential(nn.Linear(3, 128), nn.GELU(),
                                               nn.Linear(128, embed_dim))

        self.encoder = nn.ModuleList(
            [Block(embed_dim, num_heads, drop=drop) for _ in range(encoder_depth)])
        self.encoder_norm = nn.LayerNorm(embed_dim)

        # A maszk-token egyetlen tanult vektor: minden eltakart helyre ez
        # kerul, es a poziciokodolasbol tudja meg, hova kell epitenie.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.decoder = nn.ModuleList(
            [Block(embed_dim, num_heads, drop=drop) for _ in range(decoder_depth)])
        self.decoder_norm = nn.LayerNorm(embed_dim)

        # Patch-enkent K db 3D koordinatat josol.
        self.rebuild_head = nn.Linear(embed_dim, group_size * 3)

        # A tokeneket egy fix meretu latensbe pool-ozzuk az RL szamara.
        # max + mean egyutt: a max a legerosebb jellemzot viszi at, a mean az
        # atlagos tartalmat - ketto egyutt informativabb, mint kulon-kulon.
        self.to_latent = nn.Linear(embed_dim * 2, latent_dim)

        nn.init.trunc_normal_(self.mask_token, std=0.02)

    # --- patch-eles -------------------------------------------------------

    def group(self, xyz):
        """Pontfelho -> patch-ek. (B, N, 3) -> (B, G, K, 3), (B, G, 3)"""
        center = index_points(xyz, farthest_point_sample(xyz, self.hparams.num_group))
        idx = knn_point(self.hparams.group_size, xyz, center)
        # A patch-et a sajat kozeppontjahoz toljuk: igy a token a lokalis
        # alakot irja le, fuggetlenul attol, hol van a terben.
        return index_points(xyz, idx) - center.unsqueeze(2), center

    def embed(self, patches):
        """Patch-ek -> tokenek. (B, G, K, 3) -> (B, G, embed_dim)"""
        B, G, K, _ = patches.shape
        x = patches.reshape(B * G, K, 3).transpose(1, 2)
        feat = self.patch_embed["first"](x)
        feat = torch.cat([feat.max(dim=2, keepdim=True)[0].expand(-1, -1, K), feat], dim=1)
        return self.patch_embed["second"](feat).max(dim=2)[0].reshape(B, G, -1)

    # --- encoder / decoder ------------------------------------------------

    def encode(self, points):
        """Nyers pontfelho -> latens, MASZKOLAS NELKUL. EZT hasznalja az RL."""
        patches, center = self.group(points)
        x = self.embed(patches)
        pos = self.pos_embed(center)
        # A poziciokodolas MINDEN blokkban ujra bekerul (x + pos), nem csak a
        # bemeneten. Igy a mely blokkokban sem halvanyul el, hogy melyik
        # patch hol van a terben. Ez a hivatalos Point-MAE viselkedese.
        for blk in self.encoder:
            x = blk(x + pos)
        x = self.encoder_norm(x)
        z = self.to_latent(torch.cat([x.max(dim=1)[0], x.mean(dim=1)], dim=-1))
        # tanh korlatozza a latenst [-scale, scale] koze, mert az RL
        # observation-space-nek veges hatarai vannak.
        return torch.tanh(z) * self.hparams.latent_scale

    def forward(self, points):
        """Maszkolt kor: a hianyzo patch-ek rekonstrukcioja.

        Vissza: rebuild (B, M, K, 3), gt (B, M, K, 3), mask (B, G), center (B, G, 3)
        ahol M a maszkolt patch-ek szama, es a koordinatak a patch sajat
        kozeppontjahoz kepest ertendok.
        """
        patches, center = self.group(points)
        tokens = self.embed(patches)
        B, G, C = tokens.shape

        # Veletlen maszk: patch-enkent eldontjuk, latszik-e. Minden mintanal
        # ugyanannyi patch tunik el, igy a tenzoralak fix marad.
        n_mask = int(self.hparams.mask_ratio * G)
        order = torch.rand(B, G, device=tokens.device).argsort(dim=-1)
        mask = torch.zeros(B, G, dtype=torch.bool, device=tokens.device)
        mask.scatter_(1, order[:, :n_mask], True)
        n_vis = G - n_mask

        vis_center = center[~mask].reshape(B, n_vis, 3)
        mask_center = center[mask].reshape(B, n_mask, 3)

        # Az encoder CSAK a lathato patch-eket latja - ettol lesz a feladat
        # nehez, es ettol tanul valodi geometriat a halo.
        x = tokens[~mask].reshape(B, n_vis, C)
        pos = self.pos_embed(vis_center)
        for blk in self.encoder:
            x = blk(x + pos)
        x = self.encoder_norm(x)

        # A decoder megkapja a lathato jellemzoket es a maszk-tokeneket; a
        # poziciokodolas mondja meg, melyik maszk-token hova tartozik.
        full = torch.cat([x, self.mask_token.expand(B, n_mask, -1)], dim=1)
        pos_full = self.decoder_pos_embed(torch.cat([vis_center, mask_center], dim=1))
        for blk in self.decoder:
            full = blk(full + pos_full)
        # Csak a maszkolt helyeket epitjuk vissza (azok vannak a vegen).
        rec = self.decoder_norm(full[:, n_vis:])

        rebuild = self.rebuild_head(rec).reshape(B, n_mask, self.hparams.group_size, 3)
        gt = patches[mask].reshape(B, n_mask, self.hparams.group_size, 3)
        return rebuild, gt, mask, center

    def loss(self, points):
        """Chamfer rekonstrukcios loss a maszkolt patch-eken."""
        rebuild, gt, _, _ = self(points)
        B, M, K, _ = rebuild.shape
        return chamfer_distance(rebuild.reshape(B * M, K, 3), gt.reshape(B * M, K, 3))

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

    B, N = 2, 2048
    points = torch.randn(B, N, 3) * 10.0

    model = PointMAE(latent_dim=128).eval()

    with torch.no_grad():
        patches, center = model.group(points)
        rebuild, gt, mask, _ = model(points)
        z = model.encode(points)

    print(f"  bemenet      : {tuple(points.shape)}")
    print(f"  patch-ek     : {tuple(patches.shape)}  kozeppontok {tuple(center.shape)}")
    print(f"  maszkolt     : {int(mask[0].sum())} / {model.hparams.num_group} patch")
    print(f"  rekonstrukcio: {tuple(rebuild.shape)}  cel {tuple(gt.shape)}")
    print(f"  latens       : {tuple(z.shape)}  = {z.shape[1]} ertek / minta")
    assert rebuild.shape == gt.shape, "ALAK ELTERES!"

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
