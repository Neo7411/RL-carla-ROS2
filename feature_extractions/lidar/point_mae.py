
import lightning as L
import torch
import torch.nn as nn


def index_points(points, idx):
    """Pontok kigyujtese index szerint. (B, N, C) + (B, S[, K]) -> (B, S[, K], C)"""
    B = points.shape[0]
    batch_idx = torch.arange(B, device=points.device)
    batch_idx = batch_idx.view([B] + [1] * (idx.dim() - 1)).expand_as(idx)
    return points[batch_idx, idx]


def farthest_point_sample(xyz, npoint):
    B, N, _ = xyz.shape
    device = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_idx = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_idx, farthest].view(B, 1, 3)

        distance = torch.minimum(distance, ((xyz - centroid) ** 2).sum(-1))
        farthest = distance.argmax(dim=-1)
    return centroids


def _chamfer_per_sample(pred, gt):
    dist = torch.cdist(pred, gt) ** 2
    per = dist.min(dim=2)[0].mean(dim=1) + dist.min(dim=1)[0].mean(dim=1)
    return per


def chamfer_distance(pred, gt, reduce="mean"):
    per = _chamfer_per_sample(pred, gt)
    return per.mean() if reduce == "mean" else per


class Block(nn.Module):


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
    def __init__(self, num_group: int = 128,
                 group_size: int = 16, embed_dim: int = 192,
                 encoder_depth: int = 6, decoder_depth: int = 2,
                 num_heads: int = 6, mask_ratio: float = 0.6,
                 lr: float = 1e-3, drop: float = 0.0,
                 center_weight: float = 0.0):
        super().__init__()
        self.save_hyperparameters()

        self.embed_local = nn.Sequential(
            nn.Conv1d(3, 128, 1), nn.BatchNorm1d(128),
            nn.ReLU(inplace=True), nn.Conv1d(128, 256, 1))
        self.embed_global = nn.Sequential(
            nn.Conv1d(512, 512, 1), nn.BatchNorm1d(512),
            nn.ReLU(inplace=True), nn.Conv1d(512, embed_dim, 1))

        # A zaro LayerNorm a SKALA miatt kell. A kozeppont nyers, +-50 m-es
        # koordinata, a patch-token viszont a sajat kozeppontjahoz tolt, par
        # meteres alak - merve a pozicio-jel szorasa 10.8x akkora volt, mint a
        # tokene. Az `x + pos` osszeadasnal igy a pozicio elnyomta az alakot
        # (es mivel minden blokkban ujra hozzaadodik, ez 6x ismetlodott).
        # A LayerNorm egy skalara hozza a kettot: 10.8x -> 2.5x.
        self.pos_embed = nn.Sequential(nn.Linear(3, 128), nn.GELU(),
                                       nn.Linear(128, embed_dim),
                                       nn.LayerNorm(embed_dim))
        self.decoder_pos_embed = nn.Sequential(nn.Linear(3, 128), nn.GELU(),
                                               nn.Linear(128, embed_dim),
                                               nn.LayerNorm(embed_dim))

        self.encoder = nn.ModuleList(
            [Block(embed_dim, num_heads, drop=drop) for _ in range(encoder_depth)])
        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.decoder = nn.ModuleList(
            [Block(embed_dim, num_heads, drop=drop) for _ in range(decoder_depth)])
        self.decoder_norm = nn.LayerNorm(embed_dim)

        # Patch-enkent K db 3D koordinatat josol.
        self.rebuild_head = nn.Linear(embed_dim, group_size * 3)

        nn.init.trunc_normal_(self.mask_token, std=0.02)

    # --- patch-eles -------------------------------------------------------

    def group(self, xyz):
        center = index_points(xyz, farthest_point_sample(xyz, self.hparams.num_group))
        # kNN a kozeppontok korul: a legkozelebbi group_size pont.
        idx = torch.cdist(center, xyz).topk(self.hparams.group_size,
                                            dim=-1, largest=False)[1]
        # A patch-et a sajat kozeppontjahoz toljuk: igy a token a lokalis
        # alakot irja le, fuggetlenul attol, hol van a terben.
        return index_points(xyz, idx) - center.unsqueeze(2), center

    def embed(self, patches):
        """Patch-ek -> tokenek. (B, G, K, 3) -> (B, G, embed_dim)"""
        B, G, K, _ = patches.shape
        x = self.embed_local(patches.reshape(B * G, K, 3).transpose(1, 2))
        # A patch globalis jellemzoje minden pont melle: (256 + 256) = 512.
        x = torch.cat([x.max(dim=2, keepdim=True)[0].expand(-1, -1, K), x], dim=1)
        return self.embed_global(x).max(dim=2)[0].reshape(B, G, -1)

    # --- encoder / decoder ------------------------------------------------

    def encode(self, points, return_centers=False):

        patches, center = self.group(points)
        x = self.embed(patches)
        pos = self.pos_embed(center)
        for blk in self.encoder:
            x = blk(x + pos)
        x = self.encoder_norm(x)
        return (x, center) if return_centers else x

    def forward(self, points):
        patches, center = self.group(points)
        tokens = self.embed(patches)
        B, G, C = tokens.shape
        n_mask = int(self.hparams.mask_ratio * G)
        n_vis = G - n_mask
        order = torch.rand(B, G, device=tokens.device).argsort(dim=-1)
        mask = torch.zeros(B, G, dtype=torch.bool, device=tokens.device)
        mask.scatter_(1, order[:, :n_mask], True)

        vis_center = center[~mask].reshape(B, n_vis, 3)
        mask_center = center[mask].reshape(B, n_mask, 3)
        x = tokens[~mask].reshape(B, n_vis, C)
        pos = self.pos_embed(vis_center)
        for blk in self.encoder:
            x = blk(x + pos)
        x = self.encoder_norm(x)
        full = torch.cat([x, self.mask_token.expand(B, n_mask, -1)], dim=1)
        pos_full = self.decoder_pos_embed(torch.cat([vis_center, mask_center], dim=1))
        for blk in self.decoder:
            full = blk(full + pos_full)

        rec = self.decoder_norm(full[:, n_vis:])

        rebuild = self.rebuild_head(rec).reshape(B, n_mask, self.hparams.group_size, 3)
        gt = patches[mask].reshape(B, n_mask, self.hparams.group_size, 3)
        return rebuild, gt, mask, center

    def loss(self, points, parts=False):
        rebuild, gt, _, _ = self(points)
        B, M, K, _ = rebuild.shape
        cd = chamfer_distance(rebuild.reshape(B * M, K, 3), gt.reshape(B * M, K, 3))
        c_pred, c_gt = rebuild.mean(dim=2), gt.mean(dim=2)
        center_loss = ((c_pred - c_gt) ** 2).sum(-1).mean()

        total = cd + self.hparams.center_weight * center_loss
        if not parts:
            return total
        with torch.no_grad():

            spread = float(c_pred.std()) / max(float(c_gt.std()), 1e-6)
            shape = float(rebuild.std()) / max(float(gt.std()), 1e-6)
        return total, {"chamfer": float(cd), "center": float(center_loss),
                       "spread_ratio": spread, "shape_ratio": shape}

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
