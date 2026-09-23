
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


def _chamfer_per_sample(pred, gt, weight=None):
    if weight is not None:
        w = torch.as_tensor(weight, device=pred.device, dtype=pred.dtype)
        pred, gt = pred * w, gt * w
    dist = torch.cdist(pred, gt) ** 2
    per = dist.min(dim=2)[0].mean(dim=1) + dist.min(dim=1)[0].mean(dim=1)
    return per


def chamfer_distance(pred, gt, reduce="mean", weight=None):
    """Chamfer-tavolsag, opcionalis TENGELY-SULYOZASSAL.

    A `weight` egy 3 elemu vektor, amivel a koordinatak be vannak szorozva a
    tavolsagszamitas elott. Ez azert kell, mert a felho ANIZOTROP - merve a
    szorasa x 12.4 m, y 11.7 m, z 1.81 m, vagyis a magassag tartomanya 7x
    szukebb a vizszintesnel.

    A cdist izotrop: suly nelkul egy 1 m-es z-hiba ugyanannyit nyom, mint egy
    1 m-es x-hiba - csakhogy 1 m z-ben a teljes hasznos tartomany harmada,
    1 m x-ben viszont a szazaleka. A halo igy szinte ingyen elronthatja a
    magassagot, pedig az RL-nek EPP az a lenyeges jel (jardaszegely 0.15 m,
    autoteto 1.5 m, fal 3 m - vizszintesen mind ugyanott van).
    """
    per = _chamfer_per_sample(pred, gt, weight)
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
    """Maszkolt pont-autoencoder.

    A PATCH-LEFEDETTSEG a legfontosabb beallitas. A patch-eles `num_group`
    kozeppontot valaszt, es mindegyik kore a `group_size` legkozelebbi pontot
    gyujti - vagyis osszesen num_group * group_size pontot fed le a 24576-bol.
    A regi 128 x 16 = 2048 ez a pontok 8%-a volt; a maradek 92% sosem kerult be
    semmilyen patch-be.

    Ennek az a kovetkezmenye, hogy a "lokalis" patch nem lokalis: a ritka,
    tavoli regiokban a 16 legkozelebbi szomszed meterekre szorodik szet. MERVE
    (128 x 16): a patch-sugar atlaga 1.9 m, a p95 5.5 m, a maximuma 15.5 m.
    Egy 15 m atmeroju pontfelhot kellett a rebuild_head-nek 16 koordinataval
    eltalalnia - ez tartotta a Chamfert a trivialis megoldas kozeleben.

    MERVE (200 lepes, lr=1e-4, gain a trivialis patch-atlag felett):

        num_group x group_size   lefedettseg   p95 sugar   gain
              128 x 16               8%          5.5 m     +14%
              256 x 32              33%          6.6 m     +41%
              512 x 32              67%          5.8 m     +44%

    A 256 x 32 a valasztas: a 512-es alig jobb, viszont ~2x lassabb es a
    batch-et is feleznie kellene ezen a 8.3 GB-os GPU-n.
    """

    def __init__(self, num_group: int = 256,
                 group_size: int = 32, embed_dim: int = 192,
                 encoder_depth: int = 6, decoder_depth: int = 2,
                 num_heads: int = 6, mask_ratio: float = 0.6,
                 lr: float = 1e-3, drop: float = 0.0,
                 center_weight: float = 0.0,
                 pc_range: float = 50.0, patch_scale: float = 2.0,
                 z_weight: float = 6.0):
        super().__init__()
        self.save_hyperparameters()

        # --- NORMALIZALAS ----------------------------------------------------
        # A masik ket modell mar normalizal (a bev_ae a z-t [0,1]-be viszi, a
        # graph_ae log2-skalan [-1,1]-be); ebbe a modellbe nyers meterek
        # mentek. A mert skalak:
        #
        #   kozeppont szorasa   19.7 m  <- ez ment a pos_embed Linear(3,128)-ba
        #   patch lokalis szoras 1.70 m  <- ebbol josol a rebuild_head
        #
        # MIT NYER EZ, ES MIT NEM. Ablacioval merve (200 lepes, lr=1e-3), a
        # normalizalas a trivialis felett elert nyeresegen NEM valtoztat:
        # 63.7% vele, 63.7% nelkule. Amit valtoztat, az a GRADIENS SKALAJA:
        #
        #                       loss       gradiens-norma
        #   nyers meterben     34 -> 12        29.9
        #   normalizalva      9.6 -> 3.0       13.2
        #
        # Ez azert szamit, mert a train_model clip=5.0-val vag: 29.9-es
        # gradiensnel a lepesek 6x-osan le vannak vagva (a tanulas nagy reszet
        # a vagas hatarozza meg, nem a gradiens iranya), 13.2-nel sokkal
        # kevesbe. A normalizalas tehat a nagyobb LR-t teszi biztonsagossa -
        # es a tenyleges ugrast a 4.5e-6 -> 2e-3 LR hozta (-6% -> +65.7%).
        self.register_buffer("center_scale",
                             torch.tensor(float(pc_range)), persistent=False)
        # A Chamfer tengely-sulya: a z felskalazva, hogy a magassagi hiba
        # aranyosan szamitson. Lasd a chamfer_distance() docstringjet.
        #
        # EZ EGY TUDATOS CSERE, nem ingyen van. Merve (200 lepes, lr=1e-3):
        #
        #              gain a trivialis felett   sulyozatlan Chamfer [m^2]
        #   z_weight=1          +61.5%                    2.95
        #   z_weight=6          +63.7%                    4.15
        #
        # A sulyozatlan (meteres) Chamfer z-suly nelkul JOBB - termeszetesen,
        # hiszen akkor pont azt optimalizaljuk, amit mer. Csakhogy az a metrika
        # a magassagot alig veszi figyelembe (a z tartomanya 7x szukebb), es
        # nekunk EPP az kell: az RL-nek a jardaszegely (0.15 m) es az autoteto
        # (1.5 m) kulonbsege a hasznos jel, vizszintesen mindketto ugyanott van.
        # A `cd_m2` diagnosztika ezert kulon kiirja a sulyozatlan erteket is.
        self.register_buffer("axis_weight",
                             torch.tensor([1.0, 1.0, float(z_weight)]),
                             persistent=False)

        self.embed_local = nn.Sequential(
            nn.Conv1d(3, 128, 1), nn.BatchNorm1d(128),
            nn.ReLU(inplace=True), nn.Conv1d(128, 256, 1))
        self.embed_global = nn.Sequential(
            nn.Conv1d(512, 512, 1), nn.BatchNorm1d(512),
            nn.ReLU(inplace=True), nn.Conv1d(512, embed_dim, 1))

        # A zaro LayerNorm a SKALA miatt kell. Eredetileg a kozeppont nyers,
        # +-50 m-es koordinata volt, a patch-token viszont par meteres lokalis
        # alak - merve a pozicio-jel szorasa 10.8x akkora volt, mint a tokene.
        # Az `x + pos` osszeadasnal igy a pozicio elnyomta az alakot (es mivel
        # minden blokkban ujra hozzaadodik, ez 6x ismetlodott).
        # A bemenet MOST mar normalizalt (a group() osztja pc_range-dzsel), de
        # a LayerNorm marad: a ket ag skalajat igy sem a veletlen szabja meg.
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
        """Patch-eles. Visszaad: (B, G, K, 3) NORMALIZALT lokalis patch-ek es
        (B, G, 3) NORMALIZALT kozeppontok - mindketto ~egysegnyi skalan."""
        center = index_points(xyz, farthest_point_sample(xyz, self.hparams.num_group))
        # kNN a kozeppontok korul: a legkozelebbi group_size pont.
        idx = torch.cdist(center, xyz).topk(self.hparams.group_size,
                                            dim=-1, largest=False)[1]
        # A patch-et a sajat kozeppontjahoz toljuk: igy a token a lokalis
        # alakot irja le, fuggetlenul attol, hol van a terben.
        patches = index_points(xyz, idx) - center.unsqueeze(2)
        # Skalazas: a patch a sajat meretere, a kozeppont a hatosugarra.
        return patches / self.hparams.patch_scale, center / self.center_scale

    def denorm_patches(self, patches):
        """Normalizalt lokalis patch -> meter. A megjelenitesnek kell."""
        return patches * self.hparams.patch_scale

    def denorm_centers(self, center):
        """Normalizalt kozeppont -> meter. A megjelenitesnek kell."""
        return center * self.center_scale

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
        pr, gr = rebuild.reshape(B * M, K, 3), gt.reshape(B * M, K, 3)
        cd = chamfer_distance(pr, gr, weight=self.axis_weight)
        c_pred, c_gt = rebuild.mean(dim=2), gt.mean(dim=2)
        center_loss = ((c_pred - c_gt) ** 2).sum(-1).mean()

        total = cd + self.hparams.center_weight * center_loss
        if not parts:
            return total
        with torch.no_grad():
            spread = float(c_pred.std()) / max(float(c_gt.std()), 1e-6)
            shape = float(rebuild.std()) / max(float(gt.std()), 1e-6)
            # A trivialis megoldas UGYANAZZAL a sullyal - a nyers Chamfer
            # onmagaban nem mond semmit, csak a baseline-hoz kepest.
            base = chamfer_distance(gr.mean(1, keepdim=True).expand_as(gr),
                                    gr, weight=self.axis_weight)
            gain = 100.0 * (1.0 - float(cd) / max(float(base), 1e-9))
            # A sulyozatlan Chamfer METERBEN, hogy legyen fizikai jelentese.
            s = self.hparams.patch_scale
            cd_m = float(chamfer_distance(pr * s, gr * s))
        return total, {"chamfer": float(cd), "center": float(center_loss),
                       "spread_ratio": spread, "shape_ratio": shape,
                       "cd_m2": cd_m, "gain_pct": gain}

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
