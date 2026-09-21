"""
Point-MAE - maszkolt autoencoder nyers pontfelhore.

A kamera AE ugy tanul, hogy visszaepiti a sajat bemenetet. Pontfelhonel ez nem
mukodik jol: a felho rendezetlen es egyenetlen surusegu, egy sima "epitsd
vissza az egeszet" feladatot a halo trivialisan megold. A Point-MAE trukkje,
hogy a felho nagy reszet ELTAKARJA, es csak a lathatobol kell kitalalni a
hianyzot - ez mar valodi geometriai megertest igenyel.

    nyers pontfelho (B, N, 3)
        -> patch-ekre bontas    (B, G, K, 3)   FPS kozeppontok + kNN szomszedok
        -> maszkolas            a patch-ek 60%-a eltunik
        -> Transformer encoder  csak a LATHATO patch-eken fut
        -> Transformer decoder  a hianyzokat epiti vissza
        -> Chamfer loss         a rekonstrualt es a valodi pontok tavolsaga

RL-ben a maszkolas nincs: `encode()` az OSSZES patch-et bekodolja, es a
TOKENEKET adja vissza (B, G, embed_dim) - nincs pooling, lasd encode().

A harom lidar-modell kozul EZ a leggyorsabb (merve 366 frame/s, 0.71 GB batch
8-on), mert a Transformer csak G darab tokenen dolgozik, nem a teljes racson.

Referencia: Pang et al., "Masked Autoencoders for Point Cloud Self-supervised
Learning" (ECCV 2022). Az FPS / kNN / Chamfer itt tiszta PyTorch, nincs
forditando CUDA kiterjesztes.
"""

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
    """Legtavolabbi pont mintavetel (FPS): egyenletesen szorja a kozeppontokat.

    MIERT NEM VELETLEN MINTAVETEL: a lidar pontsurusege erosen egyenetlen - a
    szenzor kozeleben sokszorosa a tavoliakenak (merve a pontok atlagos
    tavolsaga 21.8 m, de a felhonek van egy suru magja az auto korul).
    Veletlen mintaval a kozeppontok odatomorulnenek, es a tavoli resz
    lefedetlen maradna. Az FPS mindig azt a pontot valasztja, ami a
    legtavolabb van az eddig kivalasztottaktol.

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


def _chamfer_per_sample(pred, gt):
    """Szimmetrikus legkozelebbi-szomszed tavolsag - a CUDA Chamfer helyett.

    MIERT NEM MSE: az MSE osszeparositja az i-edik josolt pontot az i-edik
    valodival, de a pontfelhoben NINCS sorrend - ugyanaz az alak sokfele
    sorrendben leirhato. A Chamfer minden josolt pontot a hozza legkozelebbi
    valodihoz meri (es forditva is), igy sorrendfuggetlen.

    pred (B, N, 3), gt (B, M, 3) -> skalar

    reduce="mean" : egyetlen skalar az egesz batchre (ez kell a tanitashoz).
    reduce="none" : (B,) - patch-enkenti ertek, a diagnosztikahoz.
    """
    dist = torch.cdist(pred, gt) ** 2
    per = dist.min(dim=2)[0].mean(dim=1) + dist.min(dim=1)[0].mean(dim=1)
    return per


def chamfer_distance(pred, gt, reduce="mean"):
    """Lasd _chamfer_per_sample. A visszafele kompatibilitas miatt kulon."""
    per = _chamfer_per_sample(pred, gt)
    return per.mean() if reduce == "mean" else per


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
        model = PointMAE()
        loss = model.loss(points)          # (B, N, 3) nyers pontfelho

    RL-ben (csak az encoder kell, maszkolas nelkul):
        z = model.encode(points)           # (B, num_group, embed_dim)

    num_group  : hany patch (G)
    group_size : pont patch-enkent (K)
    embed_dim  : a Transformer token szelessege
    mask_ratio : a patch-ek mekkora reszet takarjuk el tanitas kozben

    Az encode() TOKENEKET ad vissza, nem vektort - az RL sajat
    feature-extractora dolgozza fel oket (lasd encode()).

    A G es K ALAPERTEKE MERT: 200 lepes overfit 16 valodi CARLA-frame-en,
    azonos lefedettseg (G*K / 8192) mellett:

        G=64,  K=32   Chamfer 3.73   patch sugar mediana 4.40 m   18 s
        G=128, K=16   Chamfer 2.52   patch sugar mediana 2.88 m   28 s
        G=128, K=32   Chamfer 3.20   patch sugar mediana 4.08 m   49 s

    A tobb, de kisebb patch a jobb: a CARLA-felho ritkabb, mint a Point-MAE
    eredeti (surun mintavett targy-) adata, ezert K=32-vel egy "patch" mar
    nem lokalis folt, hanem fel utca - az ilyen patch alakjat nem lehet
    ertelmesen rekonstrualni a szomszedaibol.
    """

    def __init__(self, num_group: int = 128,
                 group_size: int = 16, embed_dim: int = 192,
                 encoder_depth: int = 6, decoder_depth: int = 2,
                 num_heads: int = 6, mask_ratio: float = 0.6,
                 lr: float = 1e-3, drop: float = 0.0,
                 center_weight: float = 0.0):
        super().__init__()
        self.save_hyperparameters()

        # --- patch beagyazas: mini-PointNet, (K, 3) patch -> egy token ------
        # Pontonkenti konvolucio, majd max-pool a pontok menten
        # (sorrendfuggetlen). Ket menetben: az elso utani globalis max-ot
        # visszafuzzuk minden ponthoz, igy a masodik menet mar latja a patch
        # egeszet is, nem csak az egyes pontokat.
        self.embed_local = nn.Sequential(
            nn.Conv1d(3, 128, 1), nn.BatchNorm1d(128),
            nn.ReLU(inplace=True), nn.Conv1d(128, 256, 1))
        self.embed_global = nn.Sequential(
            nn.Conv1d(512, 512, 1), nn.BatchNorm1d(512),
            nn.ReLU(inplace=True), nn.Conv1d(512, embed_dim, 1))

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

        nn.init.trunc_normal_(self.mask_token, std=0.02)

    # --- patch-eles -------------------------------------------------------

    def group(self, xyz):
        """Pontfelho -> patch-ek. (B, N, 3) -> (B, G, K, 3), (B, G, 3)

        A patch-ek MERETE az adattol fugg, nem fix: a CARLA-felhon merve a
        sugaruk medianja 4.4 m, a p95 11.4 m. Ez tobb, mint egy klasszikus
        "lokalis folt" - a ritka, tavoli reszeken egy patch fel utcat atfog.
        """
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
        """Nyers pontfelho -> TOKENEK, maszkolas nelkul. EZT hasznalja az RL.

        return : (B, num_group, embed_dim) - patch-enkent egy token
                 return_centers=True eseten (tokenek, kozeppontok)

        NINCS TOBBE POOLING. Korabban egy max+mean pooling vonta ossze a
        128 tokent egyetlen vektorra - ott veszett el az informacio nagy
        resze. A bev_ae-n es a graph_ae-n ugyanezt mertuk: a latens MERETE
        nem szamit, a SZERKEZET elvesztese a problema.

        A tokenek halmaza sorrendfuggetlen (a Transformer permutacio-
        ekvivarians), ezert az RL feature-extractoranak is annak kell lennie:
        pontonkenti MLP + pooling, vagy egy kis attention-fej. A kozeppontok
        (return_centers) megmondjak, MELYIK token HOL van a terben - ezek
        nelkul a halmaz pozicio-informacio nelkul marad.
        """
        patches, center = self.group(points)
        x = self.embed(patches)
        pos = self.pos_embed(center)
        # A poziciokodolas MINDEN blokkban ujra bekerul (x + pos), nem csak a
        # bemeneten. Igy a mely blokkokban sem halvanyul el, hogy melyik
        # patch hol van a terben. Ez a hivatalos Point-MAE viselkedese.
        for blk in self.encoder:
            x = blk(x + pos)
        x = self.encoder_norm(x)
        return (x, center) if return_centers else x

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
        n_vis = G - n_mask
        order = torch.rand(B, G, device=tokens.device).argsort(dim=-1)
        mask = torch.zeros(B, G, dtype=torch.bool, device=tokens.device)
        mask.scatter_(1, order[:, :n_mask], True)

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

    def loss(self, points, parts=False):
        """Chamfer rekonstrukcios loss a maszkolt patch-eken.

        MIERT NEM MSE - ezt KIMERTUK, nem elmelet:
        Volt egy teljes MSE-s futas (30 epoch, ugyanez az adat). Eredmeny:

          tanitas      | 30 epoch alatt | ugyanaz a modell Chamferrel merve
          -------------|----------------|----------------------------------
          Chamfer      | 4.6 -> 2.51    | 2.51
          MSE          | 4.43 -> 3.73   | 10.51   <- 4x rosszabb geometria

        Az MSE-s modellnel kulon megmerve:
          atlagos tavolsag a LEGKOZELEBBI valodi ponthoz : 0.84 m
          atlagos tavolsag az AZONOS INDEXU ponthoz      : 2.70 m

        Tehat a halo nagyjabol eltalalja, HOL vannak a pontok (0.84 m), de az
        MSE nem ezt meri, hanem azt, hogy a k-adik kiirt pont mennyire van
        kozel a k-adik valodihoz (2.70 m). A tanulas nagy resze arra ment el,
        hogy a SORRENDET probalja eltalalni - ami a pontfelhonel megoldhatatlan
        reszfeladat, mert a felho rendezetlen. Ezert lapos a gorbe: 30 epoch
        alatt 16% javulas a Chamfer 45%-aval szemben.

        A Chamfer minden josolt pontot a hozza LEGKOZELEBBI valodihoz meri (es
        forditva is), igy sorrendfuggetlen - pont azt bunteti, ami szamit.

        A FELADAT NEM DEGENERALT - ez is merve van. Trivialis megoldasok
        Chamfer-erteke a valodi adaton:

            mindig nullat josolni       23.60
            egy MASIK patch-et josolni  29.13
            a patch ATLAGAT josolni     13.03   <- a legjobb trivialis
            tanitatlan halo             16.96
            16 mintan overfitelve        2.68

        A "masik patch" 29.13-as erteke a lenyeg: a patch-ek erdemben
        kulonboznek egymastol, tehat van mit tanulni. (Osszehasonlitaskent a
        bev_ae regi valtozatanal a cel maga is tanult, es a halo a nullazassal
        ki tudott bujni a feladat alol - itt ez nem lehetseges, mert a cel a
        nyers pontfelho.)

        A CHAMFER JO, DE EGYMAGABAN NEM MUTATJA MEG, HA A MODELL KOLLAPSZAL.
        Az elmentett ckpt-t megmerve:

            tanitott modell   4.7448
            patch-atlag       4.7909   <- 1% kulonbseg

        Vagyis az a futas alig tudott tobbet annal, mint hogy minden patch-re
        a sajat kozeppontjat adja vissza. A lossbol ez nem latszott, mert a
        Chamfer atlagol: egy "atlagos patch" mar elfogadhato erteket ad.
        Ugyanazon a ckpt-n a kollapszus viszont egyertelmu:

            josolt patch-kozeppontok szorasa   0.152
            valodi ugyanez                     1.068   <- 7x

        EZERT van a parts=True: a spread_ratio (josolt/valodi kozeppont-
        szoras) kozvetlenul meri a kollapszust. 1.0 korul egeszseges, 0 fele
        kollapszus. Tanitas kozben EZT kell nezni, nem csak a Chamfert.

        Megjegyzes: a fenti 4.74-es ckpt egy ELROMLOTT futas. Friss halo
        ugyanezzel a lossal 4 epoch alatt 3.16-ot er el (31.8%-kal a patch-
        atlag alatt), tehat maga a loss nem volt hibas.

        center_weight : opcionalis kozeppont-tag, ami kozvetlenul bunteti, ha
            a josolt patch nem a helyere kerul. MERVE (friss halo, 4 epoch):

              c_w   chamfer  baseline  nyeres  spread  shape
              0.0    3.1615    4.6342   31.8%   0.211   0.486   <- alapertek
              0.5    3.1697    4.6342   31.6%   0.350   0.535
              1.0    3.2448    4.6342   30.0%   0.395   0.545
              2.0    3.2760    4.6342   29.3%   0.442   0.561
              5.0    3.3776    4.6342   27.1%   0.518   0.582

            A tag a kollapszust valoban oldja (spread 0.21 -> 0.52), de a
            Chamfert rontja. Ezert alapbol KI van kapcsolva; akkor erdemes
            bekapcsolni, ha a spread_ratio tartosan 0.2 alatt ragad.

        parts=True : (loss, dict) - a kollapszus merosszamaival egyutt.
        """
        rebuild, gt, _, _ = self(points)
        B, M, K, _ = rebuild.shape
        cd = chamfer_distance(rebuild.reshape(B * M, K, 3), gt.reshape(B * M, K, 3))

        # Patch-enkenti kozeppont: a lokalis alak sulypontja. A valodi patch
        # a sajat kozeppontjahoz van tolva, de a sulypontja NEM pontosan 0 -
        # ez hordozza, hogy a patch merre "logg ki" a kozeppontjabol.
        c_pred, c_gt = rebuild.mean(dim=2), gt.mean(dim=2)
        center_loss = ((c_pred - c_gt) ** 2).sum(-1).mean()

        total = cd + self.hparams.center_weight * center_loss
        if not parts:
            return total
        with torch.no_grad():
            # A kollapszus merteke: a josolt es a valodi kozeppontok szorasa
            # a patch-ek KOZOTT. Az arany 1.0 korul egeszseges, 0 fele
            # kollapszus. EZ az a szam, ami a regi lossbol hianyzott.
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


# =============================================================================
# 4. ONTESZT
# =============================================================================

def test_inference():
    torch.manual_seed(0)

    B, N = 2, 2048
    points = torch.randn(B, N, 3) * 10.0

    model = PointMAE().eval()

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
