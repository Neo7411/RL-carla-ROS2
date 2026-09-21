

import lightning as L
import torch
import torch.nn as nn



BEV_CHANNELS = 3

class BEVConvAE(L.LightningModule):
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
        bev = self.to_bev(points)
        rec = self.decode(self.encode_bev(bev))

        occ_t, occ_p = bev[:, 0], rec[:, 0]
        p = occ_p.clamp(1e-6, 1 - 1e-6)


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
            m_int = inter_c.unsqueeze(1).float()
            span = self.hparams.z_max - self.hparams.z_min
            z_mae = ((rec[:, 1:] - bev[:, 1:]).abs() * m_int).sum() / (
                m_int.sum().clamp(min=1) * (BEV_CHANNELS - 1)) * span / 0.95
        return total, {"occ_bce": float(occ_loss), "geo_mse": float(geo_loss),
                       "occ_prec": float(prec), "occ_rec": float(rec_r),
                       "z_mae_m": float(z_mae)}
        
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
