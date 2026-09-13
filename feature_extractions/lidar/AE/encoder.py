"""
Graf-alapu encoder: range image -> latens.

Forras: lidm/modules/diffusion/model_topolidm.py:120-163

EZ A LENYEG. Ez a repo tenyleges ujdonsaga es ez az, ami neked kell:
a range image-et pont-halmazkent kezeli, es graf-konvolucioval tomoriti.

Az eredetihez kepest ket valtozas:
  1. double_z kapcsolo -- VAE-hez (mu + logvar) vagy determinisztikus AE-hez
  2. base_channels a ch-bol is olvashato (lasd a megjegyzest lent)
"""

import torch.nn as nn

from .layers import GraphLayer, PositionalEncoding2D, Stem


class Encoder(nn.Module):
    """
    Range image -> latens Z, plusz a kozbenso graf-feature-ok.

    Adatut:
        (B, C, 64, 1024)
          -> Stem              -> (B, D, 16, 128)
          -> PositionalEncoding
          -> flatten           -> (B, D, 2048)      "pontfelho" alak
          -> GraphLayer x4     -> h1..h4
          -> Conv1d proj       -> (B, z, 2048)
          -> reshape           -> (B, z, 16, 128)

    FIGYELEM (eredeti kod csapdaja): az __init__ base_channels-t var, de a
    TopoLiDM config `ch`-t ad. Az eredetiben ez a **kwargs-ba hullott es
    ELTUNT -- vagyis a ch: 64 beallitas soha nem ert el az encoderig, mindig
    a default maradt. Itt ezt kijavitottam: ha nincs base_channels, de van
    ch, akkor azt hasznaljuk.

    Ugyanigy a ch_mult / strides / num_res_blocks az encodert NEM erinti,
    azok csak a decoder parameterei.
    """
    def __init__(self, in_channels=2, z_channels=16, base_channels=None, k=20,
                 ch=None, double_z=False, **kwargs):
        super().__init__()

        # a config `ch`-t ad, az eredeti kod `base_channels`-t vart
        if base_channels is None:
            base_channels = ch if ch is not None else 64

        self.k = k
        self.double_z = double_z
        self.z_channels = z_channels

        self.stem = Stem(in_channels, base_channels)
        self.pos_enc = PositionalEncoding2D(base_channels)

        self.layer1 = GraphLayer(base_channels, k)
        self.layer2 = GraphLayer(base_channels, k)
        self.layer3 = GraphLayer(base_channels, k)
        self.layer4 = GraphLayer(base_channels, k)

        # double_z esetén 2*z csatorna: az elso fele mu, a masodik logvar
        out_channels = 2 * z_channels if double_z else z_channels
        self.proj = nn.Conv1d(base_channels, out_channels, 1)

    def forward(self, x):
        # 1. Stem downsampling: (B, C, 64, 1024) -> (B, D, 16, 128)
        h = self.stem(x)
        # 2. Inject positional encoding
        h = self.pos_enc(h)

        B, D, H, W = h.shape
        N = H * W

        # 3. Flatten to point set format for graph layers: (B, D, N)
        h = h.view(B, D, N)

        # 4. Hierarchical Graph Encoding
        h1 = self.layer1(h)
        h2 = self.layer2(h1)
        h3 = self.layer3(h2)
        h4 = self.layer4(h3)

        # 5. Project to latent dimension: (B, z, N)  (vagy 2z, ha double_z)
        z = self.proj(h4)

        # 6. Reshape back to 2D: (B, z, 16, 128)
        z = z.view(B, -1, H, W)

        # Return Z and transpose L2/L4 features to point format (B, N, D)
        # (a h2/h4 az eredetiben a topologiai losszhoz kellett; itt csak
        #  azert hagyom bent, hogy a szignatura egyezzen -- nyugodtan
        #  figyelmen kivul hagyhatod)
        return z, h2.transpose(1, 2), h4.transpose(1, 2)
