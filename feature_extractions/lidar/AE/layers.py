"""
Epitokockak a graf-alapu encoderhez.

Forras: lidm/modules/basic.py            (CircularConv2d)
        lidm/modules/diffusion/model_topolidm.py  (knn, get_graph_feature,
                                                   GraphLayer, PositionalEncoding2D, Stem)

A CircularConv2d azert van ide masolva es nem importalva, mert a basic.py a
tetejen importalja a misc_utils-t, ami behuzza a PIL-t, a multiprocessing-et
es egy csomo diffuzios segedfuggvenyt. Maga az osztaly csak torch-ot hasznal.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# -----------------------------------------------------------------------------
# Korkoros konvolucio
# -----------------------------------------------------------------------------

class CircularConv2d(nn.Conv2d):
    """
    Vizszintesen korkoros, fuggolegesen konstans padding.

    Ez a helyes viselkedes 360 fokos LiDAR range image-nel: az azimut korbeer
    (az utolso oszlop szomszedja az elso), az elevacios szogek viszont nem.

    FIGYELEM: ha nem 360 fokos a szenzorod, ez osszeragasztja a jelenet ket
    egymassal NEM szomszedos szelet. Olyankor sima nn.Conv2d kell.
    """
    def __init__(self, *args, **kwargs):
        if 'padding' in kwargs:
            self.is_pad = True
            if isinstance(kwargs['padding'], int):
                h1 = h2 = v1 = v2 = kwargs['padding']
            elif isinstance(kwargs['padding'], tuple):
                h1, h2, v1, v2 = kwargs['padding']
            else:
                raise NotImplementedError
            self.h_pad, self.v_pad = (h1, h2, 0, 0), (0, 0, v1, v2)
            del kwargs['padding']
        else:
            self.is_pad = False

        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        if self.is_pad:
            if sum(self.h_pad) > 0:
                x = nn.functional.pad(x, self.h_pad, mode="circular")  # horizontal pad
            if sum(self.v_pad) > 0:
                x = nn.functional.pad(x, self.v_pad, mode="constant")  # vertical pad
        x = self._conv_forward(x, self.weight, self.bias)
        return x


# -----------------------------------------------------------------------------
# Graf epitese
# -----------------------------------------------------------------------------

def knn(x, k):
    """
    k legkozelebbi szomszed a FEATURE-terben (nem a 3D terben).

    x : (B, C, N)
    return: (B, N, k) indexek

    KOLTSEG: N x N tavolsagmatrix batch-enkent. N=2048-nal ez 4.2M elem
    retegenkent. A GraphLayer minden retegben ujraszamolja (dinamikus graf).
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, k=20, idx=None):
    """
    El-feature-ok epitese: [szomszed - kozep || kozep].

    Ez a DGCNN/EdgeConv alapotlete: nem a szomszed abszolut erteke szamit,
    hanem a KULONBSEG a kozepponthoz kepest (a relativ geometria), plusz a
    kozeppont sajat feature-je.

    x : (B, C, N)
    return: (B, 2C, N, k)
    """
    batch_size, num_dims, num_points = x.size()
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)
    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    # (B, 2C, N, k): edge difference features (neighbor - center) concatenated with center features
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


class GraphLayer(nn.Module):
    """
    EdgeConv blokk (GLiDR / DeepGCN nyoman).

    k-NN -> el-feature -> 1x1 conv -> max-pool a szomszedok folott.
    A max-pool teszi permutacio-invarianssa: mindegy, milyen sorrendben
    jonnek a szomszedok.
    """
    def __init__(self, channels, k=20):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):
        # x: (B, C, N)
        graph_feat = get_graph_feature(x, k=self.k)  # (B, 2C, N, K)
        out = self.conv(graph_feat)                  # (B, C, N, K)
        out = out.max(dim=-1, keepdim=False)[0]      # (B, C, N): max-pool over neighbors
        return out


# -----------------------------------------------------------------------------
# Pozicio es downsampling
# -----------------------------------------------------------------------------

class PositionalEncoding2D(nn.Module):
    """
    2D abszolut szinuszos poziciokodolas.

    Azert kell, mert a flatten utan a graf-retegek elvesztik a terbeli
    informaciot: nem tudjak, melyik "pont" hol volt a range image-en.
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))

    def forward(self, tensor):
        B, C, H, W = tensor.shape
        pos_x = torch.arange(W, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(H, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)

        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(0).repeat(H, 1, 1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1).unsqueeze(1).repeat(1, W, 1)
        emb = torch.cat((emb_x, emb_y), dim=-1).permute(2, 0, 1).unsqueeze(0)

        return tensor + emb[:, :C, :, :].to(tensor.device)


class Stem(nn.Module):
    """
    Bemenet downsamplingje: (B, C, 64, 1024) -> (B, D, 16, 128).

    Harom CircularConv2d: fuggolegesen /4, vizszintesen /8.
    Azert kell, mert 64x1024 = 65536 "pontra" a k-NN kezelhetetlen lenne.
    16x128 = 2048 pont mar megy.

    Az aszimmetria (fuggolegesen /4, vizszintesen /8) azert helyes, mert a
    bemenet maga is aszimmetrikus: 64 elevacios sor all szemben 1024 azimut
    oszloppal.
    """
    def __init__(self, in_dim=2, out_dim=64):
        super().__init__()
        # Three CircularConv2d layers: /4 vertical and /8 horizontal downsampling
        self.conv1 = CircularConv2d(in_dim, out_dim // 4, kernel_size=3, stride=(2, 2), padding=1)
        self.conv2 = CircularConv2d(out_dim // 4, out_dim // 2, kernel_size=3, stride=(2, 2), padding=1)
        self.conv3 = CircularConv2d(out_dim // 2, out_dim, kernel_size=3, stride=(1, 2), padding=1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        return x
