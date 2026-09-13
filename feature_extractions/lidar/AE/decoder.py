"""
Decoder: latens -> range image.

Forras: lidm/modules/diffusion/model_topolidm.py:169-326 (valtozatlan LiDM decoder)

FONTOS: a decoder NEM graf-alapu, sima ResNet + Upsample. Csak a TANITASHOZ
kell -- o adja a rekonstrukcios loss masik felet. Ha kesz a tanitas, es az
encodert feature extractorkent hasznalod, a decoder eldobhato.
"""

import torch
import torch.nn as nn

from .layers import CircularConv2d


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


# Decoder auxiliary modules: Upsample, ResnetBlock, AttnBlock
UPSAMPLE_STRIDE2KERNEL_DICT = {(1, 2): (1, 5), (1, 4): (1, 7), (2, 1): (5, 1), (2, 2): (3, 3)}
UPSAMPLE_STRIDE2PAD_DICT = {(1, 2): (2, 2, 0, 0), (1, 4): (3, 3, 0, 0), (2, 1): (0, 0, 2, 2), (2, 2): (1, 1, 1, 1)}


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv, stride):
        super().__init__()
        self.with_conv = with_conv
        self.stride = stride
        if self.with_conv:
            k, p = UPSAMPLE_STRIDE2KERNEL_DICT[stride], UPSAMPLE_STRIDE2PAD_DICT[stride]
            self.conv = CircularConv2d(in_channels, in_channels, kernel_size=k, padding=p)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=self.stride, mode='bilinear', align_corners=True)
        if self.with_conv:
            x = self.conv(x)
        return x


UNIFORM_KERNEL2PAD_DICT = {(3, 3): (1, 1, 1, 1), (1, 4): (1, 2, 0, 0)}


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, kernel_size=(3, 3), conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        pad = UNIFORM_KERNEL2PAD_DICT[kernel_size]

        self.norm1 = Normalize(in_channels)
        self.conv1 = CircularConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=pad)

        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)

        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = CircularConv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=pad)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = CircularConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=pad)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


def make_attn(in_channels, attn_type="none"):
    """
    FIGYELEM: az eredetiben ez MINDIG nn.Identity-t ad vissza -- az attention
    nincs implementalva. A config attn_levels: [] miatt ez nem aktiv problema,
    de ne szamits ra, hogy mukodik.
    """
    if attn_type == "none":
        return nn.Identity(in_channels)
    # Expand here for vanilla attention if needed
    return nn.Identity(in_channels)


class Decoder(nn.Module):
    """
    LiDM Decoder, valtozatlan az eredeti implementaciohoz kepest.

    z (B, z_channels, 16, 128) -> (B, out_ch, 64, 1024)
    """
    def __init__(self, *, ch, out_ch, ch_mult, strides, num_res_blocks, attn_levels,
                 dropout=0.0, resamp_with_conv=True, in_channels, z_channels, give_pre_end=False,
                 tanh_out=False, use_linear_attn=False, attn_type="vanilla", use_mask=False,
                 **ignorekwargs):
        super().__init__()
        stride2kernel = {(2, 2): (3, 3), (1, 2): (1, 4)}
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        block_in = ch * ch_mult[self.num_resolutions - 1]

        self.conv_in = CircularConv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            stride = tuple(strides[i_level - 1]) if i_level > 0 else None
            kernel = stride2kernel[stride] if stride is not None else (1, 4)
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, kernel_size=kernel, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if i_level in attn_levels:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if stride is not None:
                up.upsample = Upsample(block_in, resamp_with_conv, stride)
            self.up.insert(0, up)

        self.norm_out = Normalize(block_in)
        self.conv_out = CircularConv2d(block_in, out_ch, kernel_size=(1, 4), stride=1, padding=(1, 2, 0, 0))

    def forward(self, z):
        self.last_z_shape = z.shape
        temb = None

        h = self.conv_in(z)

        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h
