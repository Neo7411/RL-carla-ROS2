
import gymnasium as gym
import numpy as np
import torch as th
import torch.nn.functional as F
from torch import nn

from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# A szenzor-agak kulcsai. Minden mas obs kulcs automatikusan az "allapot"
# againak megy - igy ha kesobb uj meres kerul az obs-be, nem kell itt nyulni.
CAMERA_KEY = "cam_latent"
LIDAR_KEY = "lidar_latent"


class AzimuthConv2d(nn.Conv2d):
    """3x3 konvolucio, ami csak VIZSZINTESEN (azimut) padel korkorosen.

    Az azimut korbeer: a 0. es az utolso oszlop fizikailag szomszedos (az
    auto mogotti irany). Az elevacio viszont nem: a legfelso sor a horizont,
    a legalso a talaj az auto mellett. A padding_mode="circular" mindket
    tengelyen korbeert, es ezt a ket sort hamisan szomszedda tette.

    Ugyanazt adja, mint a graph_ae.CircularConv2d, de egyetlen pad muvelettel
    (a fuggoleges nulla-paddinget a konvolucio maga vegzi) - merve ~11%-kal
    gyorsabb forward+backward batch 256-on.
    """

    def __init__(self, c_in, c_out, stride):
        super().__init__(c_in, c_out, 3, stride=stride, padding=(1, 0))

    def forward(self, x):
        return self._conv_forward(F.pad(x, (1, 1, 0, 0), mode="circular"), self.weight, self.bias)


class CarlaFusionExtractor(BaseFeaturesExtractor):

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        fusion_dim: int = 256,
        state_dim: int = 64,
        cnn_base_channels: int = 16,
        fusion_mode: str = "add",
    ):
        if fusion_mode not in ("add", "concat"):
            raise ValueError(f"fusion_mode must be 'add' or 'concat', got {fusion_mode!r}")
        super().__init__(observation_space, features_dim=1)

        spaces = observation_space.spaces
        self.fusion_mode = fusion_mode


        lidar_space = spaces[LIDAR_KEY]
        if len(lidar_space.shape) != 3:
            raise ValueError(
                f"{LIDAR_KEY} must be a spatial (C,H,W) feature map, got shape "
                f"{lidar_space.shape}. Ha vektor latenst hasznalsz, a kamera-aggal "
                f"azonos Linear ag kell ide."
            )
        c_in, h, w = lidar_space.shape
        c = cnn_base_channels

        lidar_conv = nn.Sequential(
            # (16, 11, 32) -> (c, 11, 16)
            AzimuthConv2d(c_in, c, stride=(1, 2)),
            nn.GroupNorm(min(8, c), c),
            nn.ReLU(inplace=True),

            # (c, 11, 16) -> (2c, 6, 8)
            AzimuthConv2d(c, c * 2, stride=(2, 2)),
            nn.GroupNorm(min(8, c * 2), c * 2),
            nn.ReLU(inplace=True),

            # (2c, 6, 8) -> (4c, 3, 4)
            AzimuthConv2d(c * 2, c * 4, stride=(2, 2)),
            nn.GroupNorm(min(8, c * 4), c * 4),
            nn.ReLU(inplace=True),

            nn.Flatten(),
        )
        # A lapitott meretet NEM szamoljuk kezzel: megkerdezzuk a halotol. Igy
        # ha a range image merete valtozik a configban, ez a reteg koveti,
        # nem pedig csendben rossz alakot var.
        with th.no_grad():
            flat_dim = int(np.prod(lidar_conv(th.zeros(1, c_in, h, w)).shape[1:]))
        self.lidar_cnn = nn.Sequential(lidar_conv, nn.Linear(flat_dim, fusion_dim))

        # --- kamera ag ------------------------------------------------------
        # A kamera latens mar tanh-olt, fix skalaju vektor - egy Linear eleg
        # ahhoz, hogy a lidar-ag terebe forgassa.
        cam_dim = int(np.prod(spaces[CAMERA_KEY].shape))
        self.cam_fc = nn.Linear(cam_dim, fusion_dim)

        # --- a ket ag osszevezetese ----------------------------------------
        # "concat" eseten egy Linear viszi vissza fusion_dim-re; "add" eseten
        # nincs mit projektalni, mert a ket ag mar azonos hosszu.
        self.fusion_proj = (nn.Linear(fusion_dim * 2, fusion_dim)
                            if fusion_mode == "concat" else nn.Identity())
        # A LayerNorm a fuzio UTAN all: ez rogziti a szenzor-vektor teljes
        # skalajat (a lidar latensnek nincs garantalt korlatja, a graf-encoder
        # vegen nincs tanh). A ket ag EGYMASHOZ valo aranyat nem allitja be:
        # inicializalaskor a kamera-ag ~2.4x jobban mozgatja a vektort (merve,
        # valodi obs-on), mert a lidar bemenet mintarol mintara kevesbe
        # valtozik. Agankenti LayerNorm-mal is csak 2.0x lett - ezt a
        # sulyozast a halo tanulja meg.
        self.fusion_norm = nn.LayerNorm(fusion_dim)

        # --- a tobbi (kis) jel ----------------------------------------------
        # Minden nem-szenzor kulcs ide megy, rogzitett sorrendben (sorted),
        # hogy a vektor osszerakasa determinisztikus legyen.
        self.state_keys = sorted(k for k in spaces if k not in (CAMERA_KEY, LIDAR_KEY))
        # get_flattened_obs_dim: a Discrete-et one-hot merettel szamolja,
        # ugyanugy, ahogy az SB3 preprocess_obs valoban at is alakitja.
        raw_state_dim = sum(get_flattened_obs_dim(spaces[k]) for k in self.state_keys)

        centers, scales = [], []
        for k in self.state_keys:
            sp = spaces[k]
            if isinstance(sp, gym.spaces.Box):
                lo = np.broadcast_to(sp.low, sp.shape).ravel().astype(np.float32)
                hi = np.broadcast_to(sp.high, sp.shape).ravel().astype(np.float32)
                center = (hi + lo) / 2.0
                scale = np.maximum((hi - lo) / 2.0, 1e-6)   # 0 szelessegu tengely ellen
            else:
                # Discrete -> one-hot, az mar 0..1, nem kell skalazni
                center = np.zeros(get_flattened_obs_dim(sp), dtype=np.float32)
                scale = np.ones(get_flattened_obs_dim(sp), dtype=np.float32)
            centers.append(center)
            scales.append(scale)
        # buffer es nem parameter: nem tanuljuk, de a .to(device) viszi magaval
        self.register_buffer("state_center", th.as_tensor(np.concatenate(centers)))
        self.register_buffer("state_scale", th.as_tensor(np.concatenate(scales)))

        self.state_mlp = nn.Sequential(
            nn.Linear(raw_state_dim, state_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(state_dim),
        )

        # A tenyleges kimeneti meret - ezt olvassa ki az SB3 a net_arch ele.
        self._features_dim = fusion_dim + state_dim

    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        lidar_feat = self.lidar_cnn(observations[LIDAR_KEY])
        cam_feat = self.cam_fc(observations[CAMERA_KEY])

        if self.fusion_mode == "add":
            sensor = lidar_feat + cam_feat
        else:
            sensor = self.fusion_proj(th.cat([lidar_feat, cam_feat], dim=1))
        sensor = self.fusion_norm(sensor)

        # A kis jelek: mindegyiket (B, -1)-re lapitjuk, majd egyben az MLP-be.
        # A maneuver-t az SB3 mar one-hot float tenzorkent adja at.
        state_parts = [observations[k].reshape(observations[k].shape[0], -1)
                       for k in self.state_keys]
        # A Box-hatarok szerint [-1,1]-re, hogy a waypointok (+-50) es a
        # sebesseg (0..120) ne nyomjak el a maneuver one-hot jelet.
        state = (th.cat(state_parts, dim=1) - self.state_center) / self.state_scale
        state = self.state_mlp(state)

        return th.cat([sensor, state], dim=1)
