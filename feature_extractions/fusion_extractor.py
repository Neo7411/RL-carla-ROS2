"""
Kozos feature extractor az SAC MultiInputPolicy-hoz.

A MIERT: az SB3 alapertelmezett CombinedExtractora a nem-kep bemeneteket
egyszeruen LELAPITJA es egymas utan fuzi. Nalunk ez katasztrofa lenne:

    lidar_latent   (16, 11, 32)  ->  5632 dim
    cam_latent     (128,)        ->   128 dim
    vehicle        (4,)          ->     4 dim
    waypoints      (15, 2)       ->    30 dim
    maneuver       Discrete(4)   ->     4 dim  (az SB3 one-hot-olja)

vagyis a policy bemenetenek 99.3%-a a lidar lenne, es a kormanyzas szem-
pontjabol legfontosabb 38 szam (sebesseg, szog, waypointok) eltunne benne.
A halo elvileg megtanulhatna lesulyozni, de a gyakorlatban a gradiens a nagy
blokkot koveti, es a kis jelek sose kapnak eselyt.

EZ AZ OSZTALY ezt harom lepesben oldja meg:

 1. A lidar terbeli jellemzoterkepet egy KIS CNN dolgozza fel egy
    fusion_dim hosszu vektorra. Az AE sulyai fagyottak, ez a CNN viszont
    tanul - vagyis a JUTALOMRA optimalizal, es a terbeli szerkezetet meg a
    lapitas elott hasznalja ki.

 2. A kamera latenst egy Linear reteg viszi ugyanarra a fusion_dim-re,
    majd a ket agat OSSZEADJUK - egy kozos "szenzor-vektor" lesz beloluk,
    ami NEM no a modalitasok szamaval. LayerNorm zarja, hogy a ket ag
    skalaja ne csuszhasson el egymastol.

 3. A kis jeleket (vehicle, waypoints, maneuver) egy sajat kis MLP emeli
    state_dim-re, hogy szamossagban is kepesek legyenek versenyezni a
    szenzor-vektorral.

A vegeredmeny: [fusion_dim szenzor | state_dim allapot], ezt kapja a
net_arch (500, 300) MLP.
"""

import gymnasium as gym
import numpy as np
import torch as th
from torch import nn

from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# A szenzor-agak kulcsai. Minden mas obs kulcs automatikusan az "allapot"
# againak megy - igy ha kesobb uj meres kerul az obs-be, nem kell itt nyulni.
CAMERA_KEY = "cam_latent"
LIDAR_KEY = "lidar_latent"


class CarlaFusionExtractor(BaseFeaturesExtractor):
    """
    Dict observation -> egyetlen (B, features_dim) vektor.

    Agak:
        lidar_latent (C,H,W) --CNN----> fusion_dim  \\
                                                     +-> LayerNorm -> fusion_dim
        cam_latent   (D,)    --Linear-> fusion_dim  //

        minden mas obs      --flatten-> MLP -------> state_dim

        features = concat(szenzor, allapot)  ->  (fusion_dim + state_dim)

    Parameterek
    -----------
    fusion_dim : a kozos szenzor-vektor hossza.
    state_dim : a kis jelek (vehicle, waypoints, maneuver) agnak kimenete.
    cnn_base_channels : a lidar CNN SAJAT, tanult szelessege. A harom lepcso
        base, 2*base, 4*base csatornat hasznal (16 -> 32 -> 64). NEM a latens
        csatornaszama: az (z_channels=16) az observation space-bol jon.
    fusion_mode : "add" (elemenkenti osszeadas, ez az alapertelmezes) vagy
        "concat" (egymas melle fuzes, majd egy Linear vissza fusion_dim-re).
    """

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

        # A features_dim-et csak a retegek felepitese utan tudjuk, de az
        # nn.Module.__init__-nek elobb le kell futnia - ezert a dummy 1.
        super().__init__(observation_space, features_dim=1)

        spaces = observation_space.spaces
        self.fusion_mode = fusion_mode

        # --- lidar ag -------------------------------------------------------
        # A bemenet a graf-AE encoderenek kimenete, (16, 11, 32): 11 sor
        # (fuggoleges szog) es 32 oszlop (azimut). Ez NEM kep a szo szokasos
        # ertelmeben - a ket tengely merteke kulonbozik -, ezert:
        #
        #   * padding_mode="circular": az azimut korkoros, a 0. es a 31.
        #     oszlop fizikailag szomszedos. Zero padding hamis "falat" tenne
        #     a jarmu mogotti iranyba.
        #   * Az elso lepcso csak vizszintesen ritkit (stride (1,2)), mert
        #     fuggolegesen eleve csak 11 sor van.
        #   * GroupNorm es nem BatchNorm: az RL batch korrelalt (egy
        #     rolloutbol jon), a rollout kozbeni batch=1 pedig elhasalna.
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
            nn.Conv2d(c_in, c, 3, stride=(1, 2), padding=1, padding_mode="circular"),
            nn.GroupNorm(min(8, c), c),
            nn.ReLU(inplace=True),

            # (c, 11, 16) -> (2c, 6, 8)
            nn.Conv2d(c, c * 2, 3, stride=(2, 2), padding=1, padding_mode="circular"),
            nn.GroupNorm(min(8, c * 2), c * 2),
            nn.ReLU(inplace=True),

            # (2c, 6, 8) -> (4c, 3, 4)
            nn.Conv2d(c * 2, c * 4, 3, stride=(2, 2), padding=1, padding_mode="circular"),
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
        # A LayerNorm a fuzio UTAN all: ez rogziti a szenzor-vektor skalajat,
        # igy a ket ag nem tudja egymast tulharsogni azzal, hogy nagyobb
        # aktivaciokat tanul. A lidar latensnek nincs garantalt korlatja (a
        # graf-encoder vegen nincs tanh), ezert itt kell rendet tenni.
        self.fusion_norm = nn.LayerNorm(fusion_dim)

        # --- a tobbi (kis) jel ----------------------------------------------
        # Minden nem-szenzor kulcs ide megy, rogzitett sorrendben (sorted),
        # hogy a vektor osszerakasa determinisztikus legyen.
        self.state_keys = sorted(k for k in spaces if k not in (CAMERA_KEY, LIDAR_KEY))
        # get_flattened_obs_dim: a Discrete-et one-hot merettel szamolja,
        # ugyanugy, ahogy az SB3 preprocess_obs valoban at is alakitja.
        raw_state_dim = sum(get_flattened_obs_dim(spaces[k]) for k in self.state_keys)

        # A jelek NAGYSAGRENDJE nagyon kulonbozo: a maneuver one-hot 0..1, a
        # waypointok +-50 m, a sebesseg 0..120. Merve: maneuver std 0.43,
        # waypoints std 28.8 - 66x kulonbseg UGYANABBAN a Linear retegben. A
        # maneuver jele igy elveszett (0.0002 hatas az akciora).
        # Ezert minden bemenetet a SAJAT Box-hatarai alapjan [-1,1]-re viszunk.
        # A hatarok a configbol jonnek, nem tanult statisztikabol, tehat ez
        # determinisztikus es nem csuszik el futas kozben.
        centers, scales = [], []
        for k in self.state_keys:
            sp = spaces[k]
            if isinstance(sp, gym.spaces.Box):
                lo = np.broadcast_to(sp.low, sp.shape).ravel().astype(np.float32)
                hi = np.broadcast_to(sp.high, sp.shape).ravel().astype(np.float32)
                c = (hi + lo) / 2.0
                s = np.maximum((hi - lo) / 2.0, 1e-6)   # 0 szelessegu tengely ellen
            else:
                # Discrete -> one-hot, az mar 0..1, nem kell skalazni
                c = np.zeros(get_flattened_obs_dim(sp), dtype=np.float32)
                s = np.ones(get_flattened_obs_dim(sp), dtype=np.float32)
            centers.append(c)
            scales.append(s)
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
