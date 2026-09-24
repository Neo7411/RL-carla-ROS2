import json
import math

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam

from carla_env.wrappers import vector, get_displacement_vector, points_to_bev_rgb
from feature_extractions.camera.camera_ae import CameraAutoEncoder
from feature_extractions.lidar.graph_ae import (
    DDCONFIG, LidarAE, points_to_range_image, range_to_points,
)


def write_json(data, path):
    config_dict = {}
    with open(path, 'w', encoding='utf-8') as f:
        for k, v in data.items():
            if isinstance(v, str) and v.isnumeric():
                config_dict[k] = int(v)
            elif isinstance(v, dict):
                config_dict[k] = dict()
                for k_inner, v_inner in v.items():
                    config_dict[k][k_inner] = v_inner.__str__()
                config_dict[k] = str(config_dict[k])
            else:
                config_dict[k] = v.__str__()
        json.dump(config_dict, f, indent=4)


class HParamCallback(BaseCallback):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def _on_training_start(self) -> None:
        hparam_dict = {}
        for k, v in self.config.items():
            if isinstance(v, str) and v.isnumeric():
                hparam_dict[k] = int(v)
            elif isinstance(v, dict):
                hparam_dict[k] = dict()
                for k_inner, v_inner in v.items():
                    hparam_dict[k][k_inner] = v_inner.__str__()
                hparam_dict[k] = str(hparam_dict[k])
            else:
                hparam_dict[k] = v.__str__()
        metric_dict = {
            "rollout/ep_len_mean": 0,
            "train/value_loss": 0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        if self.locals['dones'][0]:
            self.logger.record("custom/total_reward", self.locals['infos'][0]['total_reward'])
            self.logger.record("custom/routes_completed", self.locals['infos'][0]['routes_completed'])
            self.logger.record("custom/total_distance", self.locals['infos'][0]['total_distance'])
            self.logger.record("custom/avg_speed", self.locals['infos'][0]['avg_speed'])
            self.logger.record("custom/mean_reward", self.locals['infos'][0]['mean_reward'])
            self.logger.dump(self.num_timesteps)
        return True


class AttentionCallback(BaseCallback):
    """Mire figyel a policy? Permutacios fontossag tanitas kozben.

    every lepesenkent vesz egy batch-et a replay bufferbol, es megnezi, mennyit
    valtozik az actor determinisztikus akcioja, ha EGY bemenet-csoportot a
    batch-en belul osszekeverunk (a tobbi marad a helyen). Amit a policy nem
    hasznal, annal a valtozas ~0. Env-egysegben merunk: kormany -1..1, gaz 0..1.

    TensorBoard:
      attention/steer/<csoport>        atlagos |d kormany|
      attention/steer_share/<csoport>  az egyedi csoportok kozti aranya [%]
      attention/throttle/..., attention/throttle_share/...
      attention/steer_sensors_vs_route (kamera+lidar) / (waypoints+szog+maneuver)
                                       a kormanyon; 1 alatt: tobbet ad a route-ra

    A mintavetel sajat RNG-vel megy: a globalis np.random-ot a route-sorsolas
    (_sample_route) is hasznalja, azt nem akarjuk elallitani.
    """

    # csoport -> [(obs kulcs, oszlop vagy None = az egesz kulcs)]
    # vehicle_measures oszlopai: 0 steer, 1 throttle, 2 speed, 3 szog a waypointhoz
    SINGLE = {
        "cam": [("cam_latent", None)],
        "lidar": [("lidar_latent", None)],
        "waypoints": [("waypoints", None)],
        "angle": [("vehicle_measures", 3)],
        "maneuver": [("maneuver", None)],
        "speed": [("vehicle_measures", 2)],
        "prev_action": [("vehicle_measures", 0), ("vehicle_measures", 1)],
    }
    COMBINED = {
        "sensors": SINGLE["cam"] + SINGLE["lidar"],
        "route": SINGLE["waypoints"] + SINGLE["angle"] + SINGLE["maneuver"],
    }

    def __init__(self, every=10_000, batch_size=2048, seed=0, verbose=1):
        super().__init__(verbose)
        self.every = every
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

    def _sample_obs(self):
        rb = self.model.replay_buffer
        upper = rb.buffer_size if rb.full else rb.pos
        if upper < 256:
            return None
        idx = self.rng.integers(0, upper, size=self.batch_size)
        return {k: rb.to_torch(v[idx, 0]) for k, v in rb.observations.items()}

    @torch.no_grad()
    def _act(self, obs):
        return self.model.actor(obs, deterministic=True)

    @staticmethod
    def _permuted(obs, parts, perm):
        o = dict(obs)
        for key, col in parts:
            if col is None:
                o[key] = obs[key][perm]
            else:
                if o[key] is obs[key]:
                    o[key] = obs[key].clone()
                o[key][:, col] = obs[key][perm, col]
        return o

    def _on_step(self) -> bool:
        if self.num_timesteps % self.every != 0:
            return True
        obs = self._sample_obs()
        if obs is None:
            return True

        # Az actor [-1, 1]-be squash-olt akciot ad; env-egysegre valtjuk.
        space = self.model.action_space
        scale = torch.as_tensor((space.high - space.low) / 2.0, device=self.model.device)
        base = self._act(obs)
        perm = torch.as_tensor(self.rng.permutation(len(base)), device=base.device)

        diff = {}
        for name, parts in {**self.SINGLE, **self.COMBINED}.items():
            d = ((self._act(self._permuted(obs, parts, perm)) - base).abs() * scale).mean(0)
            diff[name] = d.cpu().numpy()

        total = sum(diff[n] for n in self.SINGLE) + 1e-9
        for i, act in enumerate(("steer", "throttle")):
            for name, d in diff.items():
                self.logger.record(f"attention/{act}/{name}", float(d[i]))
            for name in self.SINGLE:
                self.logger.record(f"attention/{act}_share/{name}", float(100.0 * diff[name][i] / total[i]))
        ratio = float(diff["sensors"][0] / (diff["route"][0] + 1e-9))
        self.logger.record("attention/steer_sensors_vs_route", ratio)
        self.logger.dump(self.num_timesteps)

        if self.verbose:
            shares = "  ".join(f"{n} {100.0 * diff[n][0] / total[0]:4.0f}%" for n in self.SINGLE)
            print(f"[attention] {self.num_timesteps} | kormany: {shares} | "
                  f"szenzor/route = {ratio:.2f}", flush=True)
        return True


def lr_schedule(initial_value: float, end_value: float, rate: float):
    def func(progress_remaining: float) -> float:
        if progress_remaining <= 0:
            return end_value
        return end_value + (initial_value - end_value) * (10 ** (rate * math.log10(progress_remaining)))

    func.__str__ = lambda: f"lr_schedule({initial_value}, {end_value}, {rate})"
    lr_schedule.__str__ = lambda: f"lr_schedule({initial_value}, {end_value}, {rate})"
    return func


# =============================================================================
# AE betoltok
# =============================================================================

# Camera autoencoder load 
def load_cam_ae(cfg, device):
    """Kamera AE. A latent_dim es a latent_scale a checkpointbol jon, nem a
    configbol - igy nem lehet elcsuszni a tanitott halotol."""
    ck = torch.load(cfg["ckpt"], map_location="cpu", weights_only=False)
    cam_ae = CameraAutoEncoder(**ck["hparams"])
    cam_ae.load_state_dict(ck["state_dict"])
    cam_ae.eval().to(device)

    print(f"Loaded camera AE from {cfg['ckpt']}, "
          f"latent_dim {cam_ae.hparams.latent_dim}, "
          f"latent_scale {cam_ae.hparams.latent_scale}")
    return cam_ae

# Lidar autoencoder load
def load_lidar_ae(cfg, device):
    """Lidar AE. Sima nn.Module, ezert kell a DDCONFIG a peldanyositashoz."""
    ck = torch.load(cfg["ckpt"], map_location="cpu", weights_only=False)
    lidar_ae = LidarAE(DDCONFIG, **ck["hparams"])
    lidar_ae.load_state_dict(ck["best_state"])
    lidar_ae.eval().to(device)

    # A latens alakja a range image meretebol kovetkezik (a stem 4-gyel,
    # illetve 8-cal oszt), ezert NEM hardkodoljuk: megkerdezzuk a halotol.
    h, w = cfg["range_image"]["size"]
    with torch.no_grad():
        latent_shape = tuple(lidar_ae.encode(torch.zeros(1, 1, h, w, device=device)).shape[1:])

    print(f"Loaded lidar AE from {cfg['ckpt']}, latent shape {latent_shape}")
    return lidar_ae, latent_shape


# =============================================================================
# Observation
# =============================================================================
def create_encode_state_fn(cam_ae, lidar_ae, cfg, device):
    ri_params = cfg["lidar"]["range_image"]
    bev_params = dict(fov=ri_params["fov"], depth_scale=ri_params["depth_scale"],
                      depth_range=ri_params["depth_range"])

    @torch.no_grad()
    def encode_state(env):
        # dict for current CARLA state
        encoded_state = {}

        # SORREND: elobb minden bemenet a GPU-ra, aztan minden halo elindul,
        # es csak a vegen olvasunk vissza. Minden .cpu() szinkronizal - ha
        # minden halo utan visszaolvasunk, a CPU es a GPU felvaltva var
        # egymasra. Merve: 14.5 -> 8.4 ms/step, a latensek bitre azonosak.

        # Nyers RGB kep -> AE latent. A normalizalas ugyanaz, mint a
        # tanitasban: uint8/255 es (H,W,C) -> (C,H,W). A /255 azert fut a
        # CPU-n: a GPU-s osztas 1 ULP-vel mas eredmenyt ad, es a latens is
        # elmozdulna (~6e-5).
        image = np.ascontiguousarray(env.observation, dtype=np.uint8)
        x = torch.from_numpy(image).permute(2, 0, 1).float().div_(255.0)
        x = x.unsqueeze(0).to(device)                # (1, 3, 80, 160)

        # Nyers pontfelho -> range image -> TERBELI jellemzoterkep. A range
        # image parametereinek egyezniuk kell a tanitasiakkal, kulonben a
        # latens ertelmetlen - ezert jonnek a configbol.
        ri = points_to_range_image(env.lidar_points, **ri_params)
        xl = torch.from_numpy(ri).unsqueeze(0).to(device)   # (1, 1, H, W)

        z = cam_ae.encode(x)
        zl = lidar_ae.encode(xl)

        # A rekonstrukciok csak a megjelenitesnek kellenek (env.render rakja
        # ki oket) - renderelés nelkul felesleges dekodolni.
        if env.activate_render:
            cam_recon = (cam_ae.decode(z)[0].clamp(0, 1) * 255).byte().permute(1, 2, 0).contiguous()
            lidar_recon = lidar_ae.decode(zl)[0]
            # BEV a HUD-hoz: amit a halo kap. CPU munka, a GPU kozben dolgozik.
            env.lidar_bev_input = points_to_bev_rgb(range_to_points(ri, **bev_params))

        encoded_state['cam_latent'] = z[0].cpu().numpy().astype(np.float32)
        encoded_state['lidar_latent'] = zl[0].cpu().numpy().astype(np.float32)

        if env.activate_render:
            env.ae_reconstruction = cam_recon.cpu().numpy()
            # ...es amit visszaad
            env.lidar_bev_recon = points_to_bev_rgb(
                range_to_points(lidar_recon.cpu().numpy(), **bev_params))

        vehicle_measures = []

        # ask current vechile measures steer, throttle, speed, angle, waypoint
        vehicle_measures.append(env.vehicle.control.steer)
        vehicle_measures.append(env.vehicle.control.throttle)
        vehicle_measures.append(env.vehicle.get_speed())
        vehicle_measures.append(env.vehicle.get_angle(env.current_waypoint))

        # Append to dict
        encoded_state['vehicle_measures'] = vehicle_measures

        # actual vehicle maneuver
        encoded_state['maneuver'] = env.current_road_maneuver.value
        next_waypoints_state = env.route_waypoints[env.current_waypoint_index: env.current_waypoint_index + 15]
        waypoints = [vector(way[0].transform.location) for way in next_waypoints_state]
        vehicle_location = vector(env.vehicle.get_location())
        theta = np.deg2rad(env.vehicle.get_transform().rotation.yaw)
        relative_waypoints = np.zeros((15, 2))
        for i, w_location in enumerate(waypoints):
            relative_waypoints[i] = get_displacement_vector(vehicle_location, w_location, theta)[:2]
        if len(waypoints) < 15:
            # A route vege utan az utolso szakasz iranyaban egyenesen tovabb.
            # A szakaszt a route utolso ket pontjabol vesszuk, nem a maradek
            # waypointokbol: ha mar csak 0-1 pont maradt, az [start-2] index
            # a meg csupa nulla sort olvasta, es a kitoltes rossz iranyba ment.
            start_index = len(waypoints)
            a, b = (get_displacement_vector(vehicle_location, vector(env.route_waypoints[k][0].transform.location),
                                            theta)[:2] for k in (-2, -1))
            reference_vector = b - a
            for i in range(start_index, 15):
                relative_waypoints[i] = (relative_waypoints[i-1] if i > 0 else b) + reference_vector
        encoded_state['waypoints'] = relative_waypoints
        return encoded_state

    return encode_state


# Create observation space for the car
def create_observation_space(cam_ae, lidar_latent_shape):
    #OBS Space dict
    observation_space = {}

    # AE-vel kodolt kamerakep. Az encode() vegen tanh van (latent_scale=4.0),
    # ezert a latent garantaltan -4..4 - ugyanaz a nagysagrend, mint a tobbi
    # obs jele. Korlat nelkul -121..144 kozott mozgott, es elnyomta oket.
    # tanh nelkuli AE-nel (use_tanh=False) nincs korlat: inf, mint a lidarnal.
    scale = cam_ae.hparams.latent_scale if cam_ae.hparams.get("use_tanh", True) else np.inf
    observation_space['cam_latent'] = gym.spaces.Box(
        low=-scale, high=scale,
        shape=(cam_ae.hparams.latent_dim, ), dtype=np.float32)

    # A lidar latens TERBELI jellemzoterkep (C, H, W), nem vektor - az SAC sajat
    # CNN extractora dolgozza fel. A graf-encoder vegen nincs tanh, ezert itt
    # nincs garantalt korlat: inf a helyes hatar, nem egy kitalalt szam, ami
    # csendben levagna a jelet.
    observation_space['lidar_latent'] = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=lidar_latent_shape, dtype=np.float32)

    # lists for vehicle description
    low, high = [],[]
    low.append(-1), high.append(1) # steer
    low.append(0), high.append(1) # throttle
    low.append(0), high.append(120) #Speed
    low.append(-3.14), high.append(3.14) # next angle waypoint
    observation_space['vehicle_measures'] = gym.spaces.Box(low=np.array(low, dtype=np.float32), high=np.array(high, dtype=np.float32), dtype=np.float32)

    observation_space['maneuver'] = gym.spaces.Discrete(4) # manuever
    observation_space['waypoints'] = gym.spaces.Box(low=-50, high=50, shape=(15, 2),dtype=np.float32) # waypoints

    return gym.spaces.Dict(observation_space)
