import os
import time

import numpy as np

import torch

# GYM 
import gymnasium as gym

#STB3 imports 
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure


from carla_env.envs.carla_route_env import CarlaRouteEnv


from carla_env.wrappers import vector, get_displacement_vector

from feature_extractions.camera.camera_ae import CameraAutoEncoder



from carla_env.reward import reward_fn
from utils import HParamCallback, TensorboardCallback, write_json
from config import (
    LSIZE, LOG_DIR, RELOAD_MODEL, RELOAD_MODEL_PATH, TOTAL_STEPS, SEED,TOWN,
    OBS_RES, ACTION_SMOOTHING, NUM_CHECKPOINTS,
    FPS, ACTIVATE_SPECTATOR, ACTIVATE_RENDER,
    AE_CKPT_PATH, ALGORITHM_PARAMS, CONFIG,
)


# AE loader
def load_cam_ae(ckpt_path, device):
    if not os.path.exists(ckpt_path):
        raise Exception(f"Error - AE model does not exist: {ckpt_path}")

    # map_location: ha CUDA-n tanult es most CPU-n futtatnank, e nelkul elszall.
    cam_ae = CameraAutoEncoder.load_from_checkpoint(ckpt_path, map_location=device)
    # eval(): a BatchNorm maskepp viselkedik tanitas es kiertekeles kozben.
    cam_ae.eval().to(device)
    # Az AE fix: az RL nem tanitja tovabb.
    for p in cam_ae.parameters():
        p.requires_grad_(False)

    print(f"Loaded AE from {ckpt_path}, latent_dim {cam_ae.hparams.latent_dim}")
    return cam_ae


# Az encode_state-nek szuksege van a betoltott AE-re, ezert egy gyarfuggveny
# adja at neki - igy nem kell globalis valtozo.
def create_encode_state_fn(cam_ae, device):

    @torch.no_grad()
    def encode_state(env):
        # dict for current CARLA state
        encoded_state = {}

        # Nyers RGB kep -> AE latent. A normalizalas ugyanaz, mint a
        # tanitasban: uint8/255 es (H,W,C) -> (C,H,W).
        # ascontiguousarray: a CARLA kamera BGR->RGB fordulata (wrappers.py
        # array[:, :, ::-1]) negativ stride-ot hagy, amit a from_numpy nem vesz at.
        image = np.ascontiguousarray(env.observation, dtype=np.uint8)
        x = torch.from_numpy(image).permute(2, 0, 1).float().div_(255.0)
        x = x.unsqueeze(0).to(device)                # (1, 3, 80, 160)
        z = cam_ae.encode(x)
        encoded_state['ae_latent'] = z[0].cpu().numpy().astype(np.float32)

        # A rekonstrukcio csak a megjelenitesnek kell (env.render rakja ki a
        # nyers kamerakep ala) - renderelés nelkul felesleges dekodolni.
        if env.activate_render:
            recon = cam_ae.decode(z)[0].clamp(0, 1)
            env.ae_reconstruction = (recon * 255).byte().permute(1, 2, 0).cpu().numpy()

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
            start_index = len(waypoints)
            reference_vector = relative_waypoints[start_index-1] - relative_waypoints[start_index-2]
            for i in range(start_index, 15):
                relative_waypoints[i] = relative_waypoints[i-1] + reference_vector
        encoded_state['waypoints'] = relative_waypoints
        return encoded_state

    return encode_state


# Create observation space for the car
def create_observation_space():    
    #OBS Space dict  
    observation_space = {}
    
    # AE-vel kodolt kamerakep. Az encode() vegen tanh van (latent_scale=4.0),
    # ezert a latent garantaltan -4..4 - ugyanaz a nagysagrend, mint a tobbi
    # obs jele. Korlat nelkul -121..144 kozott mozgott, es elnyomta oket.
    observation_space['ae_latent'] = gym.spaces.Box(low=-4, high=4, shape=(LSIZE, ), dtype=np.float32)

    
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



def main():

    
    os.makedirs(LOG_DIR, exist_ok=True)
    
    observation_space = create_observation_space()

    # Az AE-t a CARLA elott toltjuk be: ha a checkpoint hibas, ne alljon fel
    # elotte a teljes szimulacio.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cam_ae = load_cam_ae(AE_CKPT_PATH, device)
    encode_state_fn = create_encode_state_fn(cam_ae, device)

    rl_model_path= RELOAD_MODEL_PATH+"/model_final.zip"

    env = CarlaRouteEnv(
        obs_res=OBS_RES,
        viewer_res=(1280,720),
        host="localhost",
        port=2000,
        town=TOWN,
        max_route_length=800, 
        min_route_length=400,
        reward_fn=reward_fn,
        observation_space=observation_space,
        encode_state_fn=encode_state_fn,
        fps=FPS,
        action_smoothing=ACTION_SMOOTHING,
        action_space_type='continuous',
        activate_spectator=ACTIVATE_SPECTATOR,
        activate_render=ACTIVATE_RENDER,
    )


    if RELOAD_MODEL:
        model = SAC.load(rl_model_path, env=env, device='cuda',
                         tensorboard_log=LOG_DIR, verbose=1)
    else:
        model = SAC('MultiInputPolicy', env=env, verbose=1, seed=SEED,
                    tensorboard_log=LOG_DIR, device='cuda', **ALGORITHM_PARAMS)
    model_suffix = f"{int(time.time())}_SAC"

    model_name = f'{model.__class__.__name__}_{model_suffix}'
    model_dir = os.path.join(LOG_DIR, model_name)
    new_logger = configure(model_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)
    write_json(CONFIG, os.path.join(model_dir, 'config.json'))

    try:
        model.learn(
            total_timesteps=TOTAL_STEPS,
            callback=[
                HParamCallback(CONFIG),
                TensorboardCallback(1),
                CheckpointCallback(
                    save_freq=TOTAL_STEPS // NUM_CHECKPOINTS,
                    save_path=model_dir,
                    name_prefix="model",
                ),
            ],
            reset_num_timesteps=False,
        )
        model.save(os.path.join(model_dir, "model_final"))
        print(f"Training complete — model saved to {model_dir}/model_final")
    except KeyboardInterrupt:
        print("Training interrupted — saving model...")
        model.save(os.path.join(model_dir, "model_interrupted"))
        print(f"Model saved to {model_dir}/model_interrupted")


if __name__ == '__main__':
    main()
