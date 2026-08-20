import os
import time

import numpy as np 

# torch imports
import torch
from torchvision import transforms

# GYM 
import gymnasium as gym

#STB3 imports 
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure


# Custom coded imports 
#Import VAE
from vae.models import VAE

from carla_env.envs.carla_route_env_backup import CarlaRouteEnv


from carla_env.encode_decode_functions import create_encode_state_fn


from carla_env.reward import reward_fn
from utils import HParamCallback, TensorboardCallback, write_json
from config import (
    LSIZE, LOG_DIR, RELOAD_MODEL, RELOAD_MODEL_PATH, TOTAL_STEPS, SEED,TOWN,
    OBS_RES, ACTION_SMOOTHING, NUM_CHECKPOINTS,
    FPS, ACTIVATE_SPECTATOR, ACTIVATE_RENDER,
    ALGORITHM_PARAMS, CONFIG,
)


# VAE loader function 
def load_vae(vae_dir, latent_size):
    model_dir = os.path.join(vae_dir, 'best.tar')
    model = VAE(latent_size)
    if os.path.exists(model_dir):
        state = torch.load(model_dir)
        print("Reloading model at epoch {}"
              ", with test error {}".format(
            state['epoch'],
            state['precision']))
        model.load_state_dict(state['state_dict'])
        return model
    raise Exception("Error - VAE model does not exist")


# Create observation space for the car 
def create_observation_space():    
    #OBS Space dict  
    observation_space = {}
    
    # vae encoded camera input 
    observation_space['vae_latent'] = gym.spaces.Box(low=-4, high=4, shape=(LSIZE, ), dtype=np.float32) 
    
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
    # step one clear cuda cache!!!
    torch.cuda.empty_cache()
    
    os.makedirs(LOG_DIR, exist_ok=True)

    vae = load_vae('./vae/model', LSIZE)
    encode_state_fn, decode_vae_fn = create_encode_state_fn(vae)

    
    observation_space = create_observation_space()


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
        decode_vae_fn=decode_vae_fn,
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
