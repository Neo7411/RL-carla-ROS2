import os
import time

import numpy as np 


# GYM 
import gymnasium as gym

#STB3 imports 
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure


from carla_env.envs.carla_route_env import CarlaRouteEnv


from carla_env.wrappers import vector, get_displacement_vector



from carla_env.reward import reward_fn
from utils import HParamCallback, TensorboardCallback, write_json
from config import (
    LSIZE, LOG_DIR, RELOAD_MODEL, RELOAD_MODEL_PATH, TOTAL_STEPS, SEED,TOWN,
    OBS_RES, ACTION_SMOOTHING, NUM_CHECKPOINTS,
    FPS, ACTIVATE_SPECTATOR, ACTIVATE_RENDER,
    ALGORITHM_PARAMS, CONFIG,
)


def encode_state(env):
        # dict for current CARLA state
        encoded_state = {}

        # Nyers szegmentalt kep. uint8, (80,160,3), NEM normalizalva -
        # az SB3 NatureCNN belul oszt 255-tel.
        encoded_state['seg_camera'] = np.asarray(env.observation, dtype=np.uint8)

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


# Create observation space for the car 
def create_observation_space():    
    #OBS Space dict  
    observation_space = {}
    
    # vae encoded camera input 
    # observation_space['vae_latent'] = gym.spaces.Box(low=-4, high=4, shape=(LSIZE, ), dtype=np.float32) 
    
    # Change the latent t raw camera
    observation_space['seg_camera'] = gym.spaces.Box(low=0, high=255, shape=(80, 160, 3), dtype=np.uint8)
    
    
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
        encode_state_fn=encode_state,
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
