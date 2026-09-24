import os
import time

import torch


#STB3 imports
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

from carla_env.envs.carla_route_env import CarlaRouteEnv
from carla_env.reward import reward_fn
from config import CONFIG
from utils import (
    HParamCallback, TensorboardCallback, write_json,
    load_cam_ae, load_lidar_ae, create_encode_state_fn, create_observation_space,
)


def main():

    train_cfg = CONFIG["train"]
    env_cfg = CONFIG["env"]
    algo_cfg = CONFIG["algorithm"]
    log_dir = train_cfg["log_dir"]

    os.makedirs(log_dir, exist_ok=True)

    # Az AE-ket a CARLA elott toltjuk be: ha egy checkpoint hibas, ne alljon fel
    # elotte a teljes szimulacio.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*60)
    print(f"[INFO] Available device for torch is: {device}")
    print("="*60)
    cam_ae = load_cam_ae(CONFIG["camera"], device)
    lidar_ae, lidar_latent_shape = load_lidar_ae(CONFIG["lidar"], device)

    observation_space = create_observation_space(cam_ae, lidar_latent_shape)
    encode_state_fn = create_encode_state_fn(cam_ae, lidar_ae, CONFIG, device)

    env = CarlaRouteEnv(
        obs_res=env_cfg["obs_res"],
        viewer_res=env_cfg["viewer_res"],
        host=env_cfg["host"],
        port=env_cfg["port"],
        town=env_cfg["town"],
        max_route_length=env_cfg["max_route_length"],
        min_route_length=env_cfg["min_route_length"],
        reward_fn=reward_fn,
        observation_space=observation_space,
        encode_state_fn=encode_state_fn,
        fps=env_cfg["fps"],
        action_smoothing=env_cfg["action_smoothing"],
        action_space_type=env_cfg["action_space_type"],
        activate_spectator=env_cfg["activate_spectator"],
        activate_render=env_cfg["activate_render"],
        activate_lidar=True,
    )

    if train_cfg["reload_model"]:
        rl_model_path = os.path.join(train_cfg["reload_model_path"],
                                     train_cfg["reload_model_file"])
        model = SAC.load(rl_model_path, env=env, device=algo_cfg["device"],
                         tensorboard_log=log_dir, verbose=1)
    else:
        model = SAC('MultiInputPolicy', env=env, verbose=1, seed=train_cfg["seed"],
                    tensorboard_log=log_dir, device=algo_cfg["device"],
                    **algo_cfg["params"])

    model_suffix = f"{int(time.time())}_{algo_cfg['name']}"
    model_name = f'{model.__class__.__name__}_{model_suffix}'
    model_dir = os.path.join(log_dir, model_name)
    new_logger = configure(model_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)
    write_json(CONFIG, os.path.join(model_dir, 'config.json'))

    total_steps = train_cfg["total_steps"]
    try:
        model.learn(
            total_timesteps=total_steps,
            callback=[
                HParamCallback(CONFIG),
                TensorboardCallback(1),
                CheckpointCallback(
                    save_freq=train_cfg["checkpoint_freq"],
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
    except Exception:
        # Barmilyen mas hiba (CARLA timeout, szenzorhiba, NaN) eseten is
        # mentunk, kulonben az egesz futas elveszne. Utana tovabbdobjuk.
        print("Training crashed — saving model...")
        model.save(os.path.join(model_dir, "model_crashed"))
        print(f"Model saved to {model_dir}/model_crashed")
        raise


if __name__ == '__main__':
    main()
