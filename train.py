import os
import time
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

from carla_env.envs.carla_route_env import CarlaRouteEnv
from carla_env.state_commons import create_encode_state_fn, load_vae
from carla_env.rewards import reward_functions
from utils import HParamCallback, TensorboardCallback, write_json
from config import (
    LSIZE, LOG_DIR, RELOAD_MODEL, RELOAD_MODEL_PATH, TOTAL_STEPS, SEED,
    OBS_RES, STATE, ACTION_SMOOTHING, NUM_CHECKPOINTS,
    FPS, ACTIVATE_SPECTATOR, ACTIVATE_RENDER,
    ALGORITHM_PARAMS, CONFIG,
)


def main():
    os.makedirs(LOG_DIR, exist_ok=True)

    vae = load_vae('./vae/model', LSIZE)
    observation_space, encode_state_fn, decode_vae_fn = create_encode_state_fn(vae, STATE)

    rl_model_path= RELOAD_MODEL_PATH+"/model_final.zip"
    env = CarlaRouteEnv(
        obs_res=OBS_RES,
        host="localhost",
        port=2000,
        reward_fn=reward_functions[CONFIG["reward_fn"]],
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
