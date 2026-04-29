"""PPO training entry point.

Run from the repo root:
    python -m agent.train_ppo
or from inside agent/:
    python train_ppo.py

If a previous checkpoint or a saved interrupt model is found under
TRAIN_DATA_DIR, training resumes from it automatically. Delete that folder
(or pass --fresh) to start over.

Requires: stable-baselines3, torch, gymnasium, opencv-python, rclpy,
carla_msgs, lanelet2.

Pre-flight:
  - The CARLA simulator is running.
  - carla_ros_bridge is launched in async (free-running) mode with the
    semantic_segmentation_front, semantic_lidar, odometry, speedometer,
    and collision sensors all enabled in objects.json.
"""

import argparse
import glob
import os
import re
import threading

import rclpy
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

try:
    from .tools.carla_gym_env import CarlaGymEnv
    from .tools.ppo_ros_bridge import PPOROSBridgeNode
except ImportError:
    from tools.carla_gym_env import CarlaGymEnv
    from tools.ppo_ros_bridge import PPOROSBridgeNode


HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_DATA_DIR = os.path.join(HERE, "train_data")
CHECKPOINT_DIR = os.path.join(TRAIN_DATA_DIR, "checkpoints")
TENSORBOARD_DIR = os.path.join(TRAIN_DATA_DIR, "tb")
MONITOR_PATH = os.path.join(TRAIN_DATA_DIR, "monitor.csv")
INTERRUPT_PATH = os.path.join(TRAIN_DATA_DIR, "ppo_carla.zip")
FINAL_PATH = os.path.join(TRAIN_DATA_DIR, "ppo_carla_final.zip")

CHECKPOINT_PREFIX = "ppo_carla"
TOTAL_TIMESTEPS = 1_000_000
SAVE_FREQ = 20_000


def _find_latest_checkpoint():
    pattern = os.path.join(CHECKPOINT_DIR, f"{CHECKPOINT_PREFIX}_*_steps.zip")
    candidates = []
    for path in glob.glob(pattern):
        m = re.search(r"_(\d+)_steps\.zip$", path)
        if m:
            candidates.append((int(m.group(1)), path))
    if not candidates:
        return None, 0
    steps, path = max(candidates)
    return path, steps


def _resolve_resume_path():
    ckpt_path, ckpt_steps = _find_latest_checkpoint()
    if os.path.exists(INTERRUPT_PATH):
        return INTERRUPT_PATH, ckpt_steps
    return ckpt_path, ckpt_steps


def make_env_and_node():
    rclpy.init()
    node = PPOROSBridgeNode()
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()

    env = CarlaGymEnv(node)
    env = Monitor(env, filename=MONITOR_PATH)
    return env, node, thread


def build_model(env, resume_path):
    if resume_path is not None:
        print(f"▶ Resuming from {resume_path}")
        return PPO.load(
            resume_path,
            env=env,
            tensorboard_log=TENSORBOARD_DIR,
            device="auto",
        )

    print("▶ Starting a fresh training run.")
    # MultiInputPolicy because the env returns a Dict observation
    # (image + state vector). SB3 builds a NatureCNN for the image and
    # a flatten+MLP for the vector, then concatenates the features.
    return PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log=TENSORBOARD_DIR,
        n_steps=512,
        batch_size=128,
        learning_rate=2.5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        device="auto",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fresh", action="store_true",
        help="Ignore existing checkpoints/interrupt save and train from scratch.",
    )
    args = parser.parse_args()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(TENSORBOARD_DIR, exist_ok=True)

    env, _node, thread = make_env_and_node()

    resume_path, resumed_steps = (None, 0) if args.fresh else _resolve_resume_path()
    model = build_model(env, resume_path)

    remaining = max(TOTAL_TIMESTEPS - resumed_steps, 0)
    if remaining == 0:
        print(f"Already trained for {resumed_steps} steps ≥ TOTAL_TIMESTEPS. Nothing to do.")
        env.close()
        rclpy.shutdown()
        thread.join(timeout=1.0)
        return

    checkpoint_cb = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path=CHECKPOINT_DIR,
        name_prefix=CHECKPOINT_PREFIX,
    )

    try:
        model.learn(
            total_timesteps=remaining,
            callback=checkpoint_cb,
            reset_num_timesteps=(resume_path is None),
            tb_log_name="PPO",
        )
        model.save(FINAL_PATH)
        if os.path.exists(INTERRUPT_PATH):
            os.remove(INTERRUPT_PATH)
    except KeyboardInterrupt:
        print("\n⏸  Interrupted — saving so the next run can resume…")
        model.save(INTERRUPT_PATH)
    finally:
        env.close()
        rclpy.shutdown()
        thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
