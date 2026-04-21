import os
import rclpy
import threading

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from tools.carla_env import CarlaRLEnvironment
from tools.bev_processor import BEVProcessor
from tools.custom_cnn import BEVGoalExtractor


TOTAL_TIMESTEPS = 10_000_000

# Linear schedules (SB3 passes `progress_remaining` in [1.0, 0.0])
def linear_schedule(start: float, end: float):
    def fn(progress_remaining: float) -> float:
        return end + (start - end) * progress_remaining
    return fn


class EntropyCoefScheduler(BaseCallback):
    """Linearly anneal PPO's ent_coef across training."""
    def __init__(self, start: float, end: float, total_steps: int):
        super().__init__()
        self.start, self.end, self.total = start, end, total_steps

    def _on_step(self) -> bool:
        frac = min(1.0, self.num_timesteps / max(self.total, 1))
        self.model.ent_coef = self.start + (self.end - self.start) * frac
        return True


def train():
    rclpy.init()
    ros_node = rclpy.create_node('ppo_trainer')

    MODEL_PATH = "carla_ppo.zip"
    LOG_DIR = "./ppo_carla_tensorboard/"
    CKPT_DIR = "./checkpoints/"

    try:
        bev = BEVProcessor()
        env = CarlaRLEnvironment(ros_node, bev)

        ros_thread = threading.Thread(
            target=rclpy.spin, args=(ros_node,), daemon=True)
        ros_thread.start()

        policy_kwargs = dict(
            features_extractor_class=BEVGoalExtractor,
            features_extractor_kwargs=dict(cnn_dim=512, mlp_dim=64),
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            log_std_init=-0.5,   # less noisy exploration (default 0.0)
        )

        if os.path.exists(MODEL_PATH):
            print(f"--- Resuming from {MODEL_PATH} ---")
            model = PPO.load(MODEL_PATH, env=env, tensorboard_log=LOG_DIR)
        else:
            print("--- Fresh training (goal-conditioned) ---")
            model = PPO(
                "MultiInputPolicy",
                env,
                policy_kwargs=policy_kwargs,
                learning_rate=linear_schedule(3e-4, 1e-5),
                n_steps=512,         # shorter rollouts → 4× more updates/hr
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                ent_coef=0.01,       # scheduled down by EntropyCoefScheduler
                clip_range=0.2,
                verbose=1,
                tensorboard_log=LOG_DIR,
            )

        ckpt_cb = CheckpointCallback(
            save_freq=50_000,
            save_path=CKPT_DIR,
            name_prefix="carla_ppo_goal",
        )
        ent_cb = EntropyCoefScheduler(start=0.01, end=0.001,
                                      total_steps=TOTAL_TIMESTEPS)

        model.learn(total_timesteps=TOTAL_TIMESTEPS,
                    reset_num_timesteps=False,
                    callback=[ckpt_cb, ent_cb])

    except KeyboardInterrupt:
        print("\nSaving model before exit...")
        model.save("carla_ppo")
    finally:
        ros_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    train()
