import os
import rclpy
import threading
import gymnasium as gym
from stable_baselines3 import PPO
from tools.carla_env import CarlaRLEnvironment
from tools.bev_processor import BEVProcessor
from tools.custom_cnn import BEVFeatureExtractor

def train():
    rclpy.init()
    ros_node = rclpy.create_node('ppo_trainer')
    
    # Path to your saved model
    MODEL_PATH = "carla_ppo_fast_interrupted.zip"
    LOG_DIR = "./ppo_carla_tensorboard/"

    try:
        bev = BEVProcessor()
        env = CarlaRLEnvironment(ros_node, bev)

        ros_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
        ros_thread.start()

        if os.path.exists(MODEL_PATH):
            print(f"--- Found checkpoint. Resuming training from {MODEL_PATH} ---")
            # Load the model and connect it to the current environment
            model = PPO.load(MODEL_PATH, env=env, tensorboard_log=LOG_DIR)
        else:
            print("--- No checkpoint found. Starting fresh training ---")
            policy_kwargs = dict(
                features_extractor_class=BEVFeatureExtractor,
                features_extractor_kwargs=dict(features_dim=256),
                net_arch=dict(pi=[128, 128], vf=[128, 128]) 
            )
            model = PPO(
                "CnnPolicy",
                env,
                policy_kwargs=policy_kwargs,
                learning_rate=2e-4,
                ent_coef=0.05,
                verbose=1,
                tensorboard_log=LOG_DIR
            )

        # Start learning again
        model.learn(total_timesteps=10_000_000, reset_num_timesteps=False)

    except KeyboardInterrupt:
        print("\nSaving model before exit...")
        model.save("carla_ppo_fast_interrupted")
    finally:
        ros_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    train()