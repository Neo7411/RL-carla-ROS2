import numpy as np
from config import REWARD_PARAMS

low_speed_timer = 0

min_speed = REWARD_PARAMS["min_speed"]
max_speed = REWARD_PARAMS["max_speed"]
target_speed = REWARD_PARAMS["target_speed"]
max_distance = REWARD_PARAMS["max_distance"]
max_std_center_lane = REWARD_PARAMS["max_std_center_lane"]
max_angle_center_lane = REWARD_PARAMS["max_angle_center_lane"]
penalty_reward = REWARD_PARAMS["penalty_reward"]
early_stop = REWARD_PARAMS["early_stop"]
reward_functions = {}


def create_reward_fn(reward_fn):
    def func(env):
        terminal_reason = "Running..."
        if early_stop:
            global low_speed_timer
            low_speed_timer += 1.0 / env.fps
            speed = env.vehicle.get_speed()
            if low_speed_timer > 5.0 and speed < 1.0 and env.current_waypoint_index >= 1:
                env.terminal_state = True
                terminal_reason = "Vehicle stopped"

            if env.distance_from_center > max_distance:
                env.terminal_state = True
                terminal_reason = "Off-track"

            if max_speed > 0 and speed > max_speed:
                env.terminal_state = True
                terminal_reason = "Too fast"

        reward = 0
        if not env.terminal_state:
            reward += reward_fn(env)
        else:
            low_speed_timer = 0.0
            reward += penalty_reward
            print(f"{env.episode_idx}| Terminal: ", terminal_reason)

        if env.success_state:
            print(f"{env.episode_idx}| Success")

        env.extra_info.extend([terminal_reason, ""])
        return reward

    return func


def reward_fn5(env):
    angle = env.vehicle.get_angle(env.current_waypoint)
    speed_kmh = env.vehicle.get_speed()
    if speed_kmh < min_speed:
        speed_reward = speed_kmh / min_speed
    elif speed_kmh > target_speed:
        speed_reward = 1.0 - (speed_kmh - target_speed) / (max_speed - target_speed)
    else:
        speed_reward = 1.0

    centering_factor = max(1.0 - env.distance_from_center / max_distance, 0.0)
    angle_factor = max(1.0 - abs(angle / np.deg2rad(max_angle_center_lane)), 0.0)
    std = np.std(env.distance_from_center_history)
    distance_std_factor = max(1.0 - abs(std / max_std_center_lane), 0.0)

    return speed_reward * centering_factor * angle_factor * distance_std_factor


reward_functions["reward_fn5"] = create_reward_fn(reward_fn5)
