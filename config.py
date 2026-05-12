from utils import lr_schedule

LSIZE = 64
LOG_DIR = "tensorboard"
RELOAD_MODEL = True
RELOAD_MODEL_PATH = "./tensorboard/SAC_1778506873_SAC"  # path to .zip checkpoint to resume from
TOTAL_STEPS = 130_000
SEED = 100
OBS_RES = (160, 80)
VAE_MODEL = "vae_64_augmentation"
STATE = ["steer", "throttle", "speed", "waypoints", "angle_next_waypoint", "maneuver"]
ACTION_SMOOTHING = 0.75
NUM_CHECKPOINTS = 10
FPS = 20
ACTIVATE_SPECTATOR = True
ACTIVATE_RENDER = True

ALGORITHM_PARAMS = dict(
    learning_rate=lr_schedule(1e-4, 5e-7, 2),
    buffer_size=300000,
    batch_size=256,
    ent_coef='auto',
    gamma=0.98,
    tau=0.02,
    train_freq=64,
    gradient_steps=64,
    learning_starts=10000,
    use_sde=True,
    policy_kwargs=dict(log_std_init=-3, net_arch=[500, 300]),
)

REWARD_PARAMS = dict(
    early_stop=True,
    min_speed=20.0,
    max_speed=35.0,
    target_speed=25.0,
    max_distance=2.0,
    max_std_center_lane=0.35,
    max_angle_center_lane=90,
    penalty_reward=-10,
)

CONFIG = {
    "algorithm": "SAC",
    "algorithm_params": ALGORITHM_PARAMS,
    "state": STATE,
    "vae_model": VAE_MODEL,
    "action_smoothing": ACTION_SMOOTHING,
    "reward_fn": "reward_fn5",
    "reward_params": REWARD_PARAMS,
    "obs_res": OBS_RES,
    "seed": SEED,
    "wrappers": [],
}
