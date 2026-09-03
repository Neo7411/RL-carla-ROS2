from utils import lr_schedule

LSIZE = 64
LOG_DIR = "tensorboard"
RELOAD_MODEL = False
RELOAD_MODEL_PATH = "./tensorboard/SAC_1778506873_SAC"  # path to .zip checkpoint to resume from
TOTAL_STEPS = 100_000_000
SEED = 100
OBS_RES = (160, 80)
AE_CKPT_PATH = "autoencoders/camera/camera_ae.ckpt"
STATE = ["steer", "throttle", "speed", "waypoints", "angle_next_waypoint", "maneuver"]
ACTION_SMOOTHING = 0.75
NUM_CHECKPOINTS = 10
FPS = 20
TOWN="Town04"
ACTIVATE_SPECTATOR = True
ACTIVATE_RENDER = True

ALGORITHM_PARAMS = dict(
    learning_rate=lr_schedule(1e-4, 5e-7, 2),
    buffer_size=100_000,
    batch_size=256,
    ent_coef='auto',
    gamma=0.98,
    tau=0.02,
    train_freq=64,
    gradient_steps=64,
    learning_starts=10000,
    use_sde=True,
    # A gSDE zaj-matrixa alapertelmezesben (-1) csak a rollout elejen frissul,
    # ami train_freq=64 mellett 64 tick = 3.2 masodperc: addig UGYANAZ a
    # torzitas hat a gazra, tehat 3.2 mp-ig gyorsul, majd 3.2 mp-ig lassul.
    # Ez a lathato "rangatas" oka - az action_smoothing idoallandoja csak
    # 0.17 mp, ezzel nem tudja kisimitani.
    # 8 tick (0.4 mp) mellett a zaj magasabb frekvenciaju es kisebb hatasu,
    # az exploracio viszont megmarad.
    sde_sample_freq=8,
    policy_kwargs=dict(log_std_init=-3, net_arch=[500, 300]),
)

CONFIG = {
    "algorithm": "SAC",
    "algorithm_params": ALGORITHM_PARAMS,
    "state": STATE,
    "ae_ckpt": AE_CKPT_PATH,
    "action_smoothing": ACTION_SMOOTHING,
    "obs_res": OBS_RES,
    "seed": SEED,
    "wrappers": [],
}
