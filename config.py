import os
from feature_extractions.fusion_extractor import CarlaFusionExtractor
from utils import lr_schedule

# A repo gyokere - minden path ehhez kepest ertelmezodik, igy a train_rl.py
# barmelyik konyvtarbol inditható.
ROOT = os.path.dirname(os.path.abspath(__file__))


def path(*parts):
    return os.path.join(ROOT, *parts)


# =============================================================================
# FEATURE EXTRACTOROK
# =============================================================================

CAMERA = dict(
    ckpt=path("feature_extractions", "camera", "camera_ae.ckpt"),
)

LIDAR = dict(
    ckpt=path("feature_extractions", "lidar", "graph_ae.ckpt"),
    range_image=dict(
        size=(44, 256),
        fov=(-0.9, -25.0),
        depth_range=(1.0, 50.0),
        depth_scale=5.68,
    ),
)
# =============================================================================
# KORNYEZET CARLA   
# =============================================================================

ENV = dict(
    launch_sim=True,
    carla_root="/home/xavier11/CARLA/CARLA_0.9.16",
    town="Town04",
    host="localhost",
    port=2000,
    obs_res=(160, 80),
    viewer_res=(1280, 720),
    fps=20,
    max_route_length=2500,
    min_route_length=1500,
    action_smoothing=0.75,
    action_space_type="continuous",
    activate_spectator=True,
    activate_render=True,
    reward_fn="reward_fn",
    traffic_vehicles=50,
    traffic_start_m=20.0,
    traffic_gap_m=(12.0, 30.0),
    traffic_speed_kmh=(20.0, 25.0),
    hybrid_radius=70.0,
)
# =============================================================================
# TANITAS
# =============================================================================

TRAIN = dict(
    log_dir=path("tensorboard"),
    total_steps=100_000_000,
    seed=100,
    # Ennyi env-lepesenkent ment. Korabban total_steps // 10 = 10M volt, igy
    # egy valos futas alatt soha nem mentett.
    checkpoint_freq=50_000,
    # Ennyi lepesenkent meri, mire figyel a policy (utils.AttentionCallback).
    attention_freq=10_000,
    reload_model=True,
    reload_model_path=path("tensorboard", "SAC_OVERTAKE_2"),
    reload_model_file="model_interrupted.zip",
)

ALGORITHM = dict(
    name="SAC",
    device="cuda",
    params=dict(
        learning_rate=lr_schedule(1e-4, 5e-7, 2),
        buffer_size=100_000,
        batch_size=256,
        ent_coef="auto",
        gamma=0.98,
        tau=0.02,
        train_freq=1,
        gradient_steps=1,
        learning_starts=10000,
        use_sde=True,
        sde_sample_freq=8,
        # A learning_starts alatt se egyenletes veletlen akcio menjen, hanem a
        # (betanitatlan) policy gSDE zajjal. Az egyenletes veletlen fek (a
        # simitassal) vegig bent maradt, es az auto el sem indult; a policy
        # ~0.5-os fek-akcioja a 0.7-es holtsav alatt van, igy nem fekez.
        use_sde_at_warmup=True,
        policy_kwargs=dict(
            log_std_init=-1.5,
            net_arch=[500, 300],
            features_extractor_class=CarlaFusionExtractor,
            features_extractor_kwargs=dict(
                fusion_dim=256,         # a kozos kamera+lidar szenzor-vektor
                state_dim=64,           # a vehicle/waypoint/maneuver ag kimenete
                cnn_base_channels=16,   # a lidar CNN sajat szelessege (nem a latense)
                fusion_mode="add",      # "add" vagy "concat"
            ),
        ),
    ),
)


# =============================================================================
# A train_rl.py EZT az egy dictet importalja.
# =============================================================================

CONFIG = dict(
    camera=CAMERA,
    lidar=LIDAR,
    env=ENV,
    train=TRAIN,
    algorithm=ALGORITHM,
    wrappers=[],
)
