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

# A halo alakja (latent_dim, latent_scale, ...) a checkpoint hparams-abol jon,
# nem innen - igy nem lehet elcsuszni a tanitott modelltol.
CAMERA = dict(
    ckpt=path("feature_extractions", "camera", "camera_ae.ckpt"),
)

# A range_image parameterek a MI CARLA szenzorunkhoz vannak merve (lasd
# graph_ae.points_to_range_image docstringjet) - ezeknek egyezniuk kell a
# tanitaskor hasznaltakkal, kulonben a latens ertelmetlen.
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
    town="Town04",
    host="localhost",
    port=2000,
    obs_res=(160, 80),
    viewer_res=(1280, 720),
    fps=20,
    max_route_length=1200,
    min_route_length=1000,
    action_smoothing=0.75,
    action_space_type="continuous",
    activate_spectator=True,
    activate_render=True,
)
# =============================================================================
# TANITAS
# =============================================================================

TRAIN = dict(
    log_dir=path("tensorboard"),
    total_steps=100_000_000,
    seed=100,
    num_checkpoints=10,
    reload_model=False,
    reload_model_path=path("tensorboard", "SAC_1778506873_SAC"),
    reload_model_file="model_final.zip",
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
        # Minden env step utan 1 gradient lepes. A frissites/adat arany
        # ugyanaz (1:1), mint a korabbi 64/64-nel, de az 64 lepest egyben
        # futtatott: ~0.7 s-ig allt a sim minden 64. step utan (szinkron
        # modban a vilag addig nem lep, de lathatoan akadt). Igy stepenkent
        # ~11 ms, es a teljes ido kozel ugyanannyi.
        train_freq=1,
        gradient_steps=1,
        learning_starts=10000,
        use_sde=True,
        sde_sample_freq=8,
        policy_kwargs=dict(
            # exp(-1.5) = 0.22 szoras.
            #
            # A -3 (exp = 0.05) a +-1-es akciotartomanyban gyakorlatilag
            # determinisztikus volt: 4864 gradiens lepes alatt a train/std meg
            # sem mozdult 0.0499-rol, es az agent a "lassan araszolok" lokalis
            # optimumba tanult bele (a reward -7.9-rol -8.9-re ROMLOTT).
            #
            # A -1 (0.37) viszont a masik veglet: a kormanyt lathatoan rangatta.
            # A -1.5 a kozeput - 4.5x tobb exploracio, mint az eredeti, de a
            # zaj fele akkora, mint a -1-nel.
            log_std_init=-1.5,
            net_arch=[500, 300],
            # Sajat extractor az SB3 CombinedExtractora helyett: az a lidar
            # latenst (16,11,32 = 5632 dim) egyszeruen lelapitana, es a policy
            # bemenetenek 99%-a a lidar lenne - a sebesseg, a szog es a
            # waypointok eltunnenek benne. Lasd fusion_extractor.py.
            features_extractor_class=CarlaFusionExtractor,
            features_extractor_kwargs=dict(
                fusion_dim=256,         # a kozos kamera+lidar szenzor-vektor
                state_dim=64,           # a vehicle/maneuver/route_preview ag kimenete
                cnn_base_channels=16,   # a lidar CNN sajat szelessege (nem a latense)
                fusion_mode="add",      # "add" vagy "concat"
            ),
        ),
    ),
)

STATE = ["steer", "throttle", "speed", "waypoints", "angle_next_waypoint", "maneuver"]


# =============================================================================
# A train_rl.py EZT az egy dictet importalja.
# =============================================================================

CONFIG = dict(
    camera=CAMERA,
    lidar=LIDAR,
    env=ENV,
    train=TRAIN,
    algorithm=ALGORITHM,
    state=STATE,
    wrappers=[],
)
