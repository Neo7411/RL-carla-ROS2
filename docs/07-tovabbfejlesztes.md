# 07 — Továbbfejlesztés, ismert hibák, ROS2

Ez a fájl gyakorlati: hol lehet hozzányúlni, mi van elrontva, és merre érdemes menni.

---

# 1. Ismert hibák és gyanús pontok

Ezeket a kódolvasás során találtam. Súlyosság szerint rendezve.

## 🔴 Súlyos — javítást érdemel

### 1.1 `close()` mindig hibázik
📍 [carla_route_env.py:180-186](../carla_env/envs/carla_route_env.py#L180-L186)

```python
def close(self):
    if self.carla_process:      # ← self.carla_process SEHOL nincs beállítva
        self.carla_process.terminate()
```

`AttributeError`-t dob. A `train.py` nem hívja, ezért nem derül ki, de a szereplők
(autó, kamerák) így **sosem takarítódnak el** — több egymás utáni futásnál a CARLA
szerveren felhalmozódnak a régi actorok.

**Javítás:**
```python
def close(self):
    if getattr(self, "carla_process", None):
        self.carla_process.terminate()
    if self.activate_render:
        pygame.quit()
    if self.world is not None:
        self.world.destroy()
    self.closed = True
```
És a `train.py` végére: `env.close()` (vagy `try/finally`).

### 1.2 A `low_speed_timer` nem nullázódik siker esetén
📍 [rewards.py:4, 22, 40](../carla_env/rewards.py#L22)

```python
low_speed_timer = 0      # modulszintű GLOBÁLIS

# a func()-ban:
low_speed_timer += 1.0 / env.fps
...
else:                          # csak terminal_state ágban nullázódik!
    low_speed_timer = 0.0
```

Ha az epizód **sikerrel** ér véget (3000 m), a számláló megőrzi a nagy értékét. A következő
epizód elején az autó még áll (< 1 km/h), és mivel `low_speed_timer > 5.0` már igaz,
azonnal `"Vehicle stopped"` terminálást kaphat, amint a `current_waypoint_index >= 1`.

**Javítás — tedd `env` mezővé:**
```python
# carla_route_env.py reset()-ben:
self.low_speed_timer = 0.0

# rewards.py-ban:
env.low_speed_timer += 1.0 / env.fps
if env.low_speed_timer > 5.0 and speed < 1.0 and env.current_waypoint_index >= 1:
    ...
```
Ez egyben megszünteti a globális állapotot, ami több párhuzamos környezetnél
(`SubprocVecEnv`) amúgy is eltörne.

### 1.3 A `RoadOption` enum hiányos → Town04-en összeomlana
📍 [local_planner.py:20-31](../carla_env/navigation/local_planner.py#L20-L31) vs.
[global_route_planner.py:187, 200](../carla_env/navigation/global_route_planner.py#L187)

A `global_route_planner.py` `RoadOption.CHANGELANERIGHT`-ot és `CHANGELANELEFT`-et használ,
de a `RoadOption` enum ezeket nem tartalmazza.

Town02-n nincs sávváltás, ezért nem derül ki. **Bármely többsávos térképen azonnali
`AttributeError`.**

**Javítás:**
```python
class RoadOption(Enum):
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 0
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6
```
⚠ **De vigyázz:** ekkor a `maneuver` állapot `Discrete(4)`-je is kevés lesz →
`Discrete(7)`-re kell bővíteni a `state_commons.py`-ban, **és** újra kell tanítani a
modellt (megváltozik a bemenet alakja).

## 🟡 Közepes — teljesítmény vagy pontosság

### 1.4 A gráf minden útvonalnál újraépül ★ legnagyobb gyorsítási lehetőség
📍 [planner.py:34-37](../carla_env/navigation/planner.py#L34-L37)

```python
dao = GlobalRoutePlannerDAO(world_map, resolution)
grp = GlobalRoutePlanner(dao)
grp.setup()                      # ← teljes térkép-gráf építése, MINDEN hívásnál
```

A `new_route()` minden epizódnál (és epizódon belüli útvonal-váltásnál) meghívja.
A `setup()` végigmegy a teljes topológián, 1 méterenként waypointokat kér a szervertől —
ez több száz szerver-hívás.

**Javítás — gyorsítótárazás:**
```python
_grp_cache = {}

def _get_planner(world_map, resolution):
    key = (world_map.name, resolution)
    if key not in _grp_cache:
        dao = GlobalRoutePlannerDAO(world_map, resolution)
        grp = GlobalRoutePlanner(dao)
        grp.setup()
        _grp_cache[key] = grp
    return _grp_cache[key]
```

⚠ **Figyelj:** a `GlobalRoutePlanner` **állapotot tart** (`_previous_decision`,
`_intersection_end_node`). Újrahasznosításnál ezeket nullázni kell minden `trace_route`
előtt, különben a kanyar-döntések "átszivárognak" az előző útvonalról:
```python
grp._previous_decision = RoadOption.VOID
grp._intersection_end_node = -1
```

### 1.5 A VAE CPU-n fut
📍 [state_commons.py:15-26](../carla_env/state_commons.py#L15-L26)

Nincs `.to('cuda')`. Minden lépésben CPU-n kódol egy képet, miközben a GPU üresjáratban van.

**Javítás:**
```python
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_vae(vae_dir, latent_size):
    ...
    model.load_state_dict(state['state_dict'])
    model.to(DEVICE)
    model.eval()
    return model

# encode_state-ben:
frame = preprocess_frame(env.observation).to(DEVICE)
```

### 1.6 Busy wait a szenzoradatokra
📍 [carla_route_env.py:419-438](../carla_env/envs/carla_route_env.py#L419-L438)

```python
while self.observation_buffer is None:
    pass                    # 100% CPU pörgetés, potenciális örök fagyás
```

**Javítás — `threading.Event` időkorláttal:**
```python
# __init__-ben:
self._obs_event = threading.Event()

def _set_observation_image(self, image):
    self.observation_buffer = image
    self._obs_event.set()

def _get_observation(self):
    if not self._obs_event.wait(timeout=5.0):
        raise TimeoutError("A dashcam nem küldött képet 5 másodpercen belül")
    obs = self.observation_buffer.copy()
    self.observation_buffer = None
    self._obs_event.clear()
    return obs
```

### 1.7 `time.sleep()`-ek a reset-ben
📍 [carla_route_env.py:144-146, 177](../carla_env/envs/carla_route_env.py#L144-L146)

Három darab `time.sleep(0.2)` — ez epizódonként 0.6 s tiszta várakozás. Több ezer epizódnál
ez órákat jelent. Szinkron módban ezek helyett `world.tick()` hívások lennének helyesek
(azok determinisztikusan előreléptetik a szimulációt).

### 1.8 `truncated` mindig `False`
📍 [carla_route_env.py:365](../carla_env/envs/carla_route_env.py#L365)

A 3000 m elérése (`success_state`) valójában `truncated` lenne, nem `terminated`.
Így az algoritmus úgy kezeli, mintha a világ ott véget érne, és nem folytatja a
value-becslést (bootstrapping). Kisebb torzítást okoz a tanulásban.

**Javítás:**
```python
return encoded_state, self.last_reward, self.terminal_state, self.success_state, info
```

### 1.9 Peremeset a waypoint-extrapolálásban
📍 [state_commons.py:93-97](../carla_env/state_commons.py#L93-L97)

```python
if len(waypoints) < 15:
    start_index = len(waypoints)
    reference_vector = relative_waypoints[start_index-1] - relative_waypoints[start_index-2]
```

Ha `len(waypoints) == 1`, akkor `start_index-2 == -1` → a numpy az **utolsó** elemet
indexeli (`relative_waypoints[14]`, ami csupa nulla) → értelmetlen `reference_vector`.
Ha `len(waypoints) == 0` → `IndexError`.

**Javítás:**
```python
if len(waypoints) == 0:
    reference_vector = np.array([0.0, 1.0])          # előre
elif len(waypoints) == 1:
    reference_vector = relative_waypoints[0] / (np.linalg.norm(relative_waypoints[0]) + 1e-8)
else:
    reference_vector = relative_waypoints[start_index-1] - relative_waypoints[start_index-2]
```

## 🟢 Apró / kozmetikai

### 1.10 `lr_schedule.__str__` nem működik
📍 [utils.py:77-78](../utils.py#L77-L78) — Pythonban a `str()` a *típuson* keresi a
`__str__`-t, nem a példányon, így a JSON-ba mégis `<function ... at 0x...>` kerül.
Használható helyette egy kis osztály `__str__`-rel, vagy egyszerűen írd bele a
`CONFIG`-ba a paramétereket külön kulcsként.

### 1.11 A `Town02` be van drótozva
📍 [wrappers.py:408](../carla_env/wrappers.py#L408) — `client.load_world('Town02')`.
Emeld ki a `config.py`-ba (`MAP = "Town02"`), és add át a `World` konstruktorának.

### 1.12 A `VAE_MODEL` config érték nem használt
📍 [config.py:10](../config.py#L10) vs. [train.py:22](../train.py#L22) — a `train.py`
fixen `'./vae/model'`-t tölt. Vagy használd (`load_vae(f'./vae/{VAE_MODEL}', ...)`),
vagy töröld.

### 1.13 `next_waypoint` nem frissül az útvonal végén
📍 [carla_route_env.py:304-306](../carla_env/envs/carla_route_env.py#L304-L306) — az
`if` miatt a régi érték marad, ami a `distance_to_line` számítást pontatlanná teszi az
utolsó pár méteren.

### 1.14 Kettős `for` ciklus a szegmentációs palettán
📍 [wrappers.py:342-348](../carla_env/wrappers.py#L342-L348) — 12 800 Python iteráció
képkockánként. Jelenleg nem fut, de ha bekapcsolod a `seg_camera`-t, ez lesz a szűk
keresztmetszet. Vektorizálva:
```python
palette = np.zeros((256, 3), dtype=np.uint8)
for k, v in classes.items():
    palette[k] = v
array = palette[segimg]
```

### 1.15 `vector()` névütközés
📍 [wrappers.py:80](../carla_env/wrappers.py#L80) (1 argumentum, nyers koordináták) vs.
[tools/misc.py:98](../carla_env/tools/misc.py#L98) (2 argumentum, egységvektor).
Nevezd át az egyiket, pl. `location_to_array()` és `unit_vector_between()`.

### 1.16 Az `agent.py` / `basic_agent.py` / `roaming_agent.py` importálhatatlan
Ezek `from agents.navigation...`-t importálnak (az eredeti CARLA csomagot), nem a helyi
`carla_env.navigation`-t. Vagy javítsd az importokat, vagy töröld őket.

---

# 2. Gyakorlati receptek — hogyan módosítsd

## 2.1 Új jutalomfüggvény hozzáadása

📍 `carla_env/rewards.py`

```python
def reward_fn6(env):
    """Példa: reward_fn5 + a kanyarokban külön lassítási bónusz."""
    base = reward_fn5(env)

    # Ha kanyar-manőver aktív, jutalmazd az alacsonyabb sebességet
    if env.current_road_maneuver in (RoadOption.LEFT, RoadOption.RIGHT):
        speed = env.vehicle.get_speed()
        turn_bonus = max(0.0, 1.0 - abs(speed - 15.0) / 15.0)
        return base * (0.7 + 0.3 * turn_bonus)
    return base

reward_functions["reward_fn6"] = create_reward_fn(reward_fn6)
```

Majd `config.py`: `"reward_fn": "reward_fn6"`.

**Fontos:** ha megváltoztatod a jutalmat, **ne folytass régi modellből** (`RELOAD_MODEL = False`)
— a replay bufferben lévő régi jutalmak összezavarnák a tanulást.

## 2.2 Új állapotelem hozzáadása

Példa: távolság a következő kereszteződésig.

**1) `config.py`:**
```python
STATE = [..., "distance_to_intersection"]
```

**2) `state_commons.py`** — bővítsd a `measure_flags`-et (használd az egyik `False` helyet,
a 6. indexet):
```python
measure_flags = [..., "distance_to_intersection" in measurements_to_include,  # [6]
                 False, False, False, False]
```

**3) `create_observation_space()`-ben:**
```python
if measure_flags[6]: low.append(0), high.append(100)
```
(a `vehicle_measures` Box-ba, a többi elé/mögé — a sorrendnek egyeznie kell az
`encode_state`-tel!)

**4) `encode_state()`-ben:**
```python
if measure_flags[6]:
    vehicle_measures.append(env.distance_to_intersection)
```

**5) `carla_route_env.py` `step()`-jében** számold ki:
```python
self.distance_to_intersection = 100.0
for i in range(self.current_waypoint_index, min(self.current_waypoint_index + 100, len(self.route_waypoints))):
    if self.route_waypoints[i][1] != RoadOption.LANEFOLLOW:
        self.distance_to_intersection = float(i - self.current_waypoint_index)
        break
```

⚠ Az állapottér megváltozása miatt **újra kell tanítani** — a mentett modell nem tölthető be.

## 2.3 Fék hozzáadása az akciótérhez

📍 `carla_route_env.py:46-47`

```python
self.action_space = gym.spaces.Box(
    np.array([-1, -1], dtype=np.float32),      # steer, throttle_brake
    np.array([ 1,  1], dtype=np.float32), dtype=np.float32)
```

Majd a `step()`-ben:
```python
steer, throttle_brake = [float(a) for a in action]
if throttle_brake >= 0:
    throttle, brake = throttle_brake, 0.0
else:
    throttle, brake = 0.0, -throttle_brake

self.vehicle.control.steer = smooth_action(self.vehicle.control.steer, steer, self.action_smoothing)
self.vehicle.control.throttle = smooth_action(self.vehicle.control.throttle, throttle, self.action_smoothing)
self.vehicle.control.brake = smooth_action(self.vehicle.control.brake, brake, self.action_smoothing)
```

Ezzel az egy dimenzióban kezelt gáz/fék (`-1 = teljes fék`, `+1 = padlógáz`) elkerüli,
hogy az ágens egyszerre gázt és féket adjon.

## 2.4 Térképváltás

1. `wrappers.py:408` → `client.load_world(map_name)`, `map_name` paraméterből.
2. Az `intersection_routes` és `eval_routes` spawn-index párokat **újra kell választani** —
   a Town02 indexei más térképen mást jelentenek.
3. ⚠ Ha többsávos a térkép: előbb javítsd a `RoadOption` enumot (1.3 pont).

Spawn pontok felderítése:
```python
import carla
client = carla.Client("localhost", 2000); client.set_timeout(30.0)
world = client.load_world("Town04")
for i, sp in enumerate(world.get_map().get_spawn_points()):
    print(i, sp.location)
```

## 2.5 Kiértékelés (eval) script

Jelenleg **nincs eval script**, pedig a környezet támogatja (`eval=True`). Készíts egyet:

```python
# eval.py
from stable_baselines3 import SAC
from carla_env.envs.carla_route_env import CarlaRouteEnv
from carla_env.state_commons import create_encode_state_fn, load_vae
from carla_env.rewards import reward_functions
from config import LSIZE, STATE, OBS_RES, FPS, ACTION_SMOOTHING, CONFIG

vae = load_vae('./vae/model', LSIZE)
observation_space, encode_state_fn, decode_vae_fn = create_encode_state_fn(vae, STATE)

env = CarlaRouteEnv(obs_res=OBS_RES, reward_fn=reward_functions[CONFIG["reward_fn"]],
                    observation_space=observation_space, encode_state_fn=encode_state_fn,
                    decode_vae_fn=decode_vae_fn, fps=FPS,
                    action_smoothing=ACTION_SMOOTHING, action_space_type='continuous',
                    eval=True, activate_spectator=True, activate_render=True)

model = SAC.load("tensorboard/SAC_.../model_final.zip", env=env, device='cuda')

for episode in range(4):                       # a 4 eval_routes útvonal
    obs, info = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)   # ← determinisztikus!
        obs, reward, done, truncated, info = env.step(action)
    print(f"Ep {episode}: távolság={info['total_distance']:.0f} m, "
          f"útvonal={info['routes_completed']:.2f}, átl. seb.={info['avg_speed']:.1f} km/h")
env.close()
```

**`deterministic=True`** — kiértékeléskor ne legyen felfedezési zaj.

## 2.6 Gyorsabb tanítás

| Beavatkozás | Várható nyereség |
|-------------|-----------------|
| `ACTIVATE_RENDER = False`, `ACTIVATE_SPECTATOR = False` | **nagy** — nincs 1920×1080 kamera + pygame |
| A gráf gyorsítótárazása (1.4) | **nagy** — epizódonként több száz ms |
| VAE GPU-ra (1.5) | közepes |
| `time.sleep()`-ek eltávolítása (1.7) | közepes — 0.6 s/epizód |
| CARLA indítása `-quality-level=Low -RenderOffScreen` kapcsolókkal | **nagy** |

```bash
./CarlaUE4.sh -quality-level=Low -RenderOffScreen -carla-server
```

---

# 3. ROS2 integráció — a következő nagy lépés

A repó neve `RL-carla-ROS2`, és a legutóbbi commit `"vedes done folytatas ros implementation"`.
**Jelenleg nulla ROS2 kód van a projektben.**

## Miért érdemes ROS2-t használni?

1. **Modularitás** — az érzékelés, tervezés, vezérlés külön node-okba kerül; külön
   fejleszthetők és tesztelhetők.
2. **Valós hardver felé való átjárhatóság** — ugyanaz a node futhat szimulátorral és igazi
   autóval, ha ugyanazokat a topicokat használja.
3. **Eszköztár** — `rviz2` (vizualizáció), `rosbag2` (adatrögzítés/visszajátszás),
   `tf2` (koordinátatranszformációk).

## Lehetséges architektúra

```
┌──────────────────────────────────────────────────────────────────────┐
│                     CARLA szerver (Town02)                           │
└───────────────────────────▲──────────────────────────────────────────┘
                            │ CARLA Python API
┌───────────────────────────┴──────────────────────────────────────────┐
│  carla_bridge_node                                                   │
│  - kamera → /carla/ego/camera/image_raw     (sensor_msgs/Image)      │
│  - odometria → /carla/ego/odometry          (nav_msgs/Odometry)      │
│  - ütközés → /carla/ego/collision           (std_msgs/Bool)          │
│  - feliratkozik: /carla/ego/control         (ackermann_msgs vagy     │
│                                              carla_msgs/CarlaEgo...) │
└──────────────────────────────────────────────────────────────────────┘
        │ Image                    │ Odometry           ▲ control
        ▼                          ▼                    │
┌──────────────────┐   ┌───────────────────┐   ┌────────┴─────────────┐
│ perception_node  │   │ route_planner_node│   │  rl_agent_node       │
│ VAE encode       │   │ A* + RoadOption   │   │  SAC.predict()       │
│ → /perception/   │   │ → /planning/      │   │  feliratkozik mind-  │
│    latent (64)   │   │    waypoints      │   │  kettőre, publikál   │
└──────────────────┘   └───────────────────┘   │  vezérlést           │
                                                └──────────────────────┘
```

## Konkrét lépések

**1) Kezdd a CARLA hivatalos ROS bridge-ével** — ne írj sajátot nulláról:
```bash
# https://github.com/carla-simulator/ros-bridge
ros2 launch carla_ros_bridge carla_ros_bridge_with_example_ego_vehicle.launch.py
```
Ez már publikálja a kamerát, odometriát, és fogadja a vezérlést.

**2) `perception_node`** — a `state_commons.py` `encode_state` VAE része:
```python
class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')
        self.vae = load_vae('./vae/model', 64)
        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, '/carla/ego_vehicle/rgb_front/image',
                                            self.on_image, 10)
        self.pub = self.create_publisher(Float32MultiArray, '/perception/latent', 10)

    def on_image(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
        frame = cv2.resize(frame, (160, 80))
        with torch.no_grad():
            mu, logvar = self.vae.encode(preprocess_frame(frame))
        out = Float32MultiArray(data=mu[0].cpu().numpy().tolist())
        self.pub.publish(out)
```

**3) `rl_agent_node`** — betölti a mentett SAC modellt és `predict`-el:
```python
action, _ = self.model.predict(obs_dict, deterministic=True)
```

**4) A tanítás maradhat a jelenlegi, ROS-mentes úton.** Ez fontos döntés: a ROS2 réteg
késleltetést és nem-determinizmust visz be, ami tanításnál káros. Bevált minta:
**tanítás közvetlen API-val (gyors, determinisztikus), telepítés ROS2-vel (moduláris).**

## Mit érdemes előbb megcsinálni?

Javasolt sorrend, mielőtt a ROS2-be kezdesz:

1. **Eval script** (2.5) — enélkül nem tudod megmérni, mennyit ér a modelled.
2. **A 🔴 hibák javítása** (1.1, 1.2) — különben az eredmények zajosak.
3. **Gráf-gyorsítótárazás** (1.4) — több kísérletet tudsz futtatni ugyanannyi idő alatt.
4. **Egy stabil, jól teljesítő modell betanítása.**
5. **Aztán** ROS2 telepítési réteg.

Egy ROS2 réteg egy nem működő ágens fölött nem ér semmit; egy működő ágens fölött viszont
azonnal demonstrálható rendszert ad.

---

# 4. Kísérleti ötletek a modell javítására

| Ötlet | Mit vársz tőle | Hol nyúlj hozzá |
|-------|---------------|----------------|
| **Szemantikus szegmentációs kamera** | Kevesebb vizuális zaj → jobb VAE latent → gyorsabb tanulás | `STATE`-be `"seg_camera"`; a `custom_palette` már kész (de vektorizáld!) |
| **Framestacking** (több egymást követő kép) | Sebességérzet a képből, jobb dinamika-becslés | `VecFrameStack` wrapper vagy saját deque |
| **Időjárás-randomizáció** | Általánosítás | `carla_route_env.py:77` sor kikommentezésének feloldása, véletlen időjárás `reset()`-ben |
| **Nagyobb `gamma` (0.99)** | Hosszabb előrelátás — korábbi lassítás kanyar előtt | `config.py` |
| **Több lookahead waypoint (15 → 30)** | Korábban látja a kanyart | `state_commons.py` (3 helyen a 15-ös szám) |
| **`mu` használata a `reparameterize` helyett** | Determinisztikus állapot → stabilabb tanulás | `state_commons.py:73` |
| **Kisebb `action_smoothing` (0.75 → 0.5)** | Fürgébb reakció | `config.py` |
| **Forgalom hozzáadása** (más járművek) | Realisztikusabb feladat | `wrappers.py`-ba egy `spawn_npc` függvény |
| **Közlekedési lámpák figyelembe vétele** | Teljesebb feladat | Új állapotelem + jutalombüntetés piroson áthajtásért |

**Módszertani tanács:** egyszerre **egy dolgot** változtass, és mindig ugyanazzal az
eval scripttel mérj. Az RL nagyon zajos — ha kettőt változtatsz egyszerre, nem fogod tudni,
melyik segített.

---

# 5. Hibakeresési checklist

**"Nem indul el / timeout"**
- Fut a CARLA szerver? `ps aux | grep CarlaUE4`
- Jó a port? (alapból 2000)
- Az első `load_world` lassú lehet — a timeout 60 s, ez általában elég.

**"Az autó azonnal megáll és `Vehicle stopped` jön"**
- Lásd 1.2 pont (`low_speed_timer` hiba).
- Vagy `learning_starts=10000` alatt vagy még — az első 10 000 lépés véletlen akció.

**"A jutalom nem nő"**
- Nézd meg a `custom/routes_completed`-et a TensorBoardon, nem csak az `ep_rew_mean`-t.
- Nézd meg a HUD-on a VAE-dekódolt képet: felismerhető rajta az út?
- 130 000 lépés ehhez a feladathoz **kevés lehet** — tipikusan 300k–1M kell.

**"CUDA out of memory"**
- `buffer_size=300000` × (64+4+30+1) float ≈ néhány GB RAM (nem GPU) — de a `batch_size`
  és `net_arch` GPU-t eszik. Csökkentsd a `batch_size`-t.

**"Furcsán viselkedik a modell folytatás után"**
- `SAC.load` **nem tölti vissza a replay buffert** (külön `.pkl` fájl kellene hozzá).
  Tehát folytatáskor üres bufferrel indul, és az első `learning_starts` lépés újra
  véletlen adatgyűjtés. Ez normális, de átmeneti teljesítményesést okoz.

---

# 6. Összefoglaló: hol tart a projekt?

**Ami kész és működik:**
- ✅ Teljes RL pipeline: környezet, jutalom, állapot-kódolás, tanítás, logolás
- ✅ VAE-alapú vizuális reprezentáció
- ✅ A\*-alapú globális útvonaltervezés kereszteződés-manőverekkel
- ✅ Checkpointolás, TensorBoard integráció, konfiguráció mentése
- ✅ Vizuális hibakeresés (HUD, útvonal-kivetítés, VAE rekonstrukció)

**Ami hiányzik:**
- ❌ Eval / demo script
- ❌ ROS2 integráció (a repó nevében ígért rész)
- ❌ Fék az akciótérben
- ❌ Forgalom, gyalogosok, közlekedési lámpák
- ❌ Tesztek
- ❌ README a repó gyökerében

**Ami el van rontva:** lásd az 1. fejezetet — főleg `close()`, `low_speed_timer`,
`RoadOption` enum, gráf-újraépítés.

---

⬅️ Vissza: [00-attekintes.md](00-attekintes.md)
