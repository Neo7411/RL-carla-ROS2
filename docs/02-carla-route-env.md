# 02 — `carla_env/envs/carla_route_env.py` — ★ A központi fájl

Ez a projekt szíve: itt találkozik a CARLA szimulátor és a Gymnasium RL interfész.
459 sor, végigmegyünk rajta blokkonként.

---

## Modulok importja

```python
import time
import gymnasium as gym
import pygame
import cv2
from pygame.locals import *

from carla_env.tools.hud import HUD
from carla_env.navigation.planner import RoadOption, compute_route_waypoints
from carla_env.wrappers import *

import carla
from collections import deque
import itertools
```

- `gymnasium` — a Gym utódja. Ez definiálja a standard RL interfészt (`reset`, `step`,
  `observation_space`, `action_space`).
- `from carla_env.wrappers import *` — innen jön a `World`, `Vehicle`, `Camera`, `Lidar`,
  `vector`, `distance_to_line`, `smooth_action`, `sensor_transforms`, `np` stb.
  (A `numpy`-t is ez hozza be `np` néven, mert a `wrappers.py` importálja!)

### Előre definiált útvonalak

```python
intersection_routes = itertools.cycle(
    [(57, 81), (70, 11), (70, 12), (78, 68), (74, 41), (42, 73),
     (71, 62), (74, 40), (71, 77), (6, 12), (65, 52), (63, 80)])
eval_routes = itertools.cycle([(48, 21), (0, 72), (28, 83), (61, 39)])
```

- **Mik ezek a számok?** A Town02 térkép **spawn pontjainak indexei**. A `(57, 81)` azt
  jelenti: indulj az 57-es spawn ponttól, menj a 81-es spawn pontig.
- **Miért "intersection"?** Ezek olyan párok, amelyek között az útvonal **kereszteződéseket
  tartalmaz**. Ha csak véletlen párokat használnánk, az ágens ritkán találkozna kanyarral,
  és sose tanulná meg a bekanyarodást.
- `itertools.cycle` — végtelen körkörös iterátor. A `next(intersection_routes)` mindig a
  következő párt adja, a végén újrakezdi.

```python
discrete_actions = {
    0: [-1, 1], 1: [0, 1], 2: [1, 1], 3: [0, 0],
}
```
Diszkrét akciótér esetén használható (balra+gáz, egyenesen+gáz, jobbra+gáz, semmi).
**Jelenleg nem használt**, mert `action_space_type='continuous'`.

---

## `__init__` — a környezet felépítése (26–116. sor)

### Paraméterek (26–35. sor)

```python
def __init__(self, host="127.0.0.1", port=2000,
             viewer_res=(1920, 1080),   # a pygame ablak mérete
             obs_res=(160, 80),         # a dashcam (megfigyelés) felbontása
             reward_fn=None,
             observation_space=None,
             encode_state_fn=None, decode_vae_fn=None,
             fps=15, action_smoothing=0.0, action_space_type="continuous",
             activate_spectator=True,
             activate_lidar=False,      # LIDAR alapból KI
             eval=False,                # eval mód: fix útvonalak, nincs újragenerálás
             activate_render=True):
```

**Dependency injection minta:** a `reward_fn` és `encode_state_fn` kívülről jön be. Ez azt
jelenti, hogy **kicserélheted a jutalmat vagy az állapotot anélkül, hogy ezt a fájlt
módosítanád**. Ez a projekt egyik legjobb design döntése.

### Akciótér

```python
if self.action_space_type == "continuous":
    self.action_space = gym.spaces.Box(
        np.array([-1, 0], dtype=np.float32),
        np.array([1, 1], dtype=np.float32), dtype=np.float32)  # steer, throttle
```

Két folytonos szám:
- **steer** ∈ [-1, 1] — -1 = teljesen balra, 0 = egyenesen, +1 = teljesen jobbra
- **throttle** ∈ [0, 1] — 0 = nincs gáz, 1 = padlógáz

> **Figyeld meg: nincs FÉK!** Az autó csak gurulással lassul. Ez tudatos egyszerűsítés
> — kevesebb dimenzió = könnyebb tanulás. Egy továbbfejlesztési lehetőség a fék hozzáadása.

### Callback-ek védett beállítása
```python
self.encode_state_fn = (lambda x: x) if not callable(encode_state_fn) else encode_state_fn
self.decode_vae_fn = None if not callable(decode_vae_fn) else decode_vae_fn
self.reward_fn = (lambda x: 0) if not callable(reward_fn) else reward_fn
```

Ez egy védőháló: ha nem adsz meg jutalomfüggvényt, a környezet nem omlik össze, csak
mindig 0-t ad vissza. Hasznos teszteléshez.

### CARLA csatlakozás (63–75. sor)

```python
self.client = carla.Client(host, port)
self.client.set_timeout(60.0)          # 60 mp türelem (a térkép betöltése lassú)

self.world = World(self.client)        # ← ez betölti a Town02-t! (wrappers.py:408)

settings = self.world.get_settings()
settings.fixed_delta_seconds = 1 / self.fps    # 1/20 = 0.05 s lépésenként
settings.synchronous_mode = True
self.world.apply_settings(settings)
self.client.reload_world(False)        # újratölti a térképet, de MEGTARTJA a beállításokat
```

**A szinkron mód a legfontosabb beállítás itt.** Két üzemmód van a CARLA-ban:

| Mód | Működés | Alkalmas RL-hez? |
|-----|---------|-----------------|
| Aszinkron (alapértelmezett) | a szerver saját tempójában fut, a kliens lemarad | ❌ nem — nem-determinisztikus |
| **Szinkron** | a szerver **vár** a kliensre; a `world.tick()` lépteti előre | ✅ igen |

`fixed_delta_seconds = 0.05` — minden `tick()` pontosan 0.05 szimulált másodpercet léptet,
függetlenül attól, hogy valós időben mennyi ideig tartott. Ez teszi **reprodukálhatóvá** a
tanítást.

```python
# self.world.set_weather(carla.WeatherParameters.MidRainyNoon)
```
Kikommentezve — ha bekapcsolod, esős időben tanul. Domain randomization-höz hasznos lenne.

### Jármű létrehozása (81–83. sor)

```python
self.vehicle = Vehicle(self.world, self.world.map.get_spawn_points()[0],
                       on_collision_fn=lambda e: self._on_collision(e),
                       on_invasion_fn=lambda e: self._on_invasion(e))
```

Egy Tesla Model 3 (a `wrappers.py`-ban a default) a 0. spawn ponton. A két callback:
- `_on_collision` — ütközéskor hívódik → epizód vége
- `_on_invasion` — sávelhagyáskor → csak HUD üzenet, nincs következménye

### Renderelés inicializálása (86–93. sor)

```python
if self.activate_render:
    pygame.init()
    pygame.font.init()
    self.display = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)
    self.clock = pygame.time.Clock()
    self.hud = HUD(width, height)
    self.hud.set_vehicle(self.vehicle)
    self.world.on_tick(self.hud.on_world_tick)
```

`DOUBLEBUF` — dupla pufferelés, villódzásmentes rajzolás. A `on_tick` regisztrál egy
callback-et, ami minden szimulációs lépésnél frissíti a HUD szerver-oldali adatait (FPS,
szimulációs idő).

### Kamerák (95–112. sor)

```python
seg_settings = {}
if "seg_camera" in self.observation_space.keys():
    seg_settings.update({
        'camera_type': "sensor.camera.semantic_segmentation",
        'custom_palette': True
    })
self.dashcam = Camera(self.world, out_width, out_height,
                      transform=sensor_transforms["dashboard"],
                      attach_to=self.vehicle,
                      on_recv_image=lambda e: self._set_observation_image(e),
                      **seg_settings)
```

**Három kamera lehet:**

| Kamera | Felbontás | Pozíció | Célja |
|--------|-----------|---------|-------|
| `dashcam` | 160×80 | motorháztetőn (x=1.6, z=1.7) | **ez megy a VAE-be → az ágens állapotába** |
| `camera` (spectator) | 1920×1080 | hátulról-felülről (x=-5.5, z=2.8, pitch=-15°) | csak megjelenítés |
| `lidar` | – | tetőn (z=2.4) | jelenleg kikapcsolva |

A `seg_settings` blokk akkor aktiválódna, ha a `STATE`-ben lenne `seg_camera` — akkor
szemantikus szegmentációs kamerát használna (minden pixel egy osztálycímke: út, járda,
autó...). Jelenleg **nem aktív**, mert a `STATE` nem tartalmazza.

**Fontos, hogy a kamerák aszinkron callback-eken keresztül adnak képet:** a
`_set_observation_image` egy pufferbe teszi, és a `step()` majd kiolvassa (lásd `_get_observation`).

### Az utolsó sor (116)

```python
self.reset()
```

Már a konstruktorban lefut egy reset. Tehát amint létrehozod a környezetet, az autó már a
pályán van egy generált útvonallal.

---

## `reset()` — új epizód (118–147. sor)

```python
def reset(self, seed=None, options=None):
    self.num_routes_completed = -1
    self.episode_idx += 1
    self.new_route()
```

> **Miért `-1`?** Mert a `new_route()` a végén `+1`-et ad hozzá (175. sor), így az első
> útvonalnál 0 lesz. A `routes_completed` metrika ebből számol.

> **Miért `episode_idx = -2` a konstruktorban (52. sor)?** Mert a `__init__` végén futó
> `reset()` -1-re állítja, majd a `step(None)` hívás után a *tényleges* első epizód 0 lesz.
> Ez egy kis "off-by-two" trükk, hogy a kiírt epizódszámok 0-tól induljanak.

```python
    self.terminal_state = False   # kudarccal ért véget?
    self.success_state = False    # sikerrel ért véget?
```

**Két külön flag** — ez fontos! Így meg lehet különböztetni:
- `terminal_state` = elrontotta (ütközés, lesodródás, megállás, túl gyors) → -10 büntetés
- `success_state` = teljesítette (elérte a max távolságot vagy eval módban a célt)

```python
    self.observation = self.observation_buffer = None
    self.viewer_image = self.viewer_image_buffer = None
    self.lidar_data = self.lidar_data_buffer = None
    self.step_count = 0

    # metrikák nullázása
    self.total_reward = 0.0
    self.previous_location = self.vehicle.get_transform().location
    self.distance_traveled = 0.0
    self.center_lane_deviation = 0.0
    self.speed_accum = 0.0
    self.routes_completed = 0.0
    self.world.tick()

    time.sleep(0.2)
    obs, _, _, _, info = self.step(None)     # ← "üres" lépés az első megfigyelésért
    time.sleep(0.2)
    return obs, info
```

**A `self.step(None)` trükk:** a Gym interfész szerint a `reset()` vissza kell adja az első
megfigyelést. De a megfigyelés előállításához le kell futtatni a kamera-olvasást,
waypoint-keresést stb. — pont azt, amit a `step()` csinál. Ezért `action=None`-nal hívja:
így nem alkalmaz vezérlést, csak legenerálja az állapotot.

A `time.sleep(0.2)` hívások azért kellenek, mert a CARLA szenzor-callback-ek aszinkron
szálakon futnak, és kell egy kis idő, hogy megérkezzen az első kép. **Ez törékeny megoldás**
— lásd [07-tovabbfejlesztes.md](07-tovabbfejlesztes.md).

---

## `new_route()` — útvonal generálás (149–178. sor)

```python
def new_route(self):
    self.vehicle.control.steer = float(0.0)
    self.vehicle.control.throttle = float(0.0)
    self.vehicle.set_simulate_physics(False)   # fizika KI a teleportáláshoz
```

**Miért kell kikapcsolni a fizikát?** Mert ha teleportálsz egy mozgó autót, a fizikai motor
megőrizné a sebességét és pörögne/repülne. Fizika nélkül tiszta "áthelyezés".

### Spawn pontok kiválasztása (155–162. sor)

```python
if not self.eval:
    if self.episode_idx % 2 == 0 and self.num_routes_completed == -1:
        spawn_points_list = [self.world.map.get_spawn_points()[index]
                             for index in next(intersection_routes)]
    else:
        spawn_points_list = np.random.choice(self.world.map.get_spawn_points(), 2, replace=False)
else:
    spawn_points_list = [self.world.map.get_spawn_points()[index] for index in next(eval_routes)]
```

Ez egy **curriculum/keverési stratégia**:

| Feltétel | Útvonal típusa | Miért |
|----------|---------------|-------|
| páros epizód **és** ez az első útvonal az epizódban | előre definiált **kereszteződéses** útvonal | garantált kanyar-tanulás |
| páratlan epizód **vagy** epizódon belüli következő útvonal | **véletlen** két spawn pont | változatosság, általánosítás |
| `eval=True` | fix `eval_routes` | összehasonlítható kiértékelés |

A `num_routes_completed == -1` feltétel biztosítja, hogy **csak az epizód elején** kapjon
kereszteződéses útvonalat — ha menet közben teljesíti és újat kap, az véletlen lesz.

### Az útvonal kiszámítása (163–170. sor)

```python
route_length = 1
while route_length <= 1:
    self.start_wp, self.end_wp = [self.world.map.get_waypoint(spawn.location)
                                  for spawn in spawn_points_list]
    self.route_waypoints = compute_route_waypoints(
        self.world.map, self.start_wp, self.end_wp, resolution=1.0)
    route_length = len(self.route_waypoints)
    if route_length <= 1:
        spawn_points_list = np.random.choice(self.world.map.get_spawn_points(), 2, replace=False)
```

- `get_waypoint(location)` — a spawn pontot "ráilleszti" a legközelebbi sávra.
- `compute_route_waypoints(...)` — az A\* útvonaltervező (lásd [05-navigation.md](05-navigation.md)).
  `resolution=1.0` → **1 méterenként egy waypoint**.
- A `while` ciklus: ha az útvonal degenerált (0 vagy 1 pont — pl. mert a két spawn pont
  ugyanaz vagy nem érhető el), új véletlen pontokkal újrapróbálja.

```python
    self.distance_from_center_history = deque(maxlen=30)
```
Egy **30 elemű csúszóablak** a sávközéptől való eltérésekről. A jutalomfüggvény ennek a
**szórását** használja — ez bünteti a "cikkcakkozást" akkor is, ha átlagosan középen van.

```python
    self.current_waypoint_index = 0
    self.num_routes_completed += 1
    self.vehicle.set_transform(self.start_wp.transform)   # teleport a startra
    time.sleep(0.2)
    self.vehicle.set_simulate_physics(True)               # fizika vissza
```

---

## `step(action)` — ★ a legfontosabb metódus (253–365. sor)

Minden RL lépés itt zajlik. Nézzük szakaszonként.

### 1) Akció alkalmazása (257–273. sor)

```python
if action is not None:
    # Útvonal vége → új útvonal (láncolás!)
    if self.current_waypoint_index >= len(self.route_waypoints) - 1:
        if not self.eval:
            self.new_route()          # tanításkor: új útvonal, az epizód FOLYTATÓDIK
        else:
            self.success_state = True # eval-ban: siker, epizód vége

    if self.action_space_type == "continuous":
        steer, throttle = [float(a) for a in action]
    elif self.action_space_type == "discrete":
        steer, throttle = discrete_actions[action]

    self.vehicle.control.steer = smooth_action(self.vehicle.control.steer, steer, self.action_smoothing)
    self.vehicle.control.throttle = smooth_action(self.vehicle.control.throttle, throttle, self.action_smoothing)
```

**Kulcsfontosságú tervezési döntés:** tanításkor az útvonal végén **nem ér véget az epizód**,
hanem kap egy új útvonalat. Ezért lehet a `routes_completed` metrika 1-nél nagyobb.
Az epizód csak akkor ér véget, ha:
- hibázik (`terminal_state`), vagy
- eléri a `max_distance = 3000 m`-t (`success_state`).

Ez sokkal jobb tanulást ad: hosszú, folyamatos vezetési szekvenciákat, nem 30 méteres
darabkákat.

**A `smooth_action`** (wrappers.py:88):
```python
def smooth_action(old_value, new_value, smooth_factor):
    return old_value * smooth_factor + new_value * (1.0 - smooth_factor)
```
0.75-tel: az új parancs csak 25%-ban érvényesül azonnal. Exponenciális simítás.

### 2) A szimuláció léptetése (274–275. sor)

```python
self.world.tick()
```

Ez egyetlen sor, de sok minden történik (lásd `wrappers.py` `World.tick`, 412. sor):
```python
def tick(self):
    for actor in list(self.actor_list):
        actor.tick()      # ← a Vehicle.tick() itt alkalmazza a control-t!
    self.world.tick()     # ← a CARLA szerver 0.05 s-ot előre lép
```

Tehát a sorrend: **először minden szereplő alkalmazza a vezérlését, aztán lép a fizika.**

### 3) Szenzoradatok beolvasása (277–283. sor)

```python
self.observation = self._get_observation()
if self.activate_spectator:
    self.viewer_image = self._get_viewer_image()
if self.activate_lidar:
    self.lidar_data = self._get_lidar_data()
```

A `_get_observation` (419–424. sor):
```python
def _get_observation(self):
    while self.observation_buffer is None:   # ⚠ BUSY WAIT
        pass
    obs = self.observation_buffer.copy()
    self.observation_buffer = None
    return obs
```

**Ez egy "busy wait" (aktív várakozás)** — 100%-on pörgeti a CPU magot, amíg a kamera
callback-je meg nem érkezik. Működik, de pazarló, és ha a kamera valamiért nem küld képet,
**örökre lefagy**. Javítási ötlet a [07-tovabbfejlesztes.md](07-tovabbfejlesztes.md)-ben.

### 4) Waypoint követés (285–301. sor) — ez a legtrükkösebb rész

```python
transform = self.vehicle.get_transform()

self.prev_waypoint_index = self.current_waypoint_index
waypoint_index = self.current_waypoint_index
for _ in range(len(self.route_waypoints)):
    next_waypoint_index = waypoint_index + 1
    wp, _ = self.route_waypoints[next_waypoint_index % len(self.route_waypoints)]
    dot = np.dot(vector(wp.transform.get_forward_vector())[:2],
                 vector(transform.location - wp.transform.location)[:2])
    if dot > 0.0:                 # átmentünk a waypointon?
        waypoint_index += 1
    else:
        break
self.current_waypoint_index = waypoint_index
```

**Hogyan dönti el, hogy elhagytunk-e egy útpontot?** Skaláris szorzattal:

```
                    waypoint irányvektora (forward)
                              ↑
                              │
        ─────────────────●────┼──────────────  a waypointon átmenő "kapu" vonal
                     waypoint │
                              │
      dot < 0                 │              dot > 0
   (még előtte vagyunk)       │           (már elhagytuk)
```

- `v1` = a waypoint előre mutató iránya
- `v2` = a waypointtól az autóig mutató vektor
- Ha `v1 · v2 > 0`, akkor az autó a waypoint **előtte lévő** félterében van → elhagytuk.

A ciklus **több waypointot is ugorhat egy lépésben** — ha az autó gyorsan megy (25 km/h ≈
7 m/s, 0.05 s alatt 0.35 m), egyszerre nem sok, de kanyarban vagy nagy sebességnél igen.

### 5) Útvonal-haladás számítása (303–311. sor)

```python
if self.current_waypoint_index < len(self.route_waypoints) - 1:
    self.next_waypoint, self.next_road_maneuver = self.route_waypoints[
        (self.current_waypoint_index + 1) % len(self.route_waypoints)]

self.current_waypoint, self.current_road_maneuver = self.route_waypoints[
    self.current_waypoint_index % len(self.route_waypoints)]

self.routes_completed = self.num_routes_completed + \
    (self.current_waypoint_index + 1) / len(self.route_waypoints)
```

A `routes_completed` **tört szám**: pl. `2.45` = 2 teljes útvonal + a harmadik 45%-a.
Ez a fő teljesítménymutató a TensorBoardon.

> **Figyeld meg az `if`-et a 304. sorban:** a `next_waypoint` **nem frissül**, ha elértük az
> utolsó előtti pontot. Így a régi értéke marad meg — a `distance_to_line` számításnál ez
> apró pontatlanságot okozhat az útvonal legvégén.

### 6) Sávközép-eltérés (313–317. sor)

```python
self.distance_from_center = distance_to_line(
    vector(self.current_waypoint.transform.location),
    vector(self.next_waypoint.transform.location),
    vector(transform.location))
self.center_lane_deviation += self.distance_from_center
```

A `distance_to_line` (wrappers.py:71) pont-egyenes távolságot számol vektoriális szorzattal:
```
távolság = |(B-A) × (A-p)| / |B-A|
```
ahol A = jelenlegi waypoint, B = következő waypoint, p = az autó pozíciója.
A `p[2] = 0` sor kilapítja z-ben (csak 2D távolság érdekes).

### 7) További metrikák (319–331. sor)

```python
if action is not None:
    self.distance_traveled += self.previous_location.distance(transform.location)
self.previous_location = transform.location

self.speed_accum += self.vehicle.get_speed()

if self.distance_traveled >= self.max_distance and not self.eval:
    self.success_state = True        # 3000 m után: siker

self.distance_from_center_history.append(self.distance_from_center)
```

### 8) Jutalom (333–335. sor)

```python
self.last_reward = self.reward_fn(self)
self.total_reward += self.last_reward
```

**Figyeld meg: a jutalomfüggvény az egész `env` objektumot kapja meg.** Így hozzáfér
mindenhez (sebesség, eltérés, waypoint index...). Kényelmes, de erős csatolást jelent —
a jutalomfüggvény ismeri a környezet belső mezőneveit.

**Fontos:** a `reward_fn` **mellékhatással is jár** — beállíthatja a `self.terminal_state`-et
(lásd `rewards.py` `create_reward_fn`). Tehát az epizód végét részben a jutalomfüggvény
dönti el!

### 9) Állapot kódolása (337–341. sor)

```python
encoded_state = self.encode_state_fn(self)
if self.decode_vae_fn:
    self.observation_decoded = self.decode_vae_fn(encoded_state['vae_latent'])
self.step_count += 1
```

Itt fut le a VAE encoder (kép → 64 szám) és állnak össze a mérőszámok.
A `decode_vae_fn` csak a képernyőre rajzoláshoz kell — látni lehet, mit "értett meg" a VAE.

### 10) Renderelés és ESC figyelés (348–353. sor)

```python
if self.activate_render:
    pygame.event.pump()
    if pygame.key.get_pressed()[K_ESCAPE]:
        self.terminal_state = True
    self.render()
```

`pygame.event.pump()` — feldolgozza a beérkezett eseményeket (enélkül az ablak "nem
reagálna" és a rendszer lefagyottnak hinné).

### 11) Visszatérés (355–365. sor)

```python
info = {
    "closed": self.closed,
    'total_reward': self.total_reward,
    'routes_completed': self.routes_completed,
    'total_distance': self.distance_traveled,
    'avg_center_dev': (self.center_lane_deviation / self.step_count),
    'avg_speed': (self.speed_accum / self.step_count),
    'mean_reward': (self.total_reward / self.step_count)
}
done = self.terminal_state or self.success_state
return encoded_state, self.last_reward, done, False, info
```

**A Gymnasium 5 elemű visszatérése:**

| Pozíció | Név | Nálunk |
|---------|-----|--------|
| 1 | `observation` | `encoded_state` dict |
| 2 | `reward` | `self.last_reward` |
| 3 | `terminated` | `done` |
| 4 | `truncated` | **mindig `False`** |
| 5 | `info` | metrikák szótára |

> **Elméleti megjegyzés:** a `truncated` (idő/limit miatti megszakítás) és a `terminated`
> (a feladat valóban véget ért) megkülönböztetése fontos a bootstrapping szempontjából.
> A 3000 m elérése valójában `truncated` lenne, nem `terminated`. Így az algoritmus úgy
> kezeli, mintha a világ ott véget érne, és nem folytatja a value becslést. Kisebb torzítás,
> de ez javítható lenne.

---

## `render()` — a képernyőkép (188–249. sor)

```python
if mode == "rgb_array_no_hud":  return self.viewer_image
elif mode == "rgb_array":       return np.array(pygame.surfarray.array3d(self.display), ...)
elif mode == "state_pixels":    return self.observation
```
Ezek a nem-interaktív módok (pl. videófelvételhez). Alapból `mode="human"`, tehát megy tovább.

```python
self.clock.tick()
self.hud.tick(self.world, self.clock)
```

Manőver szöveggé alakítása (202–211. sor), majd:

```python
self.extra_info.extend([
    "Episode {}".format(self.episode_idx),
    "Reward: % 19.2f" % self.last_reward,
    ...
    "Avg speed:      % 7.2f km/h" % (self.speed_accum / self.step_count),
])
```

**Az elrendezés:**
```
┌──────────────────────────────────────────────────────────────┐
│ ┌─ HUD info panel     │                  ┌────────┬────────┐ │
│ │ Server: 20 FPS      │                  │ VAE    │ dashcam│ │
│ │ Speed: 24 km/h      │                  │ dekód  │ 160×80 │ │
│ │ Episode 42          │   [spectator     └────────┴────────┘ │
│ │ Reward: 0.87        │    kamera képe                       │
│ │ Maneuver: Left      │    + kirajzolt útvonal-pontok]       │
│ │ ...                 │                                      │
│ └─────────────────────┘                                      │
└──────────────────────────────────────────────────────────────┘
```

A `_draw_path` (392–417. sor) **3D → 2D vetítéssel** rajzolja rá az útvonalat a
spectator kamera képére:

```python
world_2_camera = np.array(camera.get_transform().get_inverse_matrix())
K = build_projection_matrix(image_w, image_h, fov)      # kamera belső mátrix
x, y = get_image_point(waypoint_location, K, world_2_camera)
image = cv2.circle(image, (int(x), int(y)), radius=3, color=color, thickness=-1)
```

Csak a 2–50 m távolságban lévő pontokat rajzolja (407. sor) — a túl közeliek zavarnának,
a túl távoliak pontatlanok.

---

## Szenzor-callback-ek (440–459. sor)

```python
def _on_collision(self, event):
    if get_actor_display_name(event.other_actor) != "Road":
        self.terminal_state = True
    if self.activate_render:
        self.hud.notification("Collision with {}".format(get_actor_display_name(event.other_actor)))
```

**Ütközés → azonnali epizódvég**, kivéve ha az "ütköző fél" az út maga (az egyenetlen
terepnél előfordulhat álütközés).

```python
def _on_invasion(self, event):
    lane_types = set(x.type for x in event.crossed_lane_markings)
    ...
    self.hud.notification("Crossed line %s" % " and ".join(text))
```

**A sávelhagyásnak nincs következménye** — csak kiírja. A sávban tartást a jutalomfüggvény
`distance_from_center` része tanítja, nem ez a szenzor. Ez egy lehetséges bővítési pont.

```python
def _set_observation_image(self, image):
    self.observation_buffer = image
```

A CARLA egy külön szálon hívja ezeket, amikor új szenzoradat érkezik. Egyszerű
"legutolsó érték" puffer — ha két kép érkezne két `step()` között, a régebbi elveszne.
Szinkron módban ez nem fordul elő.

---

## `close()` — ⚠ hibás (180–186. sor)

```python
def close(self):
    if self.carla_process:      # ⚠ ilyen attribútum SOSEM lesz beállítva!
        self.carla_process.terminate()
    pygame.quit()
    ...
```

A `self.carla_process` sehol nincs definiálva → `AttributeError`, ha meghívod a `close()`-t.
(A jelenlegi `train.py` nem hívja, ezért nem derül ki.) Javítás:
[07-tovabbfejlesztes.md](07-tovabbfejlesztes.md).

---

## Összefoglaló: egy `step()` időrendben

```
1. action megérkezik ──────► smooth_action ──► vehicle.control (steer, throttle)
2. world.tick() ───────────► Vehicle.tick(): apply_control
                             CARLA szerver: 0.05 s fizika + render
                             ↓ (aszinkron callbackek)
                             _set_observation_image() puffer feltöltés
3. _get_observation() ─────► kép kiolvasása a pufferből (busy wait)
4. waypoint keresés ───────► dot product teszt, current_waypoint_index frissítés
5. distance_to_line ───────► distance_from_center
6. reward_fn(self) ────────► jutalom + esetleg terminal_state = True
7. encode_state_fn(self) ──► VAE encode + mérőszámok → dict
8. render() ───────────────► pygame kirajzolás
9. return (obs, reward, done, False, info)
```

---

➡️ Következő: [03-wrappers-rewards-state.md](03-wrappers-rewards-state.md)
