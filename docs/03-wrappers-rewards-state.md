# 03 — `wrappers.py`, `rewards.py`, `state_commons.py`

Ez a három fájl a környezet "kiszolgálója": a CARLA burkolók, a jutalom és az állapot-kódolás.

---

# 1. `carla_env/wrappers.py` — CARLA burkolók (429 sor)

## Miért kell burkoló?

A CARLA Python API elég alacsony szintű: manuálisan kell blueprint-et keresni, actort
spawnolni, listener-t regisztrálni, és a végén mindent kézzel eltakarítani. Ez a fájl ezt
csomagolja be használható osztályokba.

## Segédfüggvények (6–120. sor)

### `get_actor_display_name(actor)` (6. sor)
```python
name = " ".join(actor.type_id.replace("_", ".").title().split(".")[1:])
```
A `"vehicle.tesla.model3"` type_id-ből `"Tesla Model3"`-at csinál. Az ütközés-üzenetekben
használjuk.

### `get_displacement_vector(car_pos, waypoint_pos, theta)` (11–41. sor) ★

Ez fontos: **világkoordinátákat alakít át az autó saját koordinátarendszerébe.**

```python
relative_pos = waypoint_pos - car_pos              # eltolás

R = np.array([[ np.cos(theta), np.sin(theta), 0],  # forgatás -theta-val
              [-np.sin(theta), np.cos(theta), 0],
              [ 0,             0,             1]])
T = np.array([[0, 1, 0],                           # x és y felcserélése
              [1, 0, 0],
              [0, 0, 1]])

waypoint_car = R @ relative_pos
waypoint_car = T @ waypoint_car
```

**Miért kell ez?** Ha a waypointokat világkoordinátákban adnád az ágensnek, azt kellene
megtanulnia, hogy "ha a térkép (120, 45) pontján vagyok és a waypoint (122, 47), akkor
balra kell fordulni" — ami a térkép minden pontján más. Az autó-relatív koordinátákkal
viszont "a waypoint 2 méterre előttem és 1 méterre balra van" — **ez térképfüggetlen és
általánosítható**.

A `T` mátrix (x↔y csere) azért kell, hogy a végeredményben az **y tengely mutasson előre**
(ahogy a docstring is írja).

```python
waypoint_car[np.abs(waypoint_car) < 10e-10] = 0
```
Lebegőpontos zaj kiszűrése — kozmetikai.

### `angle_diff(v0, v1)` (44–68. sor)

Két 2D vektor **előjeles** szögkülönbsége radiánban:

```python
angle = np.arccos(dot_product)              # 0..π, előjel nélkül
cross_product = np.cross(v0_xy_u, v1_xy_u)
if cross_product < 0:
    angle = -angle                          # ← az előjelet a keresztszorzat adja
if abs(angle) >= 2.3:                       # ⚠ különös szabály
    return 0
return round(angle, 2)
```

- Az `arccos` mindig 0..π-t ad, tehát nem tudni, melyik irányba. A **keresztszorzat z
  komponense** adja meg az irányt (jobbra/balra).
- **A 2.3 rad (≈132°) szabály:** ha a szögeltérés túl nagy, 0-t ad vissza. Ez azt kezeli,
  amikor az autó **áll** (a sebességvektor iránya értelmetlen) vagy tolat/megfordul —
  ilyenkor a szöghiba félrevezető lenne. Kicsit hackes, de működik.

### `distance_to_line(A, B, p)` (71–77. sor)
```python
p[2] = 0                                       # ⚠ MELLÉKHATÁS: módosítja a bemenetet!
num = np.linalg.norm(np.cross(B - A, A - p))
denom = np.linalg.norm(B - A)
```
Pont–egyenes távolság. A `p[2] = 0` **helyben módosítja** a beadott numpy tömböt — itt
nem okoz gondot (`vector()` friss tömböt hoz létre), de veszélyes minta.

### `vector(v)` (80–85. sor)
CARLA `Location`/`Vector3D`/`Rotation` → numpy tömb.

> ⚠ **Névütközés!** Van egy másik `vector()` is a `carla_env/tools/misc.py`-ban, ami
> **két pont közti egységvektort** ad vissza. A `navigation/` modul azt használja,
> a környezet ezt. Ha valaha összekevered a kettőt, nehezen megtalálható hibát kapsz.

### `smooth_action(old, new, factor)` (88. sor)
```python
return old_value * smooth_factor + new_value * (1.0 - smooth_factor)
```
Exponenciális simítás. Lásd [01](01-train-config.md).

### `build_projection_matrix` és `get_image_point` (92–120. sor)

3D→2D vetítés a HUD útvonalrajzoláshoz.

```python
focal = w / (2.0 * np.tan(fov * np.pi / 360.0))   # gyújtótávolság pixelben
K = [[focal, 0,     w/2],
     [0,     focal, h/2],
     [0,     0,     1  ]]
```

A `get_image_point`-ban a lényeges rész:
```python
point_camera = np.dot(w2c, point)                              # világ → kamera
point_camera = [point_camera[1], -point_camera[2], point_camera[0]]  # UE4 → standard
point_img = np.dot(K, point_camera)                            # kamera → kép
point_img[0] /= point_img[2]                                   # perspektív osztás
```
Az UE4 koordinátarendszere (x=előre, y=jobbra, z=fel) más, mint a szokásos
számítógépes látás konvenció (x=jobbra, y=le, z=előre) — innen a tengelycsere.

### `sensor_transforms` (123–128. sor)
```python
"spectator": Location(x=-5.5, z=2.8), Rotation(pitch=-15)  # 5.5 m-rel hátrébb, 2.8 m magasan, 15°-kal lefelé
"dashboard": Location(x=1.6, z=1.7)                        # motorháztetőn
"lidar":     Location(x=0.0, z=2.4)                        # tetőn
"birdview":  Location(x=90, y=210, z=175), Rotation(pitch=-90)  # madártávlat (nem használt)
```

## `CarlaActorBase` (135–159. sor)

```python
class CarlaActorBase(object):
    def __init__(self, world, actor):
        self.world = world
        self.actor = actor
        self.world.actor_list.append(self)     # ← automatikus regisztráció
        self.destroyed = False

    def destroy(self):
        ...
        self.actor.destroy()
        self.world.actor_list.remove(self)

    def tick(self):
        pass

    def __getattr__(self, name):
        """Relay missing methods to underlying carla actor"""
        return getattr(self.actor, name)
```

**A `__getattr__` a legszebb trükk itt.** Ha egy attribútumot nem talál a burkolón, továbbadja
a becsomagolt CARLA actornak. Ezért működik pl. `self.vehicle.get_location()` anélkül, hogy
a `Vehicle` osztályban ez definiálva lenne.

Az `actor_list` automatikus vezetése miatt a `World.destroy()` mindent el tud takarítani.

## `Lidar` (166–220. sor)

Beállítja a LIDAR blueprintjét (64 csatorna, 20 m hatótáv, 110° vízszintes látószög), majd
a nyers pontfelhőt **felülnézeti képpé** alakítja:

```python
points = np.frombuffer(raw.raw_data, dtype=np.dtype('f4'))
points = np.reshape(points, (int(points.shape[0] / 4), 4))   # x, y, z, intenzitás
lidar_data = np.array(points[:, :2])                          # csak x, y
lidar_data *= min(self._width, self._height) / lidar_range    # skálázás pixelre
lidar_data += (0.5 * self._width, 0.5 * self._height)         # középre tolás
lidar_img[tuple(lidar_data.T)] = (255, 255, 255)              # fehér pontok
```

**Jelenleg nem használt** (`activate_lidar=False`).

## `CollisionSensor` és `LaneInvasionSensor` (227–281. sor)

Mindkettő ugyanazt a mintát követi:
```python
weak_self = weakref.ref(self)
actor = world.spawn_actor(bp, carla.Transform(), attach_to=vehicle.get_carla_actor())
actor.listen(lambda event: CollisionSensor.on_collision(weak_self, event))
```

**Miért `weakref`?** Ha a lambda erős referenciát tartana a `self`-re, körkörös hivatkozás
jönne létre (szenzor → lambda → self → szenzor), és a szemétgyűjtő nem tudná felszabadítani.
A gyenge referencia ezt megtöri. Ez a CARLA példakódok standard mintája.

## `Camera` (288–353. sor)

```python
camera_bp.set_attribute("image_size_x", str(width))
camera_bp.set_attribute("image_size_y", str(height))
camera_bp.set_attribute("fov", f"110")        # 110° széles látószög
```

A kép feldolgozása (312–350. sor):
```python
image.convert(self.color_converter)
array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
array = np.reshape(array, (image.height, image.width, 4))    # BGRA
array = array[:, :, :3]                                       # alfa eldobása → BGR
array = array[:, :, ::-1]                                     # BGR → RGB
```

### A `custom_palette` blokk (324–348. sor)

Szemantikus szegmentációs kamerához: a CARLA az osztálycímkét az R csatornába teszi, ezt
kell színekké fordítani. Ez a projekt **leegyszerűsített palettát** használ:

```python
classes = {
    6:  [157, 234, 50],   # RoadLines — sárgászöld
    7:  [50, 64, 128],    # Roads     — kék
    8:  [255, 255, 255],  # Sidewalks — fehér
    # MINDEN MÁS: [0, 0, 0] fekete
}
```

Tehát az ágens csak azt látná: **útburkolati jel, út, járda, minden más**. Ez ideális VAE
bemenet lenne, mert eltűnik a felesleges vizuális zaj (fák, épületek, égbolt).

> ⚠ **Teljesítmény:** a 342–348. sor **kettős Python `for` ciklus** minden pixelen. 160×80 =
> 12 800 iteráció **minden képkockán**. Ez nagyon lassú. Vektorizálva:
> ```python
> palette = np.array([classes[i] for i in range(13)] , dtype=np.uint8)
> array = palette[np.clip(segimg, 0, 12)]
> ```
> Jelenleg ez nem fut (nincs `seg_camera` a STATE-ben), de ha bekapcsolod, ez lesz a szűk
> keresztmetszet.

## `Vehicle` (360–399. sor)

```python
vehicle_bp = world.get_blueprint_library().find("vehicle.tesla.model3")
color = vehicle_bp.get_attribute("color").recommended_values[0]
...
self.control = carla.VehicleControl()          # ← állandó control objektum
```

```python
def tick(self):
    self.actor.apply_control(self.control)     # ← a World.tick() hívja
```

**Fontos:** a `control` objektum **perzisztens** — a `step()` csak módosítja a `steer` és
`throttle` mezőit, és a `tick()` alkalmazza. Ezért működik az akciósimítás (az előző érték
megmarad).

```python
def get_speed(self):
    velocity = self.get_velocity()
    return 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)   # m/s → km/h
```

```python
def get_angle(self, waypoint):
    fwd = vector(self.get_velocity())                                  # a MOZGÁS iránya
    wp_fwd = vector(waypoint.transform.rotation.get_forward_vector())  # a SÁV iránya
    return angle_diff(wp_fwd, fwd)
```

> **Figyeld meg:** a sebességvektort használja, **nem az autó orientációját**. Ha az autó
> áll, a sebességvektor nulla → az `angle_diff` 0-t ad vissza (a norma-ellenőrzés miatt).
> Ez a `angle_next_waypoint` állapotelem és a `reward_fn5` szöghiba-tényezője.

## `World` (406–428. sor)

```python
class World():
    def __init__(self, client):
        self.world = client.load_world('Town02')   # ← ITT VAN A TÉRKÉP RÖGZÍTVE
        self.map = self.get_map()
        self.actor_list = []
```

> **Ha másik térképen akarsz tanítani, ezt a sort kell módosítani.** Érdemes lenne
> `config.py`-ba kiemelni.

```python
def tick(self):
    for actor in list(self.actor_list):
        actor.tick()          # minden actor alkalmazza a vezérlését
    self.world.tick()         # a szerver lép egyet
```

A `list(...)` másolat azért kell, hogy iteráció közben biztonságos legyen, ha valami
módosítaná a listát.

---

# 2. `carla_env/rewards.py` — a jutalomfüggvény (71 sor)

Ez a fájl dönti el, **mit jelent jól vezetni**. Az RL-ben ez a legfontosabb tervezési döntés:
az ágens pontosan azt fogja megtanulni, amit itt megjutalmazol — se többet, se kevesebbet.

## Modul szint (4–14. sor)

```python
low_speed_timer = 0                             # ⚠ GLOBÁLIS változó

min_speed = REWARD_PARAMS["min_speed"]          # 20 km/h
max_speed = REWARD_PARAMS["max_speed"]          # 35 km/h
target_speed = REWARD_PARAMS["target_speed"]    # 25 km/h
max_distance = REWARD_PARAMS["max_distance"]    # 2.0 m
...
reward_functions = {}
```

A paraméterek **modul betöltéskor** olvasódnak ki a configból — futás közben már nem
változtathatók.

## `create_reward_fn(reward_fn)` (17–50. sor) — a "burkoló"

Ez egy **magasabb rendű függvény**: kap egy alap jutalomfüggvényt, és visszaad egy
kibővítettet, ami az epizódvég-logikát is kezeli.

```python
def create_reward_fn(reward_fn):
    def func(env):
        terminal_reason = "Running..."
        if early_stop:
            global low_speed_timer
            low_speed_timer += 1.0 / env.fps        # 0.05 s hozzáadása
            speed = env.vehicle.get_speed()
```

### A három leállási feltétel (24–34. sor)

```python
            if low_speed_timer > 5.0 and speed < 1.0 and env.current_waypoint_index >= 1:
                env.terminal_state = True
                terminal_reason = "Vehicle stopped"
```
**"Megállt"** — több mint 5 s telt el és 1 km/h alatt van. A `current_waypoint_index >= 1`
feltétel megvédi az indulást (a spawn utáni álló állapotot nem bünteti azonnal).

```python
            if env.distance_from_center > max_distance:
                env.terminal_state = True
                terminal_reason = "Off-track"
```
**"Letért a pályáról"** — több mint 2 m-re a sávközéptől.

```python
            if max_speed > 0 and speed > max_speed:
                env.terminal_state = True
                terminal_reason = "Too fast"
```
**"Túl gyors"** — 35 km/h felett. Ez az egyetlen sebességkorlát: fék hiányában így
akadályozza meg a gyorsulást.

> ⚠ **A `low_speed_timer` globális változó hibája:** csak akkor nullázódik, amikor
> terminális állapot következik be (40. sor). Ha az epizód **sikerrel** ér véget
> (`success_state`), a számláló **nem nullázódik**. Így a következő epizód elején már
> "beégett" nagy értékkel indul, és az autó azonnal `Vehicle stopped`-ot kaphat, ha az első
> pár lépésben lassan indul. Javítás: nullázni a `reset()`-ben, vagy jobb: `env` mezővé
> tenni globális helyett.

### A jutalom összeállítása (36–48. sor)

```python
        reward = 0
        if not env.terminal_state:
            reward += reward_fn(env)          # a "rendes" jutalom
        else:
            low_speed_timer = 0.0
            reward += penalty_reward          # -10
            print(f"{env.episode_idx}| Terminal: ", terminal_reason)

        if env.success_state:
            print(f"{env.episode_idx}| Success")

        env.extra_info.extend([terminal_reason, ""])
        return reward
```

Tehát: **normál lépés → 0..1 jutalom; hibás vég → -10 büntetés.**

## `reward_fn5(env)` (53–68. sor) — ★ a tényleges jutalomképlet

```python
def reward_fn5(env):
    angle = env.vehicle.get_angle(env.current_waypoint)
    speed_kmh = env.vehicle.get_speed()
```

### 1. Sebesség tényező (56–61. sor)

```python
    if speed_kmh < min_speed:                      # < 20 km/h
        speed_reward = speed_kmh / min_speed       # lineárisan nő 0 → 1
    elif speed_kmh > target_speed:                 # > 25 km/h
        speed_reward = 1.0 - (speed_kmh - target_speed) / (max_speed - target_speed)
    else:                                          # 20–25 km/h
        speed_reward = 1.0
```

```
speed_reward
   1.0 ┤        ┌─────────┐
       │       ╱           ╲
       │      ╱             ╲
       │     ╱               ╲
   0.0 ┼────┴─────┬────┬──────┴────► km/h
       0         20   25     35
                min  target max
```

**Trapéz alakú:** 20–25 km/h között teljes jutalom, alatta lineárisan kevesebb, felette
lineárisan csökken 0-ig 35 km/h-nál (ahol egyébként is leáll az epizód).

### 2. Sávközép tényező (63. sor)

```python
    centering_factor = max(1.0 - env.distance_from_center / max_distance, 0.0)
```
1.0 a sáv közepén, 0.0 két méternél (ahol amúgy is `Off-track` lesz).

### 3. Szög tényező (64. sor)

```python
    angle_factor = max(1.0 - abs(angle / np.deg2rad(max_angle_center_lane)), 0.0)
```
1.0, ha pontosan a sáv irányába megy; 0.0, ha 90°-ban keresztbe.

### 4. Simaság tényező (65–66. sor)

```python
    std = np.std(env.distance_from_center_history)     # az utolsó 30 lépés szórása
    distance_std_factor = max(1.0 - abs(std / max_std_center_lane), 0.0)
```

**Ez a legötletesebb rész.** A többi tényező pillanatnyi állapotot mér, ez viszont az
**időbeli stabilitást**. Ha az autó cikkcakkozik a sávon belül (átlagosan középen, de
folyton ingadozva), a szórás nagy lesz → büntetés. Ez ösztönzi a sima, egyenletes vezetést.

### A szorzat (68. sor)

```python
    return speed_reward * centering_factor * angle_factor * distance_std_factor
```

**Miért szorzat, nem összeg?** Mert a szorzat **AND kapcsolatot** jelent: minden feltételnek
egyszerre kell teljesülnie. Ha bármelyik tényező 0, a teljes jutalom 0.

Ha összeg lenne, az ágens "kijátszhatná": pl. tökéletesen középen állva, 0 sebességgel is
kapna a centering+angle+std tényezőkből jutalmat. A szorzattal ez lehetetlen.

**A jutalom tartománya:** 0.0 – 1.0 lépésenként, plusz -10 hibás végért.

## Regisztráció (71. sor)

```python
reward_functions["reward_fn5"] = create_reward_fn(reward_fn5)
```

A `config.py`-ban a `"reward_fn": "reward_fn5"` erre a kulcsra hivatkozik.
**Új jutalomfüggvény hozzáadása:** írj egy `reward_fn6(env)` függvényt, regisztráld
ugyanígy, és állítsd át a configot.

---

# 3. `carla_env/state_commons.py` — az állapot előállítása (112 sor)

## `load_vae(vae_dir, latent_size)` (15–26. sor)

```python
model_dir = os.path.join(vae_dir, 'best.tar')
model = VAE(latent_size)
if os.path.exists(model_dir):
    state = torch.load(model_dir)
    print("Reloading model at epoch {}, with test error {}".format(state['epoch'], state['precision']))
    model.load_state_dict(state['state_dict'])
    return model
raise Exception("Error - VAE model does not exist")
```

A checkpoint egy szótár: `epoch`, `precision`, `state_dict`.

> **Fontos:** a modell **CPU-n** marad (nincs `.to('cuda')`) és **nincs `.eval()` mód** hívva.
> Lásd a következő megjegyzést a `reparameterize`-nál.

## `preprocess_frame(frame)` (29–34. sor)

```python
preprocess = transforms.Compose([transforms.ToTensor()])
frame = preprocess(frame).unsqueeze(0)
```

A `ToTensor()` két dolgot csinál: (H, W, C) → (C, H, W) átrendezés, és 0–255 → 0.0–1.0
normalizálás. Az `unsqueeze(0)` batch dimenziót ad: (1, 3, 80, 160).

## `create_encode_state_fn(vae, measurements_to_include)` (37–112. sor) ★

Ez egy **gyár (factory) függvény**: a `STATE` lista alapján legyártja a három szükséges dolgot.

### `measure_flags` (39–49. sor)

```python
measure_flags = ["steer" in measurements_to_include,          # [0]
                 "throttle" in measurements_to_include,       # [1]
                 "speed" in measurements_to_include,          # [2]
                 "angle_next_waypoint" in measurements_to_include,  # [3]
                 "maneuver" in measurements_to_include,       # [4]
                 "waypoints" in measurements_to_include,      # [5]
                 False, False, False, False, False]           # [6..10] — fenntartva
```

A jelenlegi `STATE`-tel: `[True, True, True, True, True, True, False×5]`.

> A `False`-ok a végén valószínűleg korábbi/tervezett bővítések helyei (pl. `seg_camera`,
> `lidar`). Egy szótár olvashatóbb lenne indexelt lista helyett.

### `create_observation_space()` (51–65. sor)

```python
observation_space = {}
if vae:
    observation_space['vae_latent'] = gym.spaces.Box(low=-4, high=4, shape=(64,), dtype=np.float32)

low, high = [], []
if measure_flags[0]: low.append(-1),   high.append(1)      # steer
if measure_flags[1]: low.append(0),    high.append(1)      # throttle
if measure_flags[2]: low.append(0),    high.append(120)    # speed km/h
if measure_flags[3]: low.append(-3.14), high.append(3.14)  # angle rad
observation_space['vehicle_measures'] = gym.spaces.Box(low=np.array(low), high=np.array(high), ...)

if measure_flags[4]: observation_space['maneuver'] = gym.spaces.Discrete(4)
if measure_flags[5]: observation_space['waypoints'] = gym.spaces.Box(low=-50, high=50, shape=(15, 2), ...)

return gym.spaces.Dict(observation_space)
```

Az eredmény:
```python
Dict(
    vae_latent:       Box(-4, 4, (64,)),
    vehicle_measures: Box([-1, 0, 0, -3.14], [1, 1, 120, 3.14], (4,)),
    maneuver:         Discrete(4),
    waypoints:        Box(-50, 50, (15, 2)),
)
```

Ezért kell `'MultiInputPolicy'` a `train.py`-ban — az SB3 minden kulcshoz külön
feldolgozót épít, majd összefűzi őket.

> ⚠ **A `Discrete(4)` és a `RoadOption` értékei:** a `RoadOption.VOID = -1`. Ha valaha
> `VOID` manőver kerülne az útvonalba, a `Discrete(4)` (ami 0..3-at fogad el) érvénytelen
> értéket kapna. A `compute_route_waypoints` gyakorlatilag sosem ad VOID-ot vissza,
> de elméletileg rés.

### `encode_state(env)` (67–102. sor) — minden lépésben fut

```python
if vae:
    with torch.no_grad():                                  # nincs gradiens: gyorsabb
        frame = preprocess_frame(env.observation)
        mu, logvar = vae.encode(frame)
        vae_latent = vae.reparameterize(mu, logvar)[0].cpu().detach().numpy().squeeze()
    encoded_state['vae_latent'] = vae_latent
```

> ⚠ **Fontos részlet:** a `reparameterize` **véletlen zajt ad hozzá** (`eps * sigma + mu`).
> Tehát **ugyanaz a kép kétszer más latent vektort ad**. Ez tanításkor egyfajta
> regularizáció/augmentáció, de kiértékeléskor nem determinisztikus.
> Ha determinisztikus kódolást akarsz, használd csak a `mu`-t:
> ```python
> vae_latent = mu[0].cpu().numpy()
> ```

```python
vehicle_measures = []
if measure_flags[0]: vehicle_measures.append(env.vehicle.control.steer)
if measure_flags[1]: vehicle_measures.append(env.vehicle.control.throttle)
if measure_flags[2]: vehicle_measures.append(env.vehicle.get_speed())
if measure_flags[3]: vehicle_measures.append(env.vehicle.get_angle(env.current_waypoint))
encoded_state['vehicle_measures'] = vehicle_measures
```

**Miért kell a `steer` és `throttle` az állapotba?** Mert az akciósimítás miatt a tényleges
vezérlés nem egyenlő az utolsó akcióval. Az ágensnek tudnia kell, hol áll *most* a kormány,
hogy értelmesen tudjon rá építeni. (Ez teszi az állapotot Markov-tulajdonságúvá.)

```python
if measure_flags[4]: encoded_state['maneuver'] = env.current_road_maneuver.value
```

### A waypointok relatív koordinátákba (83–99. sor) ★

```python
next_waypoints_state = env.route_waypoints[env.current_waypoint_index : env.current_waypoint_index + 15]
waypoints = [vector(way[0].transform.location) for way in next_waypoints_state]

vehicle_location = vector(env.vehicle.get_location())
theta = np.deg2rad(env.vehicle.get_transform().rotation.yaw)

relative_waypoints = np.zeros((15, 2))
for i, w_location in enumerate(waypoints):
    relative_waypoints[i] = get_displacement_vector(vehicle_location, w_location, theta)[:2]
```

A következő 15 útpont (1 méterenként, tehát **15 m előrelátás**) az autó
koordinátarendszerében. Ez adja meg az ágensnek az út "alakját" — hogy jön-e kanyar.

```python
if len(waypoints) < 15:
    start_index = len(waypoints)
    reference_vector = relative_waypoints[start_index-1] - relative_waypoints[start_index-2]
    for i in range(start_index, 15):
        relative_waypoints[i] = relative_waypoints[i-1] + reference_vector
```

**Extrapolálás az útvonal végén:** ha kevesebb mint 15 pont maradt, az utolsó két pont
irányát folytatja lineárisan. Így az állapot alakja mindig (15, 2) marad.

> ⚠ **Peremeset:** ha `len(waypoints) == 1`, akkor `start_index - 2 == -1`, ami a numpy-ban
> az *utolsó* elemre mutat → `relative_waypoints[0] - relative_waypoints[14]`, ami
> értelmetlen vektort ad. Nagyon ritka (az útvonal utolsó pontján), de valós rés.
> Ha `len(waypoints) == 0` lenne, `IndexError`. A `step()` logikája (útvonal-váltás a
> végén) ezt gyakorlatilag megakadályozza.

### `decode_vae_state(z)` (104–109. sor)

```python
with torch.no_grad():
    sample = torch.tensor(z)
    sample = vae.decode(sample).cpu()
    generated_image = sample.view(3, 80, 160).numpy().transpose((1, 2, 0)) * 255
return generated_image
```

Visszafejti a latent vektort képpé — **csak a HUD-on való megjelenítéshez**. Nagyon hasznos
hibakereséshez: ha a visszafejtett kép elmosódott vagy értelmetlen, akkor a VAE nem
alkalmas erre a jelenetre, és az ágens sem fog tudni tanulni belőle.

---

➡️ Következő: [04-vae.md](04-vae.md)
