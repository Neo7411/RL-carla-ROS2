# `CarlaRouteEnv` — részletes leírás

A [carla_env/envs/carla_route_env.py](../carla_env/envs/carla_route_env.py) fájlban lévő
`CarlaRouteEnv` osztály dokumentációja: mi mire jó, és mi hogyan működik benne.

---

## 1. Mi ez az osztály?

Ez a **Gymnasium környezet** (`gym.Env` leszármazott), ami a CARLA szimulátort
összeköti a megerősítéses tanulással (SAC). Az RL algoritmus szemszögéből ez a
"játék": ad neki megfigyelést (kamerakép), kap tőle akciót (kormány + gáz), és
visszaad jutalmat.

A feladat: **az autó menjen végig egy útvonalon a sáv közepén tartva.**

```
   SAC (train.py)
        │  akció: [steer, throttle]
        ▼
  ┌─────────────────────┐
  │   CarlaRouteEnv     │ ◄── reward_fn (carla_env/reward.py)
  │   .step(action)     │ ◄── encode_state_fn (VAE)
  └─────────────────────┘
        │  megfigyelés, jutalom, done
        ▼
   CARLA szimulátor (Town02)
```

---

## 2. Konstruktor — `__init__`

### Fontosabb paraméterek

| Paraméter | Mire jó |
|---|---|
| `host`, `port` | A CARLA szerver címe (alapból `127.0.0.1:2000`). |
| `viewer_res` | A pygame ablak mérete (1920×1080) — csak nézegetésre. |
| `obs_res` | A **hálónak adott** kép mérete (160×80). Ez a lényeg. |
| `reward_fn` | A jutalomfüggvény. Ha nincs megadva, `lambda x: 0` lesz belőle. |
| `observation_space` | Gym megfigyelés-tér, kívülről kapja (`state_commons.py`). |
| `encode_state_fn` | Nyers kép → VAE látens vektor. |
| `decode_vae_fn` | Látens → visszafejtett kép (csak megjelenítéshez). |
| `fps` | Szimulációs lépések másodpercenként (15). |
| `action_smoothing` | Akció-simítás 0..1. 0 = nincs simítás. |
| `activate_spectator` | Külső "üldöző" kamera bekapcsolása. |
| `activate_render` | Pygame ablak kirajzolása. Tanításnál kikapcsolható. |
| `eval` | Kiértékelő mód — fix útvonalak, más epizód-vége logika. |

### Mi történik létrehozáskor (sorrendben)

1. **Akciótér beállítása** ([:50-51](../carla_env/envs/carla_route_env.py#L50-L51)):
   ```python
   gym.spaces.Box(np.array([-1, 0]), np.array([1, 1]))
   ```
   Két folytonos szám: **kormány** (−1 balra … +1 jobbra) és **gáz** (0 … 1).
   Fék nincs — az autó csak gyorsítani tud, illetve gurulni.

2. **Kapcsolódás a CARLA szerverhez**, 60 mp timeouttal.

3. **World létrehozása** — a `World` wrapper betölti a **Town02** térképet.

4. **Szinkron mód bekapcsolása** ([:75-79](../carla_env/envs/carla_route_env.py#L75-L79)):
   ```python
   settings.fixed_delta_seconds = 1 / self.fps   # 1/15 mp
   settings.synchronous_mode = True
   ```
   Ez **kritikus** RL-hez: a szimulátor csak akkor lép egyet, ha mi megmondjuk
   (`world.tick()`). Enélkül a szimuláció a saját tempójában futna, és a
   megfigyelés–akció párok összecsúsznának.

5. **Jármű létrehozása** — Tesla Model 3, az első spawn ponton.
   Kap egy **ütközés-szenzort** és egy **sávelhagyás-szenzort**, callbackekkel.

6. **Pygame + HUD** inicializálása, ha `activate_render`.

7. **Kamerák felrakása:**
   - `dashcam` — a motorháztetőn (`x=1.6, z=1.7`), 110° látószög. **Ez adja a
     megfigyelést a hálónak.** Ha az observation_space tartalmaz `seg_camera`-t,
     akkor szemantikus szegmentációs kamera lesz belőle egyedi palettával
     (út = kék, sávfelfestés = zöld, járda = fehér, minden más = fekete).
   - `camera` (spectator) — hátulról-felülről (`x=-5.5, z=2.8`), csak nézni.
   - `lidar` — opcionális, alapból ki van kapcsolva.

8. Végül **`self.reset()`** — az első epizód elindítása.

---

## 3. Epizód-kezelés

### `reset()` — új epizód

Ez fut minden epizód elején. Amit csinál:

- `episode_idx += 1` — epizódszámláló.
- `new_route()` — új útvonal generálása (lásd lentebb).
- **Állapotjelzők nullázása:**
  - `terminal_state = False` — "elrontottuk" (ütközés, lesodródás)
  - `success_state = False` — "sikeresen teljesítettük"

  Ez a két külön változó a lényeg: mindkettő epizódot zár, de a
  jutalom szempontjából **nem ugyanaz** — a `terminal_state` büntetést kap,
  a `success_state` nem.
- **Metrikák nullázása:** `total_reward`, `distance_traveled`,
  `center_lane_deviation`, `speed_accum`, `step_count`.
- Egy `world.tick()`, majd egy **`step(None)`** hívás, hogy megszülessen az
  első megfigyelés. A `None` akció azt jelenti: "ne csinálj semmit, csak
  olvasd ki az állapotot".

> A `time.sleep(0.2)` hívások körülötte azért kellenek, hogy a CARLA
> szenzorai biztosan leszállítsák az első képkockát.

### `new_route()` — útvonal generálása

Ez **nem csak epizód elején** fut, hanem menet közben is, amikor az autó
végigért az útvonalon (lásd `step()`).

1. **Puha reset:** kormány és gáz nullázása, fizika **kikapcsolása**
   (`set_simulate_physics(False)`) — hogy a teleportálás ne dobja meg az autót.

2. **Kezdő- és végpont választása:**
   - **Tanítás közben, páros epizódban:** fix kereszteződéses útvonalak a
     `intersection_routes` listából (körkörösen, `itertools.cycle`).
     Ez azért van, hogy a modell **direkt gyakorolja a kanyarodást** —
     véletlen útvonalakkal ritkán jönne szembe kereszteződés.
   - **Egyébként:** két véletlen spawn pont.
   - **Eval módban:** fix `eval_routes` lista — hogy az összehasonlítás
     mérhető legyen.

3. **Útvonal kiszámítása** a `compute_route_waypoints()` hívással,
   1 méteres felbontással. Az eredmény `(waypoint, RoadOption)` párok listája,
   ahol a `RoadOption` a manőver: LANEFOLLOW / LEFT / RIGHT / STRAIGHT.

4. **Ciklus, amíg értelmes útvonal nem lesz:** ha a hossz ≤ 1 (a két pont
   ugyanaz vagy nincs útvonal köztük), új véletlen pontokat sorsol.

5. `distance_from_center_history = deque(maxlen=30)` — az utolsó 30 lépés
   sáv-középtől való eltérése. **Ebből számol szórást a jutalomfüggvény**
   a "cikcakkozás" büntetéséhez.

6. Autó teleportálása a startpontra, majd fizika **vissza be**.

---

## 4. `step(action)` — a lényeg

Ez fut minden szimulációs lépésben. Végigmegyek rajta sorrendben.

### 4.1 Akció alkalmazása

```python
if self.current_waypoint_index >= len(self.route_waypoints) - 1:
    if not self.eval:
        self.new_route()      # tanításnál: új útvonal, megy tovább
    else:
        self.success_state = True   # evalnál: kész, vége
```

Fontos különbség: **tanítás közben az epizód nem ér véget a cél elérésekor**,
csak kap egy új útvonalat. Így egy epizód sokáig tud tartani, és a modell
folyamatosan tanul. Evalnál viszont ez a siker feltétele.

Utána az akció szétszedése és **simítása:**

```python
steer = smooth_action(régi_steer, új_steer, action_smoothing)
```

ami `régi * f + új * (1-f)`. Ez megakadályozza, hogy a kormány
lépésről lépésre ugráljon — simább, valósághűbb vezetés.

### 4.2 Szimuláció léptetése

```python
self.world.tick()
```

Ez alkalmazza a vezérlést az autóra és lépteti a szimulátort egy
`1/fps` időlépéssel.

### 4.3 Megfigyelés kiolvasása

```python
self.observation = self._get_observation()
```

A `_get_observation()` egy **busy-wait ciklus:**

```python
while self.observation_buffer is None:
    pass
```

A kamera aszinkron callbackben tölti fel a buffert
(`_set_observation_image`), ez a ciklus pedig megvárja. Egyszerű, de
CPU-t pörget — szinkron módban viszont garantáltan megjön a kép.

### 4.4 Waypoint-követés

Ez határozza meg, **hol tartunk az útvonalon:**

```python
dot = np.dot(waypoint_iránya, (autó_pozíció - waypoint_pozíció))
if dot > 0.0:      # elhagytuk a waypointot?
    waypoint_index += 1
```

A **skaláris szorzat előjele** mondja meg, hogy az autó a waypoint előtt
vagy mögött van. Ha mögötte (`dot > 0`), akkor túlhaladtunk rajta,
léphetünk a következőre. A ciklus egyszerre több waypointot is át tud
ugrani, ha az autó gyors volt.

### 4.5 Sáv-középtől való eltérés

```python
self.distance_from_center = distance_to_line(
    current_waypoint, next_waypoint, autó_pozíció)
```

Az aktuális és a következő waypoint **egyenest feszít ki** — ez a sáv
közepe. Az autó ettől való merőleges távolsága a `distance_from_center`.
Ezt használja a jutalomfüggvény és az epizód-vége feltétel is.

### 4.6 Metrikák

- `center_lane_deviation +=` — összegzett eltérés (átlaghoz)
- `distance_traveled +=` — megtett út (az előző pozícióhoz képest)
- `speed_accum +=` — összegzett sebesség (átlaghoz)
- `distance_from_center_history.append(...)` — a 30 hosszú deque

**Max távolság:** ha `distance_traveled >= 3000 m` és nem eval, akkor
`success_state = True` — az epizód nem tarthat a végtelenségig.

### 4.7 Jutalom

```python
self.last_reward = self.reward_fn(self)
self.total_reward += self.last_reward
```

A jutalomfüggvény **magát a környezetet kapja meg** paraméterként, és abból
olvassa ki, amire szüksége van (`env.distance_from_center`,
`env.vehicle.get_speed()`, stb.). Emiatt tud a
[carla_env/reward.py](../carla_env/reward.py) teljesen külön fájlban élni.

Fontos: a reward függvény **be is állíthatja** a `terminal_state`-et —
így dönt a megállásról, lesodródásról, túl gyors haladásról.

### 4.8 Állapot kódolása

```python
encoded_state = self.encode_state_fn(self)
```

A nyers 160×80-as kép helyett a VAE **látens vektora** megy a hálónak
(plusz esetleg sebesség, manőver stb. — ezt a `state_commons.py` dönti el).
Ez drasztikusan csökkenti a tanulandó dimenziót.

### 4.9 Visszatérés

```python
done = self.terminal_state or self.success_state
return encoded_state, self.last_reward, done, False, info
```

Gymnasium 5-elemű formátum: `(obs, reward, terminated, truncated, info)`.

Az `info` dict a naplózáshoz: `total_reward`, `routes_completed`,
`total_distance`, `avg_center_dev`, `avg_speed`, `mean_reward`.

---

## 5. Megjelenítés — `render()`

Csak vizuális, a tanulást nem befolyásolja.

- **Módok:** `rgb_array_no_hud`, `rgb_array`, `state_pixels` — ezek csak
  visszaadnak egy tömböt (pl. videórögzítéshez). Alapból (`human`) rajzol.
- A spectator kamera képe a háttér, rárajzolva az **útvonal pontjai**
  (`_draw_path`).
- Jobb felül a **hálónak adott megfigyelés**, mellette a **VAE által
  visszafejtett kép** — így szemmel látod, mennyit "ért meg" a VAE.
- A HUD-on az `extra_info` lista szövegei: epizód, jutalom, manőver,
  átlagsebesség, átlagos eltérés stb.

### `_draw_path()` — hogyan kerül a 3D útvonal a 2D képre

Ez egy **kamera-projekció**:

1. `world_2_camera` — a kamera inverz transzformációs mátrixa
   (világ-koordináta → kamera-koordináta).
2. `build_projection_matrix(w, h, fov)` — a K mátrix a fókusztávolságból.
3. Koordináta-rendszer váltás: az Unreal Engine `(x, y, z)` rendszeréből
   a szokásos `(y, -z, x)`-be.
4. `K · pont`, majd osztás a mélységgel → képpont-koordináta.
5. `cv2.circle` — kék pöttyök az útvonalra, piros a célra.

Csak a **2 és 50 méter között** lévő pontokat rajzolja ki.

---

## 6. Callback-ek (szenzor-események)

| Metódus | Mikor fut | Mit csinál |
|---|---|---|
| `_on_collision` | ütközéskor | `terminal_state = True` — **kivéve** ha az "Road"-dal ütköztünk (az a talaj). |
| `_on_invasion` | sávfelfestés átlépésekor | Csak HUD üzenet — **nem zárja az epizódot**. |
| `_set_observation_image` | minden dashcam képkockánál | Bufferbe teszi a képet. |
| `_set_viewer_image` | spectator képkockánál | Bufferbe teszi. |

---

## 7. Ki mit állít be — az állapot-változók térképe

Mert ez a leggyakoribb kérdés: **honnan jön egy adott mező?**

| Változó | Hol áll be | Ki használja |
|---|---|---|
| `terminal_state` | `_on_collision`, `reward_fn`, ESC | `step()` → `done` |
| `success_state` | `step()` (max táv, eval cél) | `step()` → `done` |
| `distance_from_center` | `step()` 4.5 | `reward_fn`, HUD |
| `distance_from_center_history` | `step()` 4.6 | `reward_fn` (szórás) |
| `current_waypoint` | `step()` 4.4 | `reward_fn` (szög), `render` |
| `current_waypoint_index` | `step()` 4.4 | `reward_fn` (megállás-feltétel) |
| `episode_idx` | `reset()` | `reward_fn` (kiírás), `new_route` |
| `extra_info` | `reward_fn`, `render` | HUD |
| `fps` | `__init__` | `reward_fn` (időzítő) |

Ebből látszik, hogy a **jutalomfüggvény és a környezet szorosan összefonódik** —
a reward_fn nem csak olvas, hanem ír is (`terminal_state`, `extra_info`).

---

## 8. Amire érdemes figyelni

Néhány dolog, ami elsőre meglepő lehet a kódban:

- **Nincs fék.** Az akciótér csak kormány + gáz. Az autó lassítani csak
  gázelvétellel tud.
- **A `close()` hibás lenne:** a `self.carla_process` attribútum sehol nincs
  beállítva a `__init__`-ben, szóval `AttributeError`-t dobna. Jelenleg nem
  hívja meg semmi, ezért nem tűnik fel.
- **A `_get_observation()` busy-wait ciklusa** egy magot 100%-on pörget,
  amíg vár a képkockára.
- **A `next_waypoint` az útvonal legvégén nem frissül** — a
  `if self.current_waypoint_index < len(...) - 1` feltétel miatt az utolsó
  waypointnál a régi érték marad benne.
- **A szemantikus kamera egyedi palettája Python ciklussal megy**
  pixelenként (`wrappers.py`) — ez 160×80-nál is 12800 iteráció képkockánként,
  érezhetően lassít.
- **Az `intersection_routes` és `eval_routes` indexek Town02-höz** vannak
  hangolva. Más térképen értelmetlen (vagy hibás) indexek lennének.
