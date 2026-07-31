# 01 — `train.py`, `config.py`, `utils.py`

Ez a három fájl a "vezérlőpult": itt indul a tanítás és itt vannak a beállítások.

---

# 1. `train.py` — a belépési pont

## Importok (1–16. sor)

```python
import os
import time
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
```

- `SAC` — **Soft Actor-Critic**, az RL algoritmus. Folytonos akciótérhez való (nálunk a
  kormány és a gáz folytonos számok), és nagyon mintahatékony (sample efficient), ami
  szimulátorban fontos, mert egy lépés lassú.
- `CheckpointCallback` — időnként lementi a modellt tanítás közben.
- `configure` — beállítja, hova és milyen formátumban logoljon (stdout, csv, tensorboard).

```python
from carla_env.envs.carla_route_env import CarlaRouteEnv
from carla_env.state_commons import create_encode_state_fn, load_vae
from carla_env.rewards import reward_functions
from utils import HParamCallback, TensorboardCallback, write_json
```

A saját moduljaink. A `reward_functions` egy **szótár**: `{"reward_fn5": <függvény>}`.

## `main()` 

### VAE betöltés
```python
vae = load_vae('./vae/model', LSIZE)
```
Betölti az előre betanított VAE-t a `vae/model/best.tar` fájlból. `LSIZE = 64` a latent
vektor mérete. Ha a fájl nem létezik, kivételt dob.

### Állapot-kódoló létrehozása
```python
observation_space, encode_state_fn, decode_vae_fn = create_encode_state_fn(vae, STATE)
```
Ez a sor **három dolgot ad vissza**:

| Visszatérés | Mi ez | Ki használja |
|-------------|-------|--------------|
| `observation_space` | Gym `Dict` tér leírás — megmondja az SAC-nak, milyen alakú bemenetre számítson | SAC hálózat építése |
| `encode_state_fn` | függvény: `env → dict állapot` | `env.step()` minden lépésben |
| `decode_vae_fn` | függvény: `latent → kép` (visszafejtés, csak megjelenítéshez) | `env.render()` |

A `STATE` a `config.py`-ban van felsorolva — ez dönti el, mely mérőszámok kerüljenek be.


```python
rl_model_path = RELOAD_MODEL_PATH + "/model_final.zip"
```
Ha folytatni akarunk egy korábbi tanítást, innen tölti be. **Figyelem:** ez a sor akkor is
lefut, ha `RELOAD_MODEL = False` — de akkor csak egy sztringet állít elő, nem baj.


```python
env = CarlaRouteEnv(
    obs_res=OBS_RES,                              # (160, 80) — a dashcam felbontása
    host="localhost", port=2000,                  # CARLA szerver címe
    reward_fn=reward_functions[CONFIG["reward_fn"]],   # a jutalomfüggvény
    observation_space=observation_space,
    encode_state_fn=encode_state_fn,
    decode_vae_fn=decode_vae_fn,
    fps=FPS,                                      # 20 — szimulációs lépések/mp
    action_smoothing=ACTION_SMOOTHING,            # 0.75 — akciósimítás
    action_space_type='continuous',
    activate_spectator=ACTIVATE_SPECTATOR,        # külső nézetű kamera
    activate_render=ACTIVATE_RENDER,              # pygame ablak
)
```

**Fontos mellékhatás:** a `CarlaRouteEnv.__init__` végén meghívódik a `self.reset()`, tehát
már itt csatlakozik a CARLA-hoz, spawnolja az autót és legenerálja az első útvonalat.

### 41–46. sor — a modell
```python
if RELOAD_MODEL:
    model = SAC.load(rl_model_path, env=env, device='cuda',
                     tensorboard_log=LOG_DIR, verbose=1)
else:
    model = SAC('MultiInputPolicy', env=env, verbose=1, seed=SEED,
                tensorboard_log=LOG_DIR, device='cuda', **ALGORITHM_PARAMS)
```

- `'MultiInputPolicy'` — azért ez, mert az `observation_space` egy **`Dict`** (több
  különböző kulcs: `vae_latent`, `vehicle_measures`, `waypoints`, `maneuver`).
  Ha egyetlen `Box` lenne, `'MlpPolicy'` kellene.
- `**ALGORITHM_PARAMS` — a `config.py`-ból kicsomagolt hiperparaméterek.

### 47–53. sor — logolás beállítása
```python
model_suffix = f"{int(time.time())}_SAC"          # pl. "1778506873_SAC"
model_name = f'{model.__class__.__name__}_{model_suffix}'   # "SAC_1778506873_SAC"
model_dir = os.path.join(LOG_DIR, model_name)     # "tensorboard/SAC_1778506873_SAC"
new_logger = configure(model_dir, ["stdout", "csv", "tensorboard"])
model.set_logger(new_logger)
write_json(CONFIG, os.path.join(model_dir, 'config.json'))
```
Minden futás **saját mappát kap időbélyeggel**, és a `config.json`-ba lementi, milyen
beállításokkal futott. Ez jó gyakorlat: később rekonstruálható, mi hogyan lett tanítva.

### 55–68. sor — a tanítás
```python
model.learn(
    total_timesteps=TOTAL_STEPS,      # 130_000
    callback=[
        HParamCallback(CONFIG),        # hiperparaméterek TensorBoardba
        TensorboardCallback(1),        # egyedi metrikák epizód végén
        CheckpointCallback(
            save_freq=TOTAL_STEPS // NUM_CHECKPOINTS,   # 13 000 lépésenként
            save_path=model_dir,
            name_prefix="model",
        ),
    ],
    reset_num_timesteps=False,
)
```

`reset_num_timesteps=False` — folytatáskor nem nullázza a lépésszámlálót, így a
TensorBoard grafikonok folytatólagosak lesznek.

### 69–74. sor — mentés
```python
model.save(os.path.join(model_dir, "model_final"))
...
except KeyboardInterrupt:
    model.save(os.path.join(model_dir, "model_interrupted"))
```
Ha Ctrl+C-vel megszakítod, **nem vész el a modell** — `model_interrupted.zip` néven ment.

---

# 2. `config.py` — minden beállítás

## Alapbeállítások (3–16. sor)

| Változó | Érték | Jelentés |
|---------|-------|----------|
| `LSIZE` | 64 | VAE latent vektor mérete |
| `LOG_DIR` | `"tensorboard"` | logok mappája |
| `RELOAD_MODEL` | `True` | **folytatás korábbi modellből** |
| `RELOAD_MODEL_PATH` | `"./tensorboard/SAC_1778506873_SAC"` | honnan töltsön |
| `TOTAL_STEPS` | 130 000 | összes tanítási lépés |
| `SEED` | 100 | véletlenszám-mag (reprodukálhatóság) |
| `OBS_RES` | (160, 80) | dashcam felbontása (szélesség, magasság) |
| `VAE_MODEL` | `"vae_64_augmentation"` | csak dokumentációs címke (a kód nem használja!) |
| `STATE` | lista, lásd lent | mely mérőszámok kerülnek az állapotba |
| `ACTION_SMOOTHING` | 0.75 | akciósimítás mértéke |
| `NUM_CHECKPOINTS` | 10 | hány mentés készüljön |
| `FPS` | 20 | szimulációs lépés/mp (1 lépés = 0.05 s) |
| `ACTIVATE_SPECTATOR` | True | külső nézetű kamera bekapcsolva |
| `ACTIVATE_RENDER` | True | pygame ablak bekapcsolva |

> **Tipp:** ha gyorsabban akarsz tanítani, állítsd `ACTIVATE_RENDER = False` és
> `ACTIVATE_SPECTATOR = False` — a renderelés és a második kamera sok időt visz.

### `STATE` — az állapottér összetétele (11. sor)
```python
STATE = ["steer", "throttle", "speed", "waypoints", "angle_next_waypoint", "maneuver"]
```

Ez a lista dönti el, mit "lát" az ágens. A `state_commons.py` ezt fordítja `measure_flags`
logikai listává. Az eredmény jelenleg:

| Kulcs az obs-ban | Alak | Tartalom |
|------------------|------|----------|
| `vae_latent` | (64,) | a dashcam kép tömörítve |
| `vehicle_measures` | (4,) | [steer, throttle, speed, angle_next_waypoint] |
| `maneuver` | Discrete(4) | LANEFOLLOW / LEFT / RIGHT / STRAIGHT |
| `waypoints` | (15, 2) | a következő 15 útpont **az autó koordinátarendszerében** |

### `ACTION_SMOOTHING` — miért kell?  
```python
new_control = old_control * 0.75 + action * 0.25
```
Az RL ágens kimenete zajos, lépésenként ugrálhat. A simítás miatt a kormány nem tud
hirtelen -1-ről +1-re ugrani, ami valósághűbb és stabilabb vezetést eredményez.
Cserébe **lassabb a reakció** — ha 0.75 túl magas, az autó "lomha" lesz.

## `ALGORITHM_PARAMS` — SAC hiperparaméterek 

```python
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
```

| Paraméter | Érték | Mit csinál |
|-----------|-------|-----------|
| `learning_rate` | ütemezett 1e-4 → 5e-7 | tanulási ráta, idővel csökken |
| `buffer_size` | 300 000 | **replay buffer**: ennyi múltbeli tapasztalatot tárol |
| `batch_size` | 256 | egy gradiens lépéshez ennyi mintát húz ki |
| `ent_coef` | `'auto'` | **entrópia együttható** automatikusan hangolva — ez a felfedezés (exploration) mértéke |
| `gamma` | 0.98 | **diszkont faktor**: mennyire számít a jövő. 0.98 ≈ ~50 lépés (2.5 mp) előrelátás |
| `tau` | 0.02 | a cél-hálózat (target network) frissítési sebessége (soft update) |
| `train_freq` | 64 | 64 környezeti lépésenként tanul |
| `gradient_steps` | 64 | ...és akkor 64 gradiens lépést tesz |
| `learning_starts` | 10 000 | az első 10 000 lépés csak véletlen adatgyűjtés |
| `use_sde` | True | **State-Dependent Exploration**: a zaj az állapottól függ, nem tisztán véletlen — folytonos vezérlésnél simább felfedezést ad |
| `net_arch` | [500, 300] | 2 rejtett réteg, 500 és 300 neuronnal |
| `log_std_init` | -3 | kezdeti szórás logaritmusa ≈ e⁻³ ≈ 0.05, tehát kis kezdeti zaj |

**Miért `train_freq=64` és `gradient_steps=64`?** Mert a CARLA lépés lassú (szinkron mód,
renderelés). Így nem tanul minden lépés után külön (ami sok apró GPU hívás lenne), hanem
kötegelve — hatékonyabb.

## `REWARD_PARAMS` — jutalom paraméterek (32–41. sor)

| Paraméter | Érték | Jelentés |
|-----------|-------|----------|
| `early_stop` | True | korai epizódvég engedélyezve |
| `min_speed` | 20.0 km/h | ez alatt arányosan csökken a sebességjutalom |
| `max_speed` | 35.0 km/h | e fölött **azonnali epizódvég** (büntetéssel) |
| `target_speed` | 25.0 km/h | ideális sebesség |
| `max_distance` | 2.0 m | ennyivel térhet el a sáv közepétől; e fölött epizódvég |
| `max_std_center_lane` | 0.35 | a középvonaltól való eltérés szórásának maximuma (kanyargás büntetése) |
| `max_angle_center_lane` | 90° | szöghiba normalizálási határa |
| `penalty_reward` | -10 | büntetés hibás epizódvégért |

📄 Részletek a jutalomról: [03-wrappers-rewards-state.md](03-wrappers-rewards-state.md#rewardspy)

## `CONFIG` (43–54. sor)

Egy összefoglaló szótár, ami:
1. eldönti, melyik jutalomfüggvényt használjuk (`"reward_fn": "reward_fn5"`),
2. lementődik `config.json`-ba,
3. bekerül a TensorBoard hiperparaméter-fülébe.

> **Megjegyzés:** a `"wrappers": []` kulcs jelenleg **nem használt** — valószínűleg egy
> korábbi verzió maradványa, ahol Gym wrappereket lehetett hozzáadni.

---

# 3. `utils.py` — segédfüggvények

## `write_json(data, path)` (9–22. sor)

```python
def write_json(data, path):
    config_dict = {}
    with open(path, 'w', encoding='utf-8') as f:
        for k, v in data.items():
            if isinstance(v, str) and v.isnumeric():
                config_dict[k] = int(v)
            elif isinstance(v, dict):
                config_dict[k] = dict()
                for k_inner, v_inner in v.items():
                    config_dict[k][k_inner] = v_inner.__str__()
                config_dict[k] = str(config_dict[k])
            else:
                config_dict[k] = v.__str__()
        json.dump(config_dict, f, indent=4)
```

**Miért ilyen bonyolult?** Mert a `CONFIG` tartalmaz olyan értékeket, amiket a `json` modul
nem tud sorosítani — például a `lr_schedule(...)` egy **függvényobjektum**. Ezért mindent
sztringgé alakít (`__str__()`).

Ezért van a `lr_schedule`-ben az a trükk, hogy felüldefiniálja a `__str__`-t (lásd lent).

## `HParamCallback` (25–50. sor)

```python
class HParamCallback(BaseCallback):
    def _on_training_start(self) -> None:
        hparam_dict = {}
        # ... ugyanaz a sztringesítés, mint write_json-ban ...
        metric_dict = {
            "rollout/ep_len_mean": 0,
            "train/value_loss": 0,
        }
        self.logger.record("hparams", HParam(hparam_dict, metric_dict), exclude=(...))
```

Ez a TensorBoard **HPARAMS** fülét tölti fel. Így több futást össze tudsz hasonlítani:
"melyik hiperparaméter-kombináció adott jobb `ep_len_mean`-t?".

A `metric_dict` a *nyomon követendő metrikák* listája (a 0 érték csak helykitöltő).

## `TensorboardCallback`

```python
class TensorboardCallback(BaseCallback):
    def _on_step(self) -> bool:
        if self.locals['dones'][0]:              # ha épp véget ért egy epizód
            self.logger.record("custom/total_reward",     self.locals['infos'][0]['total_reward'])
            self.logger.record("custom/routes_completed", self.locals['infos'][0]['routes_completed'])
            self.logger.record("custom/total_distance",   self.locals['infos'][0]['total_distance'])
            self.logger.record("custom/avg_speed",        self.locals['infos'][0]['avg_speed'])
            self.logger.record("custom/mean_reward",      self.locals['infos'][0]['mean_reward'])
            self.logger.dump(self.num_timesteps)
        return True
```

- `self.locals` — a Stable-Baselines3 belső ciklusának lokális változói. Trükkös, de működő
  módszer, hogy hozzáférjünk a `dones` és `infos` értékekhez.
- `infos[0]` — a `[0]` azért van, mert a SB3 mindig vektorizált környezetet vár; nálunk
  1 környezet van, tehát mindig a 0. index.
- Ezek az értékek a `CarlaRouteEnv.step()` `info` szótárából jönnek (355–363. sor).
- `return True` — ha `False`-t adna vissza, leállítaná a tanítást.

**Ez adja a legfontosabb saját metrikákat a TensorBoardon:**
- `custom/routes_completed` — hány útvonalat teljesített (ez a fő siker-mérőszám!)
- `custom/total_distance` — mennyit ment egy epizódban
- `custom/mean_reward` — átlagos jutalom lépésenként

## `lr_schedule(initial_value, end_value, rate)` (71–79. sor)

```python
def lr_schedule(initial_value: float, end_value: float, rate: float):
    def func(progress_remaining: float) -> float:
        if progress_remaining <= 0:
            return end_value
        return end_value + (initial_value - end_value) * (10 ** (rate * math.log10(progress_remaining)))
    ...
```

A Stable-Baselines3 a `learning_rate` helyére elfogad **függvényt**, amit
`progress_remaining` értékkel hív (1.0 = kezdet, 0.0 = vég).

A képlet `10^(rate * log10(p))` = `p^rate`. Tehát `rate=2`-vel ez egyszerűen:

```
lr(p) = end + (initial - end) * p²
```

| Haladás | `progress_remaining` | Tanulási ráta |
|---------|---------------------|---------------|
| 0 %     | 1.0 | 1e-4 |
| 50 %    | 0.5 | ≈ 2.5e-5 |
| 90 %    | 0.1 | ≈ 1e-6 |
| 100 %   | 0.0 | 5e-7 |

**Miért jó?** Az elején nagy lépésekkel tanul (gyors haladás), a végén finomhangol
(stabil konvergencia).

```python
func.__str__ = lambda: f"lr_schedule({initial_value}, {end_value}, {rate})"
```
Ez a sor a `write_json` miatt kell, hogy olvasható legyen a mentett configban.

> **Apró hiba:** a `func.__str__ = ...` értékadás **nem működik úgy, ahogy szánták** —
> Pythonban a `str(obj)` a *típuson* keresi a `__str__`-t, nem a példányon. Egy függvényre
> beállított `__str__` attribútum figyelmen kívül marad, így a JSON-ba mégis a
> `<function lr_schedule.<locals>.func at 0x...>` kerül. Nem töri el a futást, csak a log
> lesz kevésbé olvasható.

---

➡️ Következő: [02-carla-route-env.md](02-carla-route-env.md) — a környezet, sorról sorra.
