# 00 — Projekt áttekintés

## Mi ez a projekt egy mondatban?

Egy **megerősítéses tanulásos (Reinforcement Learning, RL) autonóm vezetés** projekt a
**CARLA** szimulátorban. Egy virtuális autó megtanul kamerakép + néhány mérőszám alapján
kormányozni és gyorsítani úgy, hogy egy előre kiszámolt útvonalat (route) végigkövessen,
a sáv közepén maradva, megfelelő sebességgel.

## A tanulás alapötlete (ha még nem ismerős az RL)

```
       ┌────────────────────────────────────────────────┐
       │                                                │
       │   ÁGENS (SAC neurális háló)                    │
       │   "Mit csináljak most?"                        │
       │                                                │
       └───────┬──────────────────────────▲─────────────┘
               │ akció (kormány, gáz)     │ állapot + jutalom
               ▼                          │
       ┌────────────────────────────────────────────────┐
       │   KÖRNYEZET (CARLA szimulátor + CarlaRouteEnv) │
       └────────────────────────────────────────────────┘
```

- **Állapot (state / observation):** amit az ágens "lát". Itt: a kamerakép tömörített
  változata + sebesség + kormányállás + a következő 15 útpont pozíciója stb.
- **Akció (action):** amit az ágens "csinál". Itt: 2 szám — kormányzás [-1, 1] és gáz [0, 1].
- **Jutalom (reward):** egy szám, ami megmondja, mennyire volt jó a lépés. Itt: sáv közepén
  vagy-e, jó sebességgel mész-e, jó irányba nézel-e.
- **Epizód (episode):** egy "menet" a spawn ponttól addig, amíg el nem rontja (ütközés,
  lesodródás, megállás) vagy be nem fejezi az útvonalat.

Az ágens célja: **maximalizálni az epizód alatt összegyűjtött jutalmat**. Ehhez sok ezer
lépésen keresztül próbálkozik, és a neurális hálója fokozatosan javul.

## A projekt 3 nagy pillére

### 1. VAE (Variational Autoencoder) — a "szem"
A kamerakép 160×80×3 = 38 400 szám. Ez túl sok egy RL ágensnek. A VAE ezt **64 számmá**
tömöríti (latent vektor), és ez a 64 szám kerül be az ágens állapotába.

Fontos: a VAE **előre be van tanítva** (`vae/model/best.tar`), a projekt nem tanítja tovább.
Csak használja mint fix "képtömörítőt".

📄 Részletek: [04-vae.md](04-vae.md)

### 2. Útvonaltervezés (navigation) — a "térkép"
A CARLA térképéből egy **gráfot** épít (útszakaszok = élek, kereszteződések = csomópontok),
és **A\* algoritmussal** megkeresi a legrövidebb utat A pontból B pontba. Az eredmény egy
lista: `[(waypoint, RoadOption), (waypoint, RoadOption), ...]` — 1 méterenkénti útpontok
és hogy ott mi a manőver (egyenes, balra, jobbra, sávkövetés).

📄 Részletek: [05-navigation.md](05-navigation.md)

### 3. RL környezet + SAC ágens — az "agy"
A `CarlaRouteEnv` egy Gymnasium-kompatibilis környezet: van `reset()` és `step(action)`
metódusa. A **Stable-Baselines3 SAC** algoritmus ezen tanul.

📄 Részletek: [02-carla-route-env.md](02-carla-route-env.md) és [01-train-config.md](01-train-config.md)

## Fájlstruktúra — mi hol van

```
RL-carla-ROS2/
├── train.py                    ← BELÉPÉSI PONT: itt indul a tanítás
├── config.py                   ← MINDEN beállítás egy helyen (hiperparaméterek)
├── utils.py                    ← segédek: TensorBoard callbackek, LR ütemező
├── requirements.txt            ← Python függőségek
│
├── carla_env/                  ← a környezet (a "játék" oldala)
│   ├── envs/
│   │   └── carla_route_env.py  ← ★ A LEGFONTOSABB FÁJL: a Gym környezet
│   ├── wrappers.py             ← CARLA objektumok (Vehicle, Camera, World) Python burkolói
│   ├── rewards.py              ← a jutalomfüggvény
│   ├── state_commons.py        ← állapot-kódolás (kép → VAE latent + mérőszámok)
│   │
│   ├── navigation/             ← útvonaltervezés (CARLA PythonAPI-ból átvéve)
│   │   ├── planner.py          ← compute_route_waypoints(): a fő belépési pont
│   │   ├── global_route_planner.py      ← A* gráfkeresés a térképen
│   │   ├── global_route_planner_dao.py  ← adathozzáférés a CARLA térképhez
│   │   ├── local_planner.py    ← RoadOption enum + PID-alapú útpontkövetés
│   │   ├── controller.py       ← PID szabályozók
│   │   ├── agent.py / basic_agent.py / roaming_agent.py  ← klasszikus (nem RL) ágensek
│   │
│   └── tools/
│       ├── hud.py              ← pygame képernyő-kijelzés (sebesség, jutalom, stb.)
│       └── misc.py             ← kis matek segédfüggvények
│
├── vae/
│   ├── models.py               ← VAE neurális háló definíciója
│   └── model/best.tar          ← ★ az előre betanított VAE súlyok
│
├── maps/Town04.osm             ← OpenStreetMap export (jelenleg nem használt)
└── tensorboard/                ← tanítási logok + mentett modellek (checkpointok)
```

## Futási folyamat — mi történik indításkor?

```
python train.py
   │
   ├─1─ load_vae('./vae/model', 64)                    [state_commons.py]
   │      → betölti a betanított VAE-t
   │
   ├─2─ create_encode_state_fn(vae, STATE)             [state_commons.py]
   │      → visszaad: observation_space, encode_state_fn, decode_vae_fn
   │
   ├─3─ CarlaRouteEnv(...)                             [carla_route_env.py]
   │      ├── csatlakozás a CARLA szerverhez (localhost:2000)
   │      ├── Town02 betöltése, szinkron mód 20 FPS-re
   │      ├── Tesla Model 3 spawnolása + kamerák + szenzorok
   │      └── reset() → első útvonal generálása
   │
   ├─4─ SAC(...) létrehozása vagy SAC.load(...)        [stable_baselines3]
   │
   └─5─ model.learn(total_timesteps=130_000)
          │
          └─── ismétlődő ciklus 130 000-szer:
                 obs → model.predict() → action
                 action → env.step(action) → új obs, reward, done
                 tapasztalat eltárolása replay bufferbe
                 időnként: gradiens lépés (tanulás)
```

## Előfeltételek a futtatáshoz

1. **CARLA szimulátor szerver** fut a háttérben (0.9.x verzió), 2000-es porton:
   ```bash
   ./CarlaUE4.sh
   ```
2. A `carla` Python csomag telepítve (a CARLA-hoz mellékelt `.egg` vagy pip).
3. GPU (a `train.py` `device='cuda'`-t használ).
4. `pip install -r requirements.txt`

```bash
python train.py
```

## Fontos: mit jelent a repó neve, "ROS2"?

A repó neve `RL-carla-ROS2`, és a legutóbbi commit üzenete `"vedes done folytatas ros implementation"`.
**A jelenlegi kódban NINCS ROS2 kód** — ez a következő fejlesztési lépés lesz.
A ROS2 integráció valószínűleg azt jelentené, hogy a `CarlaRouteEnv` helyett/mellett
ROS2 topicokon keresztül érkeznének a szenzoradatok és mennének ki a vezérlőparancsok.

📄 Ötletek a továbbfejlesztéshez: [07-tovabbfejlesztes.md](07-tovabbfejlesztes.md)

## Olvasási sorrend javaslat

| # | Fájl | Miről szól |
|---|------|-----------|
| 0 | **00-attekintes.md** (ez) | nagy kép |
| 1 | [01-train-config.md](01-train-config.md) | `train.py`, `config.py`, `utils.py` — indítás, hiperparaméterek |
| 2 | [02-carla-route-env.md](02-carla-route-env.md) | ★ a Gym környezet, sorról sorra |
| 3 | [03-wrappers-rewards-state.md](03-wrappers-rewards-state.md) | `wrappers.py`, `rewards.py`, `state_commons.py` |
| 4 | [04-vae.md](04-vae.md) | a VAE háló |
| 5 | [05-navigation.md](05-navigation.md) | útvonaltervezés, A*, RoadOption |
| 6 | [06-fogalomtar.md](06-fogalomtar.md) | szótár: minden szakkifejezés magyarul |
| 7 | [07-tovabbfejlesztes.md](07-tovabbfejlesztes.md) | hol lehet hozzányúlni, ROS2, ismert hibák |
