"""Mire figyel a betanitott SAC policy vezetes kozben? Ablacios meres.

  collect  CARLA-ban vezet a modell (determinisztikusan), seedelt route-okon.
           Egy epizod = egy route: a route vegen megall, meg a teleport elott.
           Elobb "baseline": a valodi obs-okat elmenti. Utana UGYANAZOKON a
           route-okon ugy vezet, hogy egy bemenetcsoportot kicserelunk vagy
           eltolunk (zart hurok) - ez az oksagi meres.
  analyze  CARLA nelkul, a baseline obs-okon: permutacios fontossag (egy
           csoportot egy masik lepes ertekere csereljuk, es nezzuk, mennyit
           valtozik az akcio). Csak kiegeszito: a policy nyers kimenete zajos,
           amit vezetes kozben az action_smoothing kisimit.
  summary  a zart hurku eredmenyek tablazata.

A csere csak az AGENSNEK adott obs-ot erinti; a reward, a terminal es a
metrikak a valodi allapotbol szamolodnak.

Figyelem: a 2026-09-24 elott tanitott modellek meg a z-bugos
wrappers.distance_to_line-nal tanultak (6 m-nel magasabb uton Off-track),
ezert emelkedon maguktol megallhatnak - lasd a SIK vs. EMELKEDOS tablat.

Futtatas a repo gyokerebol, a carla_rl envben, futo CARLA szerverrel:
  python eval_ablation.py collect --model tensorboard/<run>/model_interrupted.zip
  python eval_ablation.py analyze --model tensorboard/<run>/model_interrupted.zip
"""
import argparse
import json
import os
import time

import numpy as np
import torch as th
from stable_baselines3 import SAC

import carla_env.reward as reward_mod
from carla_env.envs.carla_route_env import CarlaRouteEnv
from carla_env.reward import reward_fn
from config import CONFIG
from utils import (create_encode_state_fn, create_observation_space,
                   load_cam_ae, load_lidar_ae)

OBS_KEYS = ["cam_latent", "lidar_latent", "maneuver", "vehicle_measures", "waypoints"]
SHIFT_M = 1.0   # savon belul marad (a sav ~3.5 m), igy a kamera es a waypoint "verseng"

CONDITIONS = {
    "baseline":        "valodi obs",
    "baseline_rep":    "valodi obs meg egyszer - a zaj-alsohatar (a lidar veletlenul ejt pontokat)",
    "no_cam":          "kamera <- veletlen masik lepese",
    "no_lidar":        "lidar <- veletlen masik lepese",
    "no_sensors":      "kamera + lidar <- veletlen masik lepese (minden lepesben mas)",
    "frozen_sensors":  "kamera + lidar <- egyetlen rogzitett kep az egesz epizodban",
    "sensors_swapped": "kamera + lidar <- egy MASIK route folyama (kovetkezetes, de hamis)",
    "no_waypoints":    "csak a waypointok <- veletlen masik lepese",
    "no_route":        "waypoints + szog + maneuver <- veletlen masik lepese",
    "no_all":          "szenzorok + route <- veletlen masik lepese (also hatar)",
    "wp_shift_left":   f"waypointok {SHIFT_M} m-rel balra tolva",
    "wp_shift_right":  f"waypointok {SHIFT_M} m-rel jobbra tolva",
}

# Az obs "reszei". vm = vehicle_measures oszlopa: 0 steer, 1 throttle (az
# elozo, simitott akcio), 2 sebesseg, 3 szog a waypointhoz.
PARTS = {"cam": ("cam_latent", None), "lidar": ("lidar_latent", None),
         "wp": ("waypoints", None), "man": ("maneuver", None),
         "steer_prev": ("vehicle_measures", 0), "thr_prev": ("vehicle_measures", 1),
         "speed": ("vehicle_measures", 2), "angle": ("vehicle_measures", 3)}
ROUTE = ["wp", "man", "angle"]
SENSORS = ["cam", "lidar"]
PREV = ["steer_prev", "thr_prev"]

RANDOM_SWAP = {"no_cam": ["cam"], "no_lidar": ["lidar"], "no_sensors": SENSORS,
               "no_waypoints": ["wp"], "no_route": ROUTE, "no_all": SENSORS + ROUTE}


def load_model(path, device):
    # A tanitaskori lr_schedule closure-kent van picklezve; kiertekelesnel nem kell.
    return SAC.load(path, device=device,
                    custom_objects={"learning_rate": 0.0, "lr_schedule": lambda _: 0.0})


def to_np(obs):
    """Ugyanazok a dtype-ok, mint tanitaskor (a DummyVecEnv a space dtype-jara konvertal)."""
    return dict(cam_latent=np.asarray(obs["cam_latent"], np.float32),
                lidar_latent=np.asarray(obs["lidar_latent"], np.float32),
                maneuver=np.int64(obs["maneuver"]),
                vehicle_measures=np.asarray(obs["vehicle_measures"], np.float32),
                waypoints=np.asarray(obs["waypoints"], np.float32))


def lateral_offset(wp):
    """Az auto oldaliranyu helyzete a VALODI route-hoz kepest [m], jobbra +.
    A waypointok az auto rendszereben vannak (x jobbra, y elore); ahol a route
    az auto y=0 vonalat metszi x0-nal, ott az auto -x0-ra van tole."""
    x, y = wp[:, 0], wp[:, 1]
    i = int(np.argmax(y > 0))
    if i == 0 or y[i] <= 0:
        return float(-x[0])
    f = -y[i - 1] / (y[i] - y[i - 1])
    return float(-(x[i - 1] + f * (x[i] - x[i - 1])))


def turn_ahead(env, a=15, b=50):
    """A route iranyvaltozasa [fok] az auto elott 15..50 m kozott: kozeledo
    kanyar, amit a 15 pontos waypoint-ablak meg nem lat, a kamera viszont igen."""
    r, i = env.route_waypoints, env.current_waypoint_index
    ya = r[min(i + a, len(r) - 1)][0].transform.rotation.yaw
    yb = r[min(i + b, len(r) - 1)][0].transform.rotation.yaw
    return float((yb - ya + 180.0) % 360.0 - 180.0)


def _put(o, pool, j, part):
    key, col = PARTS[part]
    if col is None:
        o[key] = pool[key][j]
    else:
        o[key][col] = pool[key][j, col]


def ablate(cond, obs, ctx):
    if cond in ("baseline", "baseline_rep"):
        return obs
    o = {k: np.copy(v) for k, v in obs.items()}
    pool = ctx["pool"]
    if cond in RANDOM_SWAP:
        # Egy j a csoport minden reszere: a csere belul konzisztens marad.
        j = ctx["rng"].integers(len(pool["maneuver"]))
        for part in RANDOM_SWAP[cond]:
            _put(o, pool, j, part)
    elif cond == "frozen_sensors":
        for part in SENSORS:
            _put(o, pool, ctx["j_frozen"], part)
    elif cond == "sensors_swapped":
        idx = ctx["other_idx"]
        for part in SENSORS:
            _put(o, pool, idx[min(ctx["t"], len(idx) - 1)], part)
    elif cond == "wp_shift_left":
        o["waypoints"][:, 0] -= SHIFT_M
    elif cond == "wp_shift_right":
        o["waypoints"][:, 0] += SHIFT_M
    return o


# =============================================================================
# collect
# =============================================================================

def run_episode(env, model, cond, seed, max_steps, ctx):
    # A _sample_route a globalis np.random-ot hasznalja: ugyanaz a seed ->
    # ugyanaz a route minden feltetelben.
    np.random.seed(seed)
    ctx["rng"] = np.random.default_rng([seed, list(CONDITIONS).index(cond)])
    # A reward.py szamlaloja csak terminalkor nullazodik - epizodonkent kezdjuk tisztan.
    reward_mod._low_speed_timer = 0.0
    obs, _ = env.reset()
    reset_reason = env.extra_info[-2] if len(env.extra_info) >= 2 else ""
    env.extra_info.clear()

    route = env.route_waypoints
    ep = dict(condition=cond, seed=seed, route_len=len(route),
              route_zmax=round(max(wp.transform.location.z for wp, _ in route), 2))
    if env.terminal_state:
        # A reset() vegen futo step(None) mar terminalt - nem a policy hibaja.
        ep.update(end="reset_terminal", reset_reason=reset_reason, steps=0)
        return ep
    pool = ctx["pool"]
    ctx["j_frozen"] = int(ctx["rng"].integers(len(pool["maneuver"]))) if pool is not None else None
    rec = ctx.get("record")

    speeds, devs, offsets = [], [], []
    end = "timeout"
    for t in range(max_steps):
        ctx["t"] = t
        true_obs = to_np(obs)
        action, _ = model.predict(ablate(cond, true_obs, ctx), deterministic=True)
        if rec is not None:
            for k in OBS_KEYS:
                rec[k].append(true_obs[k])
            rec["action"].append(action)
            rec["seed"].append(seed)
            rec["t"].append(t)
            rec["turn_ahead"].append(turn_ahead(env))

        obs, _, done, _, _ = env.step(action)

        speeds.append(env.vehicle.get_speed())
        devs.append(env.distance_from_center)
        offsets.append(lateral_offset(np.asarray(obs["waypoints"], np.float32)))
        reason = env.extra_info[-2] if len(env.extra_info) >= 2 else ""
        env.extra_info.clear()
        if done:
            # Utkozeskor a callback allitja a terminal_state-et, a reward_fn
            # ilyenkor nem ad okot ("Running...").
            end = "Collision" if reason in ("", "Running...") else reason
            break
        if env.current_waypoint_index >= len(route) - 1:
            # A kovetkezo step() mar uj route-ra teleportalna.
            end = "route_done"
            break

    if end == "timeout" and env.distance_traveled < 5.0:
        end = "stuck"   # el sem indult - a "Vehicle stopped" csak az 1. waypoint utan elesedik
    cw = env.current_waypoint
    ep.update(
        end=end, steps=len(speeds),
        # min: a route utolso lepeseben az env indexe a % len miatt korbeerhet (akar 2*len-ig).
        progress=round(min(1.0, (env.current_waypoint_index + 1) / len(route)), 4),
        distance_m=round(env.distance_traveled, 1),
        avg_speed_kmh=round(float(np.mean(speeds)), 2),
        mean_dev_m=round(float(np.mean(devs)), 3),
        mean_abs_offset_m=round(float(np.mean(np.abs(offsets))), 3),
        # A tolasos feltetelekhez: a bealt allapot oldaliranyu helyzete.
        offset_med_after60=round(float(np.median(offsets[60:])), 3) if len(offsets) > 60 else None,
        end_maneuver=int(env.current_road_maneuver.value), end_junction=bool(cw.is_junction),
        end_z=round(cw.transform.location.z, 2), end_speed_kmh=round(speeds[-1], 1))
    return ep


def _swapped_indices(pool, seeds, seed, min_len=300):
    """A sensors_swapped-hoz: a kovetkezo (legalabb min_len hosszu) seed
    baseline-folyamanak pool-indexei, lepesenkent sorban."""
    i = seeds.index(seed)
    for k in range(1, len(seeds)):
        other = seeds[(i + k) % len(seeds)]
        idx = np.where(pool["seed"] == other)[0]
        if len(idx) >= min_len:
            return idx[np.argsort(pool["t"][idx])]
    return np.where(pool["seed"] != seed)[0]


def collect(args, out_dir):
    device = th.device(args.device)
    cam_ae = load_cam_ae(CONFIG["camera"], device)
    lidar_ae, lidar_shape = load_lidar_ae(CONFIG["lidar"], device)
    obs_space = create_observation_space(cam_ae, lidar_shape)
    model = load_model(args.model, device)
    for k in OBS_KEYS:
        saved, now = model.observation_space[k], obs_space[k]
        assert type(saved) is type(now) and saved.shape == now.shape, \
            f"{k}: a modell {saved}-et var, a mostani kod {now}-t ad"

    conds = sorted(args.conditions, key=lambda c: c != "baseline")   # a baseline mindig elol
    seeds = [args.seed + i for i in range(args.episodes)]
    pool_path = os.path.join(out_dir, "baseline_obs.npz")
    ep_path = os.path.join(out_dir, "episodes.jsonl")
    if "baseline" in conds and os.path.exists(ep_path):
        os.replace(ep_path, os.path.join(out_dir, "episodes_prev.jsonl"))
    run_id = time.strftime("%Y%m%d-%H%M%S")

    env_cfg = CONFIG["env"]
    env = None
    try:
        env = CarlaRouteEnv(
            obs_res=env_cfg["obs_res"], viewer_res=env_cfg["viewer_res"],
            host=env_cfg["host"], port=env_cfg["port"], town=env_cfg["town"],
            max_route_length=env_cfg["max_route_length"],
            min_route_length=env_cfg["min_route_length"],
            reward_fn=reward_fn, observation_space=obs_space,
            encode_state_fn=create_encode_state_fn(cam_ae, lidar_ae, CONFIG, device),
            fps=env_cfg["fps"], action_smoothing=env_cfg["action_smoothing"],
            action_space_type=env_cfg["action_space_type"],
            activate_spectator=False, activate_render=False, activate_lidar=True)

        pool = None
        for cond in conds:
            ctx = dict(pool=pool, record=None)
            if cond == "baseline":
                ctx["record"] = {k: [] for k in OBS_KEYS + ["action", "seed", "t", "turn_ahead"]}
            elif pool is None:
                d = np.load(pool_path)
                pool = {k: d[k] for k in d.files}
                ctx["pool"] = pool
                print(f"korabbi baseline pool: {pool_path}, seedek {sorted(set(pool['seed'].tolist()))}")
            for seed in seeds:
                if cond == "sensors_swapped":
                    ctx["other_idx"] = _swapped_indices(pool, seeds, seed)
                t0 = time.perf_counter()
                ep = run_episode(env, model, cond, seed, args.max_steps, ctx)
                ep.update(wall_s=round(time.perf_counter() - t0, 1), run_id=run_id, max_steps=args.max_steps)
                print(f"[{cond:15s}] seed {seed}: {ep['end']:15s} {ep.get('steps', 0):5d} step "
                      f"progress {ep.get('progress', 0):5.2f}  {ep.get('avg_speed_kmh', 0):5.1f} km/h  "
                      f"|offset| {ep.get('mean_abs_offset_m', 0):.2f} m  zmax {ep['route_zmax']:5.1f}  "
                      f"({ep['wall_s']} s)", flush=True)
                with open(ep_path, "a") as f:
                    f.write(json.dumps(ep) + "\n")
            if cond == "baseline":
                rec = ctx["record"]
                np.savez(pool_path, **{k: np.stack(v) for k, v in rec.items()})
                pool = {k: np.stack(v) for k, v in rec.items()}
                print(f"baseline obs mentve: {pool_path} ({len(rec['action'])} lepes)")
    finally:
        if env is not None:
            try:
                env.world.destroy()
            except Exception as e:
                print("cleanup hiba (actorok):", e)
        # Kulon kliensen, hogy akkor is lefusson, ha az env felepitese szallt el
        # (a szinkron mod mar az __init__ elejen bekapcsol). Enelkul a szerver
        # a kovetkezo kliensnek "befagyottnak" latszana.
        try:
            import carla
            client = carla.Client(env_cfg["host"], env_cfg["port"])
            client.set_timeout(10.0)
            world = client.get_world()
            s = world.get_settings()
            s.synchronous_mode = False
            s.fixed_delta_seconds = None
            world.apply_settings(s)
        except Exception as e:
            print("cleanup hiba (szinkron mod):", e)
    summarize(out_dir)


# =============================================================================
# summary
# =============================================================================

def _boot_ci(x, n_boot=4000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n_boot, len(x))).mean(1)
    return float(np.percentile(m, 5)), float(np.percentile(m, 95))


def load_episodes(out_dir):
    """Csak a legutolso futas (run_id) sorai, (feltetel, seed)-enkent az utolso."""
    rows = [json.loads(l) for l in open(os.path.join(out_dir, "episodes.jsonl"))]
    last = {}
    for r in rows:
        if "progress" in r:
            # A 2026-09-24 elotti env-ben az index a route vegen "% len"-nel
            # korbeerhetett: a progress 1 folott lehetett, es ha a korbeeres
            # felutton allt meg, a current_waypoint egy tavoli pontra ugrott,
            # amibol hamis Off-track lett. Aki elerte a route vegere, az
            # teljesitette - barmi jott is utana ugyanabban a lepesben.
            if r["progress"] >= 1.0 and r["end"] != "route_done":
                r["end_raw"], r["end"] = r["end"], "route_done"
            r["progress"] = min(1.0, r["progress"])
        last[(r["condition"], r["seed"])] = r
    return list(last.values())


def summarize(out_dir):
    all_eps = load_episodes(out_dir)
    bad = sorted({e["seed"] for e in all_eps if e["end"] == "reset_terminal"})
    if bad:
        print(f"kihagyott seedek (mar a reset alatt terminaltak): {bad}")
    eps = [e for e in all_eps if e["seed"] not in bad]
    fails = lambda e: e["end"] not in ("route_done", "timeout")
    base = {e["seed"]: e for e in eps if e["condition"] == "baseline"}

    print("\n=== ZART HUROK: vegigviszi-e a route-ot? (ugyanazok a route-ok feltetelenkent) ===")
    print(f"{'feltetel':16s} {'n':>3s} {'kesz':>6s} {'haladas':>8s} {'d haladas vs baseline [90% CI]':>31s} "
          f"{'hiba/km':>8s} {'km/h':>6s} {'|offset|':>8s}  befejezesek")
    for cond in CONDITIONS:
        e = [x for x in eps if x["condition"] == cond]
        if not e:
            continue
        km = sum(x["distance_m"] for x in e) / 1000.0
        n_fail = sum(fails(x) for x in e)
        ends = {}
        for x in e:
            ends[x["end"]] = ends.get(x["end"], 0) + 1
        paired = [x["progress"] - base[x["seed"]]["progress"] for x in e if x["seed"] in base]
        lo, hi = _boot_ci(paired)
        pd = f"{np.mean(paired):+.2f} [{lo:+.2f}, {hi:+.2f}]" if paired and cond != "baseline" else ""
        print(f"{cond:16s} {len(e):3d} {sum(x['end'] == 'route_done' for x in e):3d}/{len(e):<2d} "
              f"{np.mean([x['progress'] for x in e]):8.2f} {pd:>31s} {n_fail / max(km, 1e-6):8.2f} "
              f"{np.mean([x['avg_speed_kmh'] for x in e]):6.1f} {np.mean([x['mean_abs_offset_m'] for x in e]):8.2f}  {ends}")
    print("  kesz = a route vegeig eljutott; haladas = a route hanyad reszeig jutott (0..1)")

    # A tanitasban a z-bug miatt a 6 m feletti ut Off-track volt: kulon nezzuk
    # a sik es az emelkedos route-okat.
    print("\n=== SIK vs. EMELKEDOS ROUTE (route_zmax < 1 m / >= 1 m): kesz / osszes ===")
    for cond in CONDITIONS:
        e = [x for x in eps if x["condition"] == cond]
        if e:
            flat = [x for x in e if x["route_zmax"] < 1.0]
            hill = [x for x in e if x["route_zmax"] >= 1.0]
            done = lambda xs: sum(x["end"] == "route_done" for x in xs)
            print(f"{cond:16s} sik {done(flat)}/{len(flat)}   emelkedos {done(hill)}/{len(hill)}  "
                  + ", ".join(f"s{x['seed']}:{x['end']}@z={x.get('end_z')}" for x in hill))

    # Hol hibazott: kereszetezodesben (ahol route nelkul nem tudhatja az iranyt)
    # vagy sima uton, illetve emelkedon.
    print("\n=== HOL ERT VEGET A HIBAS EPIZOD ===")
    for cond in CONDITIONS:
        e = [x for x in eps if x["condition"] == cond and fails(x)]
        if e:
            print(f"{cond:16s} kereszetezodesben {sum(x['end_junction'] for x in e)}, "
                  f"sima uton {sum(not x['end_junction'] for x in e)}; "
                  + ", ".join(f"s{x['seed']}:{x['end']}@{x['progress']:.2f} z={x['end_z']}" for x in e))

    # Tolas: mennyire koveti a hamis waypointot? 1 = teljesen, 0 = egyaltalan nem.
    print("\n=== WAYPOINT-TOLAS: kovetesi erosites (1 = a waypointot koveti, 0 = a szenzort) ===")
    for cond, sign in [("wp_shift_left", -1.0), ("wp_shift_right", +1.0)]:
        g = [(x["offset_med_after60"] - base[x["seed"]]["offset_med_after60"]) / (sign * SHIFT_M)
             for x in eps if x["condition"] == cond and x["seed"] in base
             and x.get("offset_med_after60") is not None and base[x["seed"]].get("offset_med_after60") is not None]
        if g:
            print(f"{cond:16s} median {np.median(g):.2f}  (seedenkent: {', '.join(f'{v:.2f}' for v in g)})")


# =============================================================================
# analyze
# =============================================================================

def predict_batched(model, obs, bs=512):
    n = len(obs["maneuver"])
    return np.concatenate([model.predict({k: v[s:s + bs] for k, v in obs.items()},
                                         deterministic=True)[0] for s in range(0, n, bs)])


def permuted(obs, parts, perm):
    o = dict(obs)
    vm = None
    for part in parts:
        key, col = PARTS[part]
        if col is None:
            o[key] = obs[key][perm]
        else:
            if vm is None:
                vm = obs["vehicle_measures"].copy()
            vm[:, col] = obs["vehicle_measures"][perm, col]
    if vm is not None:
        o["vehicle_measures"] = vm
    return o


ALL_PARTS = list(PARTS)
# (csoport neve, a permutalt reszek). A "CSAK ... marad" sorok mindent
# permutalnak, KIVEVE az adott csoportot: ott a KIS ertek jelenti, hogy az a
# csoport egymagaban eleg a donteshez.
GROUPS = [
    ("waypoints", ["wp"]),
    ("szog", ["angle"]),
    ("maneuver", ["man"]),
    ("route (wp+szog+man)", ROUTE),
    ("route + elozo akcio", ROUTE + PREV),
    ("sebesseg", ["speed"]),
    ("elozo akcio", PREV),
    ("kamera", ["cam"]),
    ("lidar", ["lidar"]),
    ("kamera+lidar", SENSORS),
    ("kamera+lidar + elozo akcio", SENSORS + PREV),
    ("CSAK route + sebesseg marad", [p for p in ALL_PARTS if p not in ROUTE + ["speed"]]),
    ("CSAK szenzor + sebesseg marad", [p for p in ALL_PARTS if p not in SENSORS + ["speed"]]),
    ("MINDEN", ALL_PARTS),
]


def _window_turn_deg(wp):
    """A 15 pontos ablak iranyvaltozasa [fok] - fuggetlen attol, hogy az auto
    eppen milyen szogben all a route-hoz (a |x14 - x0| ezt osszekeverte)."""
    d0, d1 = wp[:, 3] - wp[:, 0], wp[:, -1] - wp[:, -4]
    a = np.arctan2(d1[:, 1], d1[:, 0]) - np.arctan2(d0[:, 1], d0[:, 0])
    return np.degrees(np.abs((a + np.pi) % (2 * np.pi) - np.pi))


def _n_events(mask, seeds):
    """Hany kulonallo szakasz (egymas utani lepesek sorozata) van a maszkban."""
    starts = mask & ~np.r_[False, mask[:-1] & (seeds[1:] == seeds[:-1])]
    return int(starts.sum())


def analyze(args, out_dir):
    d = np.load(os.path.join(out_dir, "baseline_obs.npz"))
    obs = {k: d[k] for k in OBS_KEYS}
    seeds = d["seed"]
    n = len(seeds)
    model = load_model(args.model, args.device)

    base = predict_batched(model, obs)
    print(f"{n} baseline lepes, {len(set(seeds.tolist()))} route. "
          f"Offline vs. vezetes kozbeni akcio max elteres: {np.abs(base - d['action']).max():.2e}")

    turn_now = _window_turn_deg(obs["waypoints"])
    ahead = np.abs(d["turn_ahead"]) if "turn_ahead" in d.files else np.zeros(n)
    subsets = {
        "osszes": np.ones(n, bool),
        "egyenes (most is, 15-50 m-en is)": (turn_now < 1.5) & (ahead < 5) & (obs["maneuver"] == 0),
        "kanyar most (ablakban >= 3 fok)": turn_now >= 3,
        "kanyar jon (15-50 m, ablak meg egyenes)": (turn_now < 1.5) & (ahead >= 10),
        "keresztezodes": obs["maneuver"] != 0,
    }
    for s, m in subsets.items():
        print(f"  {s:42s} {int(m.sum()):6d} lepes, {_n_events(m, seeds):3d} szakasz")

    rng = np.random.default_rng(args.seed)
    perms = [rng.permutation(n) for _ in range(args.perms)]
    diffs = {}
    for name, parts in GROUPS:
        diffs[name] = np.mean([np.abs(predict_batched(model, permuted(obs, parts, p)) - base)
                               for p in perms], axis=0)                     # (n, 2)

    res = {}
    for s, m in subsets.items():
        if m.sum() < 20:
            continue
        u = sorted(set(seeds[m].tolist()))
        res[s] = {}
        print(f"\n=== PERMUTACIOS FONTOSSAG - {s} ({int(m.sum())} lepes, {len(u)} route) ===")
        print(f"{'csoport':32s} {'|d kormany| [90% CI seedekre]':>30s} {'% MINDEN':>9s} "
              f"{'|d gaz|':>8s} {'% MINDEN':>9s}")
        ref = diffs["MINDEN"][m].mean(0)
        for name, _ in GROUPS:
            per_seed = np.array([diffs[name][m & (seeds == sd)].mean(0) for sd in u])   # (seeds, 2)
            lo, hi = _boot_ci(per_seed[:, 0])
            v = diffs[name][m].mean(0)
            res[s][name] = dict(steer=float(v[0]), throttle=float(v[1]), steer_ci=[lo, hi])
            print(f"{name:32s} {v[0]:8.3f} [{lo:.3f}, {hi:.3f}]      {100 * v[0] / ref[0]:8.0f}% "
                  f"{v[1]:8.3f} {100 * v[1] / ref[1]:8.0f}%")
    print("\n  Elhagyott csoport: NAGY % = arra tamaszkodik. 'CSAK ... marad': KICSI % = az a csoport egymagaban eleg.")

    with open(os.path.join(out_dir, "analysis.json"), "w") as f:
        json.dump(dict(n=n, permutation=res), f, indent=2)
    if os.path.exists(os.path.join(out_dir, "episodes.jsonl")):
        summarize(out_dir)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["collect", "analyze", "summary"])
    p.add_argument("--model", required=True)
    p.add_argument("--out", default=None, help="alapertelmezes: <modell mappa>/ablation")
    p.add_argument("--device", default="cuda" if th.cuda.is_available() else "cpu")
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--max-steps", type=int, default=2400, help="20 fps mellett 2400 = 120 s")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--perms", type=int, default=5)
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    args = p.parse_args()

    out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(args.model)), "ablation")
    os.makedirs(out_dir, exist_ok=True)
    {"collect": collect, "analyze": analyze, "summary": lambda a, o: summarize(o)}[args.mode](args, out_dir)


if __name__ == "__main__":
    main()
