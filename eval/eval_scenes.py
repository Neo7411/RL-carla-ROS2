"""Modellek kiertekelese a scenes.json 10 rogzitett scene-jen (Town04).

scenes.json:
  town        a terkep
  params      min_avg_kmh: ennel lassabb atlaggal Timeout
              lane_center_tol_m: ennyin belul "a sav kozepen" [m]
              stop_on_collision: false = utkozes utan is megy tovabb (igy
                szamolhato, hany autot utott el), true = az elso utkozes a vege
  traffic     a forgalmas scene-ek forgalma (mint a config ENV-ben)
  highway, city_to_highway   5-5 scene:
              name, length_m (a route hossza), traffic (van-e forgalom),
              seed (a forgalom elhelyezese), points: A, koztes pontok, B
              [x, y, z]. A route a pontok kozti savvaltas nelkuli
              legrovidebb utakbol all ossze.

Eredmeny, results/<datum>_scenes.csv, scene-enkent egy sor:
  reached_b         eljutott-e A-bol B-be (0/1)
  completion_pct    a route hany %-at tette meg
  time_s, avg_speed_kmh   mennyi ido alatt, milyen atlagsebesseggel
  overtakes         hany autot elozott meg: elotte volt a savjaban, mellette
                    elment, es mogeje kerult
  pass_gap_avg_m, pass_gap_min_m   a megelozott autok mellett a ket
                    karosszeria kozti oldaltavolsag, atlag / legkisebb [m]
  hit_cars          hany kulonbozo autot utott el
  lane_center_pct   az ido hany %-aban volt lane_center_tol_m-en belul a
                    (legkozelebbi) sav kozepetol
  steer_jerk        atlagos |kormany valtozas| lepesenkent (rangatas)
  end               mi zarta a scene-t (Route done / Timeout / Off-track / ...)
results/<datum>_hits.csv: minden elutott auto / targy, es hol (x, y, route_m).

Futtatas a repo gyokerebol (a CARLA-t a config szerint inditja):
  python eval/eval_scenes.py                  # a configban beallitott reload_model
  python eval/eval_scenes.py --model tensorboard/SAC_4/model_interrupted.zip tensorboard/SAC_3/model_interrupted.zip
"""
import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(EVAL_DIR))   # a repo gyokere (config, carla_env, utils)

import carla
import numpy as np
import torch
from stable_baselines3 import SAC

import carla_env.reward as rewards
from carla_env.envs.carla_route_env import CarlaRouteEnv
from carla_env.navigation.planner import compute_route_waypoints
from carla_env.wrappers import get_actor_display_name
from config import CONFIG
from utils import (
    load_cam_ae, load_lidar_ae, create_encode_state_fn, create_observation_space, launch_simulator
)

SCENE_COLS = ["model", "group", "scene", "length_m", "traffic", "cars", "end", "reached_b",
              "completion_pct", "time_s", "avg_speed_kmh", "overtakes", "pass_gap_avg_m",
              "pass_gap_min_m", "hit_cars", "lane_center_pct", "steer_jerk"]
HIT_COLS = ["model", "scene", "hit", "x", "y", "route_m"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", nargs="+",
                        default=[os.path.join(CONFIG["train"]["reload_model_path"],
                                              CONFIG["train"]["reload_model_file"])],
                        help="egy vagy tobb SAC .zip - mindegyik ugyanazt a 10 scene-t vezeti")
    parser.add_argument("--scenes", default=os.path.join(EVAL_DIR, "scenes.json"))
    args = parser.parse_args()

    with open(args.scenes) as f:
        cfg = json.load(f)
    params, traffic = cfg["params"], cfg["traffic"]
    scenes = [(group, sc) for group in ("highway", "city_to_highway") for sc in cfg[group]]

    # Eredmenyfajlok: scene-enkent irjuk, igy Ctrl+C utan is megmarad, ami kesz.
    os.makedirs(os.path.join(EVAL_DIR, "results"), exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    scenes_csv = os.path.join(EVAL_DIR, "results", f"{stamp}_scenes.csv")
    hits_csv = os.path.join(EVAL_DIR, "results", f"{stamp}_hits.csv")
    with open(scenes_csv, "w", newline="") as f:
        csv.writer(f).writerow(SCENE_COLS)
    with open(hits_csv, "w", newline="") as f:
        csv.writer(f).writerow(HIT_COLS)

    env_cfg = CONFIG["env"]
    sim_proc = None
    if env_cfg["launch_sim"]:
        sim_proc = launch_simulator(
            env_cfg["carla_root"],
            docker_image=env_cfg["carla_docker"] if env_cfg["is_carla_in_docker"] else None,
            container_name=env_cfg["carla_container"],
        )
        print("[INFO] Simulator launched...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cam_ae = load_cam_ae(CONFIG["camera"], device)
    lidar_ae, lidar_latent_shape = load_lidar_ae(CONFIG["lidar"], device)
    observation_space = create_observation_space(cam_ae, lidar_latent_shape)
    encode_state_fn = create_encode_state_fn(cam_ae, lidar_ae, CONFIG, device)

    try:
        env = CarlaRouteEnv(
            obs_res=env_cfg["obs_res"],
            viewer_res=env_cfg["viewer_res"],
            host=env_cfg["host"],
            port=env_cfg["port"],
            town=cfg["town"],
            max_route_length=env_cfg["max_route_length"],
            min_route_length=env_cfg["min_route_length"],
            reward_fn=getattr(rewards, env_cfg["reward_fn"]),
            observation_space=observation_space,
            encode_state_fn=encode_state_fn,
            fps=env_cfg["fps"],
            action_smoothing=env_cfg["action_smoothing"],
            action_space_type=env_cfg["action_space_type"],
            activate_spectator=env_cfg["activate_spectator"],
            activate_render=env_cfg["activate_render"],
            activate_lidar=True,
            traffic_vehicles=traffic["max_vehicles"],
            traffic_start_m=traffic["start_m"],
            traffic_gap_m=tuple(traffic["gap_m"]),
            traffic_speed_kmh=tuple(traffic["speed_kmh"]),
            hybrid_radius=env_cfg["hybrid_radius"],
        )

        # Savvaltas nelkuli route-ok. A planner savvaltas-elei 0 m hosszuak,
        # igy kanyarban a belso savba atugrani mindig "rovidebb" lenne (a
        # tanitas is eldobja a savvaltasos route-okat). Csak ebben a folyamatban hat.
        env._grp._graph.remove_edges_from([(u, v) for u, v, d in env._grp._graph.edges(data=True)
                                           if d["type"].name.startswith("CHANGELANE")])

        # Utkozes: az env-e lezarna az epizodot, ez csak feljegyzi (ki, hol,
        # hanyadik route-meternel). Az ut ("Road") nem szamit.
        hits = []
        env._on_collision = lambda e: (
            hits.append((e.other_actor.id, e.other_actor.type_id, e.transform.location, env.current_waypoint_index))
            if get_actor_display_name(e.other_actor) != "Road" else None)

        for model_path in args.model:
            # A tanitaskori lr_schedule closure-kent van picklezve, kiertekeleshez nem kell.
            model = SAC.load(model_path, device=device,
                             custom_objects={"learning_rate": 0.0, "lr_schedule": lambda _: 0.0})
            print("=" * 60)
            print(f"[INFO] Model: {model_path}")
            print("=" * 60)

            for i, (group, sc) in enumerate(scenes):
                # A route: a pontok kozti szakaszok egymas utan, a szakaszhataron
                # a dupla pontot kihagyjuk.
                pts = [env.map.get_waypoint(carla.Location(*p)) for p in sc["points"]]
                route = []
                for a, b in zip(pts, pts[1:]):
                    seg = compute_route_waypoints(env.map, a, b, resolution=1.0, grp=env._grp)
                    route += [p for p in seg if not route
                              or p[0].transform.location.distance(route[-1][0].transform.location) > 0.5]
                # A reset() ezt a route-ot kapja a veletlen helyett.
                env._sample_route = lambda: (pts[0], pts[-1], route)

                env.traffic_vehicles = traffic["max_vehicles"] if sc["traffic"] else 0
                # Ugyanaz a seed -> ugyanaz a forgalom minden modellnel.
                np.random.seed(sc["seed"])
                if env.tm is not None:
                    env.tm.set_random_device_seed(sc["seed"])
                obs, _ = env.reset()
                hits.clear()

                max_steps = int(len(route) / (params["min_avg_kmh"] / 3.6) * env.fps)
                ego_ext = env.vehicle.bounding_box.extent
                leads, pass_gap = set(), {}   # valaha elottem a savomban / oldaltavolsag mellette
                passed = set()                # a leadek kozul, akik mar mogem kerultek
                lane_ok, steer_jerk = 0, 0.0
                prev_steer = env.vehicle.control.steer
                terminated = truncated = False
                steps = 0
                while not (terminated or truncated) and steps < max_steps:
                    # Ugyanazok a dtype-ok, mint tanitaskor (a DummyVecEnv a space dtype-jara konvertal).
                    obs = {k: np.asarray(v, dtype=np.int64 if k == "maneuver" else np.float32)
                           for k, v in obs.items()}
                    action, _ = model.predict(obs, deterministic=True)
                    if env.activate_render:
                        env.extra_info.append(f"Scene {i + 1}/{len(scenes)}: {sc['name']}")
                    obs, _, terminated, truncated, _ = env.step(action)
                    steps += 1

                    # Sav kozepe: a legkozelebbi sav kozepvonalatol (elozes kozben a masik savetol).
                    tr = env.vehicle.get_transform()
                    loc = tr.location
                    center = env.map.get_waypoint(loc).transform.location
                    lane_ok += np.hypot(loc.x - center.x, loc.y - center.y) < params["lane_center_tol_m"]

                    steer = env.vehicle.control.steer
                    steer_jerk += abs(steer - prev_steer)
                    prev_steer = steer

                    # Elozes. Mellette: a ket auto hosszban fedi egymast, es a
                    # szomszed savban van (6 m-en belul) - ekkor merjuk a ket
                    # karosszeria kozti oldaltavolsagot.
                    surr = env.surroundings
                    if surr["front_id"] is not None:
                        leads.add(surr["front_id"])
                    fwd, right = tr.get_forward_vector(), tr.get_right_vector()
                    near_ids = [a_id for a_id, lon in surr["lon"].items() if abs(lon) < 10.0]
                    for other in (env.world.get_actors(near_ids) if near_ids else []):
                        o, ext = other.get_location(), other.bounding_box.extent
                        lon = (o.x - loc.x) * fwd.x + (o.y - loc.y) * fwd.y
                        lat = (o.x - loc.x) * right.x + (o.y - loc.y) * right.y
                        if abs(lon) < ego_ext.x + ext.x and abs(lat) < 6.0:
                            pass_gap[other.id] = min(pass_gap.get(other.id, np.inf),
                                                     abs(lat) - ego_ext.y - ext.y)
                    passed |= {a_id for a_id in leads if surr["lon"].get(a_id, 0.0) < -5.0}

                    if params["stop_on_collision"] and hits:
                        break

                if params["stop_on_collision"] and hits:
                    end = "Collision"
                elif terminated:
                    end = env.terminal_reason or "Terminated"
                else:
                    end = "Route done" if truncated else "Timeout"

                # Egy auto / targy egyszer szamit (az utkozes tobb eventet is ad): az elso erintes.
                first_hit = {}
                for actor_id, type_id, hit_loc, route_m in list(hits):
                    first_hit.setdefault(actor_id, (type_id, hit_loc, route_m))
                gaps = [pass_gap[a_id] for a_id in passed if a_id in pass_gap]
                time_s = steps / env.fps

                row = dict(
                    model=model_path, group=group, scene=sc["name"], length_m=len(route),
                    traffic=int(sc["traffic"]), cars=len(env.traffic_ids), end=end,
                    reached_b=int(truncated and not terminated),
                    completion_pct=round(100.0 * (env.current_waypoint_index + 1) / len(route), 1),
                    time_s=round(time_s, 1),
                    avg_speed_kmh=round(3.6 * env.distance_traveled / time_s, 1),
                    overtakes=len(gaps),
                    pass_gap_avg_m=round(float(np.mean(gaps)), 2) if gaps else "",
                    pass_gap_min_m=round(float(np.min(gaps)), 2) if gaps else "",
                    hit_cars=sum(t.startswith("vehicle.") for t, _, _ in first_hit.values()),
                    lane_center_pct=round(100.0 * lane_ok / steps, 1),
                    steer_jerk=round(steer_jerk / steps, 4),
                )
                with open(scenes_csv, "a", newline="") as f:
                    csv.DictWriter(f, SCENE_COLS).writerow(row)
                with open(hits_csv, "a", newline="") as f:
                    for type_id, hit_loc, route_m in first_hit.values():
                        csv.writer(f).writerow([model_path, sc["name"], type_id,
                                                round(hit_loc.x, 1), round(hit_loc.y, 1), route_m])

                print(f"[{sc['name']:<16}] {end:<14} | {row['completion_pct']:5.1f}% "
                      f"{row['time_s']:6.1f} s {row['avg_speed_kmh']:4.1f} km/h | "
                      f"overtakes {row['overtakes']} (gap avg {row['pass_gap_avg_m'] or '-'} m, "
                      f"min {row['pass_gap_min_m'] or '-'} m) | hit cars {row['hit_cars']} | "
                      f"lane center {row['lane_center_pct']:.1f}% | steer jerk {row['steer_jerk']:.4f}",
                      flush=True)

    except KeyboardInterrupt:
        print("[INFO] Evaluation interrupted")
    finally:
        print(f"[INFO] Results: {scenes_csv}")
        # Mint a train_rl.py-ban: kilepeskor leallitjuk a szimulatort.
        if sim_proc is not None:
            if env_cfg["is_carla_in_docker"]:
                subprocess.run(["docker", "rm", "-f", env_cfg["carla_container"]],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                os.killpg(os.getpgid(sim_proc.pid), signal.SIGKILL)
            print(f"[INFO] Simulator closed")


if __name__ == '__main__':
    main()
