"""Modellek kiertekelese a scenes.json 10 rogzitett scene-jen (Town04).

A scene nem all meg az elso hibanal. Ha az auto lemegy az utrol, megall vagy
utkozik, feljegyezzuk, MI es HOL volt a baj, visszatesszuk az utolso erintett
route-waypointra, es megy tovabb - a CARLA Leaderboard logikaja szerint, ahol a
route vegigmegy es az infrakciok szamlalodnak. Igy minden modell ugyanannyi
lepest vezet, es a sebesseg / sav / elozes metrikak osszehasonlithatok (korabban
egy 5%-on megallo scene 23 s-bol adott atlagot egy 268 s-os mellett). 10
visszatevesnel a scene "Too many resets"-tel zarul.

scenes.json:
  town        a terkep
  params      min_avg_kmh: ennel lassabb atlaggal Timeout
              lane_center_tol_m: ennyin belul "a sav kozepen" [m]
              stuck_min_s: ennyi egyhuzamban ugyanazon auto mogott mar
                beragadas [s]
              stop_on_collision: mar nem hasznalt (utkozes utan is megy tovabb)
  traffic     a forgalmas scene-ek forgalma (mint a config ENV-ben)
  highway, city_to_highway   5-5 scene:
              name, length_m (a route hossza), traffic (van-e forgalom),
              seed (a forgalom elhelyezese), points: A, koztes pontok, B
              [x, y, z]. A route a pontok kozti savvaltas nelkuli
              legrovidebb utakbol all ossze.

Eredmeny: results/eval_<datum>.csv - egyetlen fajl a teljes futasra (a futtatott
modellek mind ebbe kerulnek). Minden sorban model (a .zip mappaja + a .zip neve,
pl. SAC_reward_1_model_final), group, scene, es a row oszlop mondja meg, mi a sor:
  row = scene     scene-enkent egy osszesito sor:
    reached_b         eljutott-e A-bol B-be (0/1)
    completion_pct    a route hany %-at tette meg
    time_s, avg_speed_kmh   mennyi ido alatt, milyen atlagsebesseggel
    overtakes         hany autot elozott meg: elotte volt a savjaban, mellette
                      elment, es mogeje kerult (barhogyan, szabalyossag nelkul)
    legal_overtakes   ebbol hany volt szabalyos: a reward_fn elozesi allapotgepe
                      szabalyosnak zarta a manovert (balra, szaggatott vonalon at,
                      szabad savba ki, elhaladt, szaggatotton vissza), es az auto
                      a jobbomon volt mellette - egy manoverben tobb auto is
    legal_overtakes_no_hit   a szabalyosak kozul, akit nem utott el
    other_passes      az overtakes-bol, ami nem szabalyos (jobbrol, zarovonalon,
                      visszasorolas nelkul vagy felbehagyott manoverben)
    overtake_attempts hany elozesi kiserlet volt: az allapotgep kihuzodast
                      eszlelt (szabad volt kiallni: szaggatott vonal, szabad sav)
    overtake_success  ebbol hany sikerult: szabalyosan zarult (elhaladt,
                      szaggatotton vissza), es kozben nem utott el autot
    overtake_success_pct   a sikeres kiserletek aranya [%] (egy kiserletben
                      tobb autot is megelozhet, ezert ez nem a legal_overtakes)
    pass_gap_avg_m, pass_gap_min_m   a megelozott autok mellett a ket
                      karosszeria kozti oldaltavolsag, atlag / legkisebb [m]
    hit_cars          hany kulonbozo autot utott el
    resets            hany visszatevesre volt szukseg
    off_track, stopped, too_fast   ebbol hany volt letero / megallo / 60 km/h feletti
    resets_per_km     a fo osszehasonlito szam: beavatkozas / km
    lane_center_pct   az ido hany %-aban volt lane_center_tol_m-en belul a
                      (legkozelebbi) sav kozepetol
    steer_jerk        atlagos |kormany valtozas| lepesenkent (rangatas)
    behind_s          mennyi idot toltott lead mogott (auto a savjaban 40 m-en
                      belul, es nem eloz eppen) [s]
    behind_can_s      ebbol mennyit, amikor elozhetett volna (balra szaggatott
                      vonal es szabad sav, mint a reward_fn "Can overtake"-je)
    behind_blocked_s  ebbol mennyit, amikor nem elozhetett ("Blocked")
    stuck, stuck_s    hany beragadas volt (legalabb stuck_min_s egyhuzamban
                      ugyanazon auto mogott), es ezek osszes ideje [s]
    total_reward      a scene alatt kapott reward osszege
    mean_reward       lepesenkenti atlag - a scene-ek kulonbozo hosszuak, ez
                      teszi a rewardot kozottuk osszevethetove
    reward_no_penalty a visszatevesek buntetese (-10 / -30) nelkul
    end               mi zarta a scene-t (Route done / Timeout / Too many resets)
  row = fail      minden visszateveshez egy sor: reason, hol (route_m,
                  completion_pct, x, y), mikor (t_s), es mi volt az allapot
                  (speed_kmh, offset_m, lane_idx, collision_with, front_id,
                  front_dist, left_free, steer, throttle, penalty).
  row = hit       minden elutott auto / targy: actor_id, actor_type, x, y,
                  route_m, t_s.
  row = overtake  minden megelozott / mellette elhaladt auto: actor_id,
                  actor_type, legal (szabalyos-e), hit (elutotte-e), ego_side
                  (melyik oldalan mentem el mellette: left = balrol), pass_gap_m,
                  es a legkozelebbi pillanatban (egymas mellett): az elozott auto
                  helye (x, y), route_m, t_s, other_offset_m (a sav kozepetol
                  mert tavolsaga) es other_in_lane (a teljes karosszeriaja a
                  savjaban volt-e, 0/1). illegal_reason, ha nem szabalyos:
                  right_side (jobbrol elozott), not_allowed (elozesi kiserleten
                  kivul: kiallt, amikor nem volt szabad, vagy a rossz savban
                  haladt), vagy annak a kiserletnek a kimenete, amelyikben
                  megelozte (aborted / solid_line_return / repeat / reset /
                  scene_end, lasd row = attempt), side_unknown (sikeres
                  kiserletben, de nem latszott, melyik oldalan ment el).
  row = attempt   minden elozesi kiserlet: hol es mikor allt ki (x, y,
                  route_m, t_s), actor_id (akit elozni akart), attempt_s
                  (meddig tartott), passed_cars (hany auto kerult kozben mogeje)
                  es attempt_outcome: success (szabalyos, nem utott el autot),
                  hit (szabalyos, de utott), aborted (a target melletti
                  elhaladas nelkul sorolt vissza), solid_line_return
                  (zarovonalon at sorolt vissza), repeat (mar megelozott autot
                  elozott ujra), reset (visszateves szakitotta felbe),
                  scene_end (a scene vegen meg tartott).
  row = stuck     minden beragadas: actor_id (a lead auto), hol kezdodott
                  (x, y, route_m, t_s), stuck_s (meddig tartott),
                  stuck_can_s (ebbol mennyi ideig elozhetett volna) es
                  stuck_reason, miert ragadt be (ami az ido nagyobb reszeben
                  igaz volt): could_overtake (szabad lett volna elozni, megsem
                  tette), vagy nem volt szabad: no_left_lane (nincs bal sav),
                  solid_line (zarovonal), left_lane_busy (foglalt a bal sav).

Futtatas a repo gyokerebol (a CARLA-t a config szerint inditja):
  python eval/eval_scenes.py                  # a configban beallitott reload_model
  python eval/eval_scenes.py --model tensorboard/SAC_reward_1/model_final.zip tensorboard/SAC_reward_5/model_final.zip
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
from carla_env.carla_env import CarlaEnv
from carla_env.navigation.planner import compute_route_waypoints
from carla_env.wrappers import get_actor_display_name
from config import CONFIG
from utils import (
    load_cam_ae, load_lidar_ae, create_encode_state_fn, create_observation_space, launch_simulator
)

# Egy CSV minden sortipusnak: az adott sorhoz nem tartozo oszlopok uresek.
COLS = ["model", "group", "scene", "row",
        # row = scene
        "length_m", "traffic", "cars", "end", "reached_b",
        "completion_pct", "time_s", "avg_speed_kmh", "overtakes", "legal_overtakes",
        "legal_overtakes_no_hit", "other_passes",
        "overtake_attempts", "overtake_success", "overtake_success_pct", "pass_gap_avg_m",
        "pass_gap_min_m", "hit_cars", "lane_center_pct", "steer_jerk",
        "behind_s", "behind_can_s", "behind_blocked_s", "stuck",
        "resets", "off_track", "stopped", "too_fast", "resets_per_km",
        "total_reward", "mean_reward", "reward_no_penalty",
        # row = fail / hit / overtake: hol es mikor
        "x", "y", "route_m", "t_s",
        # row = fail
        "reason", "speed_kmh", "offset_m", "lane_idx", "collision_with", "front_id",
        "front_dist", "left_free", "steer", "throttle", "penalty",
        # row = hit / overtake
        "actor_id", "actor_type",
        # row = overtake
        "legal", "hit", "ego_side", "pass_gap_m", "other_offset_m", "other_in_lane", "illegal_reason",
        # row = attempt
        "attempt_s", "attempt_outcome", "passed_cars",
        # row = stuck (es row = scene: stuck_s az osszes beragadas ideje)
        "stuck_s", "stuck_can_s", "stuck_reason"]


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

    # Egyetlen eredmenyfajl a teljes futasra (minden modell, minden scene).
    # Scene-enkent irjuk, igy Ctrl+C utan is megmarad, ami kesz. A futas
    # idobelyege a nevben, igy egy ujrafuttatas nem irja felul a regit.
    results_dir = os.path.join(EVAL_DIR, "results")
    results_csv = os.path.join(results_dir, f"eval_{time.strftime('%Y%m%d-%H%M%S')}.csv")

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
        env = CarlaEnv(
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
        # hanyadik route-meternel, hanyadik lepesben). Az ut ("Road") nem szamit.
        hits = []
        env._on_collision = lambda e: (
            hits.append((e.other_actor.id, e.other_actor.type_id, e.transform.location,
                         env.current_waypoint_index, env.step_count))
            if get_actor_display_name(e.other_actor) != "Road" else None)

        for model_path in args.model:
            # A tanitaskori lr_schedule closure-kent van picklezve, kiertekeleshez nem kell.
            model = SAC.load(model_path, device=device,
                             custom_objects={"learning_rate": 0.0, "lr_schedule": lambda _: 0.0})
            print("=" * 60)
            print(f"[INFO] Model: {model_path}")
            print("=" * 60)

            # A modell neve = a .zip-et tartalmazo mappa (pl. SAC_reward_1), a
            # fajlnev a .zip neve nelkul hozzafuzve, ha tobb checkpointot
            # futtatunk ugyanabbol a mappabol (model_final, model_500000_steps).
            model_name = (f"{os.path.basename(os.path.dirname(os.path.abspath(model_path)))}"
                          f"_{os.path.splitext(os.path.basename(model_path))[0]}")
            print(f"[INFO] Results: {results_csv}")

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
                pass_side = {}                # mellette a jobbomon volt-e (balrol mentem el mellette)
                pass_info = {}                # az elozott auto a legkozelebbi pillanatban (hely, sav)
                legal, man_front = set(), set()   # szabalyosan megelozott / a mostani manover alatt elottem
                prev_ot = 0                   # env.overtakes az elozo lepesben
                prev_man = man_reset = False  # elozott-e (allapotgep) az elozo lepesben / visszateves szakitotta-e felbe
                man, attempts, car_man = None, [], {}   # a mostani kiserlet / az osszes / auto -> a kiserlet kimenete
                behind_n = can_n = 0          # lepesek lead mogott / ebbol elozhetett volna
                stuck, cur = [], None         # beragadasok / a mostani lead mogotti szakasz
                lane_ok, steer_jerk = 0, 0.0
                prev_steer = env.vehicle.control.steer
                terminated = truncated = False
                steps = 0
                # Visszatevesek. A scene nem all meg az elso hibanal (CARLA
                # Leaderboard-stilus: a route vegigmegy, a hibak szamlalodnak),
                # igy a sebesseg / sav / elozes metrikak ugyanannyi lepesbol
                # jonnek minden modellnel es osszehasonlithatok.
                fails = []
                n_off_track = n_stopped = n_too_fast = 0
                while not truncated and steps < max_steps and len(fails) < 10:
                    # Ugyanazok a dtype-ok, mint tanitaskor (a DummyVecEnv a space dtype-jara konvertal).
                    obs = {k: np.asarray(v, dtype=np.int64 if k == "maneuver" else np.float32)
                           for k, v in obs.items()}
                    action, _ = model.predict(obs, deterministic=True)
                    if env.activate_render:
                        env.extra_info.append(f"Scene {i + 1}/{len(scenes)}: {sc['name']}")
                    obs, reward, terminated, truncated, _ = env.step(action)
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
                        # Csak ami tenyleg mellettem van: legfeljebb ket savnyira
                        # (10 m), es nem egy masik szinten (hid / alatta futo ut).
                        if abs(lon) < ego_ext.x + ext.x and abs(lat) < 10.0 and abs(o.z - loc.z) < 3.0:
                            # Melyik oldalamon van, akkor is, ha ket savval arrebb (elozes +2 savba).
                            pass_side[other.id] = lat > 0.0
                            if abs(lat) < 6.0:
                                pass_gap[other.id] = min(pass_gap.get(other.id, np.inf),
                                                         abs(lat) - ego_ext.y - ext.y)
                            # Az elozott auto a legkozelebbi pillanatban: hol van, es a
                            # sajat savjaban van-e (a teljes karosszeriaja a sav
                            # hatarain belul) - vagy kitert / meglokte az ego.
                            if abs(lon) < pass_info.get(other.id, {}).get("lon", np.inf):
                                ow = env.map.get_waypoint(o)
                                d = np.hypot(o.x - ow.transform.location.x, o.y - ow.transform.location.y)
                                pass_info[other.id] = dict(
                                    lon=abs(lon), x=round(o.x, 1), y=round(o.y, 1),
                                    route_m=env.current_waypoint_index, t_s=round(steps / env.fps, 1),
                                    actor_type=other.type_id, other_offset_m=round(d, 2),
                                    other_in_lane=int(d + ext.y < ow.lane_width / 2.0))
                    passed |= {a_id for a_id in leads if surr["lon"].get(a_id, 0.0) < -5.0}

                    # Szabalyos elozes. A reward_fn allapotgepe zarja le az elozest
                    # szabalyosnak (env.overtakes no): balra, szaggatott vonalon at
                    # szabad savba allt ki, elhaladt a target mellett, es szaggatott
                    # vonalon sorolt vissza. Ilyenkor minden auto szabalyosan
                    # megelozott, ami a manover alatt elottem volt, most 5 m-rel
                    # mogottem van, es mellette a jobbomon volt - igy a manover
                    # alatt megelozott tobbi auto is szamit, nem csak a target.
                    if rewards._overtaking:
                        man_front |= {a_id for a_id, lon in surr["lon"].items() if 0.0 < lon < 40.0}
                    # Elozesi kiserlet: az allapotgep kihuzodast eszlelt (szabad
                    # volt kiallni: szaggatott vonal, szabad sav).
                    if rewards._overtaking and not prev_man:
                        man = dict(t_s=round(steps / env.fps, 1), x=round(loc.x, 1), y=round(loc.y, 1),
                                   route_m=env.current_waypoint_index, actor_id=rewards._target_id,
                                   step=steps, hits=len(hits))
                    if env.overtakes > prev_ot:
                        legal |= {a_id for a_id in man_front
                                  if surr["lon"].get(a_id, 0.0) < -5.0 and pass_side.get(a_id, False)}
                        legal.add(rewards._target_id)   # akit az allapotgep elozott, az biztosan
                    # A kiserlet vege, es mi lett belole: success (szabalyosan
                    # zarult, nem utott el autot), hit (szabalyosan zarult, de
                    # utott), aborted (a target melletti elhaladas nelkul sorolt
                    # vissza), solid_line_return (zarovonalon at sorolt vissza),
                    # repeat (mar korabban megelozott autot elozott ujra), reset
                    # (visszateves szakitotta felbe). A kozben mogem kerult autok
                    # ennek a kiserletnek a kimenetet kapjak (illegal_reason).
                    if prev_man and not rewards._overtaking:
                        if man_reset:
                            outcome = "reset"
                        elif env.overtakes > prev_ot:
                            outcome = ("hit" if any(h[1].startswith("vehicle.") for h in hits[man["hits"]:])
                                       else "success")
                        elif not rewards._passed:
                            outcome = "aborted"
                        elif not rewards._return_ok:
                            outcome = "solid_line_return"
                        else:
                            outcome = "repeat"
                        man_passed = {a_id for a_id in man_front if surr["lon"].get(a_id, 0.0) < -5.0}
                        for a_id in man_passed:
                            car_man.setdefault(a_id, outcome)
                        attempts.append(dict(man, outcome=outcome, passed_cars=len(man_passed),
                                             attempt_s=round((steps - man["step"]) / env.fps, 1)))
                        man_reset = False
                    if not rewards._overtaking:
                        man_front = set()
                    prev_ot = env.overtakes
                    prev_man = rewards._overtaking

                    # Lead mogott (a reward_fn szerint: auto a savomban 40 m-en
                    # belul), es nem eloz eppen. Elozhetne, ha balra (+1, vagy
                    # +2 a +1-en at) szaggatott a vonal es szabad a sav - mint a
                    # reward_fn "Can overtake"-je, kulonben "Blocked".
                    lanes = surr["lanes"]
                    behind = surr["front_id"] is not None and not rewards._overtaking
                    can_ot = behind and any(lanes.get(n) and lanes[n]["broken"] and lanes[n]["free"]
                                            and (n == 1 or lanes[1]["cross"]) for n in (1, 2))
                    behind_n += behind
                    can_n += can_ot
                    # Beragadas: legalabb stuck_min_s egyhuzamban ugyanazon auto
                    # mogott. A szakasz veget er, ha mar nincs mogotte (elozni
                    # kezd, vagy a lead elmegy) vagy masik auto lesz a lead.
                    if cur is not None and (not behind or cur["actor_id"] != surr["front_id"]):
                        if cur["n"] >= params["stuck_min_s"] * env.fps:
                            stuck.append(cur)
                        cur = None
                    if behind:
                        if cur is None:
                            cur = dict(actor_id=surr["front_id"], x=round(loc.x, 1), y=round(loc.y, 1),
                                       route_m=env.current_waypoint_index, t_s=round(steps / env.fps, 1),
                                       n=0, can=0, why={})
                        cur["n"] += 1
                        cur["can"] += can_ot
                        # Miert van mogotte: elozhetett volna, vagy nem volt szabad
                        # elozni (nincs bal sav / zarovonal / foglalt a bal sav).
                        why = ("could_overtake" if can_ot else "no_left_lane" if 1 not in lanes
                               else "solid_line" if not lanes[1]["broken"] else "left_lane_busy")
                        cur["why"][why] = cur["why"].get(why, 0) + 1

                    # --- Visszatevés -------------------------------------
                    # Ha az env terminalt (Off-track / Vehicle stopped /
                    # Collision / Too fast), feljegyezzuk, MI es HOL volt a
                    # baj, majd visszatesszuk az autot az utolso erintett
                    # route-waypointra, es a scene megy tovabb.
                    if terminated:
                        reason = env.terminal_reason or "Terminated"
                        wp = env.route_waypoints[env.current_waypoint_index][0]
                        # A route savjanak kozepetol elojeles oldaltavolsag
                        # (+ balra), es hogy hanyadik savban van - ebbol latszik,
                        # hogy elozes kozben vagy a sajat savjaban szallt el.
                        wtr = wp.transform
                        wr = wtr.get_right_vector()
                        side = ((loc.x - wtr.location.x) * wr.x
                                + (loc.y - wtr.location.y) * wr.y)
                        fails.append(dict(
                            model=model_name, group=group, scene=sc["name"], row="fail",
                            reason=reason, route_m=env.current_waypoint_index,
                            completion_pct=round(100.0 * (env.current_waypoint_index + 1) / len(route), 1),
                            t_s=round(steps / env.fps, 1),
                            x=round(loc.x, 1), y=round(loc.y, 1),
                            speed_kmh=round(env.vehicle.get_speed(), 1),
                            offset_m=round(-side, 2),
                            lane_idx=int(round(-side / surr["lane_width"])),
                            collision_with=env.collision_with or "",
                            front_id=surr["front_id"] if surr["front_id"] is not None else "",
                            front_dist=round(surr["front_dist"], 1) if surr["front_id"] is not None else "",
                            left_free=int(surr["left_free"]),
                            steer=round(env.vehicle.control.steer, 3),
                            throttle=round(env.vehicle.control.throttle, 3),
                            # A terminal lepes rewardja maga a buntetes: a
                            # reward_fn ilyenkor azonnal kilep (-10, jarmu-
                            # utkozesnel -30, reward 5-nel a PBRS -Phi-vel).
                            penalty=round(float(reward), 2),
                        ))
                        n_off_track += reason == "Off-track"
                        n_stopped += reason == "Vehicle stopped"
                        n_too_fast += reason == "Too fast"
                        print(f"    !! {reason:<22} @ {env.current_waypoint_index:5d} m "
                              f"({fails[-1]['completion_pct']:5.1f}%) "
                              f"x={loc.x:7.1f} y={loc.y:7.1f} "
                              f"v={fails[-1]['speed_kmh']:5.1f} km/h "
                              f"offset={fails[-1]['offset_m']:+5.2f} m "
                              f"lane={fails[-1]['lane_idx']:+d} "
                              f"lead={fails[-1]['front_dist'] or '-'} "
                              f"steer={fails[-1]['steer']:+.3f} "
                              f"-> visszateve ({len(fails)}/10)", flush=True)

                        # Teleport a route-ra, a route iranyaba nezve. A fizika
                        # kikapcsolasa kell, kulonben a motor megtartja a
                        # sebesseget/impulzust (ugyanaz, mint a new_route()-ban).
                        env.vehicle.control.steer = 0.0
                        env.vehicle.control.throttle = 0.0
                        env.vehicle.control.brake = 0.0
                        env.vehicle.set_simulate_physics(False)
                        env.vehicle.set_transform(carla.Transform(
                            wtr.location + carla.Location(z=0.5), wtr.rotation))
                        env.vehicle.set_simulate_physics(True)
                        # A teleport ne szamitson bele a megtett utba (ebbol jon
                        # az avg_speed_kmh), es az allo auto ne inditsa ujra a
                        # "Vehicle stopped" 5 s-os timert a kovetkezo lepesben.
                        env.previous_location = wtr.location
                        env.terminal_state = False
                        env.collision_with = None
                        env.terminal_reason = ""
                        rewards._low_speed_timer = 0.0
                        # A felbehagyott elozes ne zarulhasson le szabalyosnak
                        # attol, hogy a route savjanak kozepere tettuk. A reward 5
                        # terminal lepese mar levonta a potencialt (-Phi), ne
                        # vonja le meg egyszer.
                        rewards._overtaking = False
                        rewards._prev_phi = 0.0
                        man_reset = prev_man          # a kovetkezo lepes "reset"-kent zarja a kiserletet
                        env.world.tick()

                # A scene vegen meg tarto elozesi kiserlet.
                if prev_man:
                    man_passed = {a_id for a_id in man_front if surr["lon"].get(a_id, 0.0) < -5.0}
                    outcome = "reset" if man_reset else "scene_end"
                    for a_id in man_passed:
                        car_man.setdefault(a_id, outcome)
                    attempts.append(dict(man, outcome=outcome, passed_cars=len(man_passed),
                                         attempt_s=round((steps - man["step"]) / env.fps, 1)))

                # A scene vegen meg tarto lead mogotti szakasz.
                if cur is not None and cur["n"] >= params["stuck_min_s"] * env.fps:
                    stuck.append(cur)

                if truncated:
                    end = "Route done"
                elif len(fails) >= 10:
                    end = "Too many resets"
                else:
                    end = "Timeout"

                # Egy auto / targy egyszer szamit (az utkozes tobb eventet is ad): az elso erintes.
                first_hit = {}
                for actor_id, type_id, hit_loc, route_m, step in list(hits):
                    first_hit.setdefault(actor_id, (type_id, hit_loc, route_m, step))
                gaps = [pass_gap[a_id] for a_id in passed if a_id in pass_gap]
                gap_ids = {a_id for a_id in passed if a_id in pass_gap}
                time_s = steps / env.fps

                row = dict(
                    model=model_name, group=group, scene=sc["name"], row="scene", length_m=len(route),
                    traffic=int(sc["traffic"]), cars=len(env.traffic_ids), end=end,
                    reached_b=int(truncated),
                    completion_pct=round(100.0 * (env.current_waypoint_index + 1) / len(route), 1),
                    time_s=round(time_s, 1),
                    avg_speed_kmh=round(3.6 * env.distance_traveled / time_s, 1),
                    overtakes=len(gaps),
                    legal_overtakes=len(legal),
                    legal_overtakes_no_hit=len(legal - set(first_hit)),
                    other_passes=len(gap_ids - legal),
                    overtake_attempts=len(attempts),
                    overtake_success=sum(a["outcome"] == "success" for a in attempts),
                    overtake_success_pct=(round(100.0 * sum(a["outcome"] == "success" for a in attempts)
                                                / len(attempts), 1) if attempts else ""),
                    pass_gap_avg_m=round(float(np.mean(gaps)), 2) if gaps else "",
                    pass_gap_min_m=round(float(np.min(gaps)), 2) if gaps else "",
                    hit_cars=sum(h[0].startswith("vehicle.") for h in first_hit.values()),
                    lane_center_pct=round(100.0 * lane_ok / steps, 1),
                    steer_jerk=round(steer_jerk / steps, 4),
                    behind_s=round(behind_n / env.fps, 1),
                    behind_can_s=round(can_n / env.fps, 1),
                    behind_blocked_s=round((behind_n - can_n) / env.fps, 1),
                    stuck=len(stuck),
                    stuck_s=round(sum(s["n"] for s in stuck) / env.fps, 1),
                    resets=len(fails), off_track=n_off_track, stopped=n_stopped, too_fast=n_too_fast,
                    # A fo osszehasonlito szam: hany beavatkozas kellett 1 km-en.
                    resets_per_km=round(1000.0 * len(fails) / max(env.distance_traveled, 1.0), 2),
                    total_reward=round(env.total_reward, 1),
                    # Lepesenkenti atlag: a scene-ek kulonbozo hosszuak, a
                    # total_reward-ot ez teszi osszevethetove kozottuk.
                    mean_reward=round(env.total_reward / steps, 3),
                    # A visszatevesek buntetese nelkul: igy a vezetes minosege
                    # es a hibak szama (resets) kulon szam, nem egy osszegben.
                    reward_no_penalty=round(env.total_reward - sum(f["penalty"] for f in fails), 1),
                )
                # A mappat / fejlecet iraskor hozzuk letre: ha futas kozben eltunik, ujra lesz.
                os.makedirs(results_dir, exist_ok=True)
                new_file = not os.path.exists(results_csv)
                with open(results_csv, "a", newline="") as f:
                    w = csv.DictWriter(f, COLS, restval="")
                    if new_file:
                        w.writeheader()
                    w.writerow(row)
                    w.writerows(fails)
                    for actor_id, (type_id, hit_loc, route_m, step) in first_hit.items():
                        w.writerow(dict(model=model_name, group=group, scene=sc["name"], row="hit",
                                        actor_id=actor_id, actor_type=type_id,
                                        x=round(hit_loc.x, 1), y=round(hit_loc.y, 1),
                                        route_m=route_m, t_s=round(step / env.fps, 1)))
                    for a_id in sorted(gap_ids | legal):
                        info = {k: v for k, v in pass_info.get(a_id, {}).items() if k != "lon"}
                        # Miert nem szabalyos: jobbrol elozott, kiserleten kivul
                        # (not_allowed), vagy annak a kiserletnek a kimenete,
                        # amelyikben mogem kerult. side_unknown: sikeres
                        # kiserletben, de nem lattuk, melyik oldalan ment el.
                        why = car_man.get(a_id, "not_allowed")
                        w.writerow(dict(model=model_name, group=group, scene=sc["name"], row="overtake",
                                        actor_id=a_id, legal=int(a_id in legal), hit=int(a_id in first_hit),
                                        ego_side={True: "left", False: "right"}.get(pass_side.get(a_id), ""),
                                        pass_gap_m=round(pass_gap[a_id], 2) if a_id in pass_gap else "",
                                        illegal_reason=("" if a_id in legal else
                                                        "right_side" if pass_side.get(a_id) is False else
                                                        "side_unknown" if why in ("success", "hit") else why),
                                        **info))
                    for a in attempts:
                        w.writerow(dict(model=model_name, group=group, scene=sc["name"], row="attempt",
                                        actor_id=a["actor_id"], x=a["x"], y=a["y"],
                                        route_m=a["route_m"], t_s=a["t_s"], attempt_s=a["attempt_s"],
                                        attempt_outcome=a["outcome"], passed_cars=a["passed_cars"]))
                    for s in stuck:
                        w.writerow(dict(model=model_name, group=group, scene=sc["name"], row="stuck",
                                        actor_id=s["actor_id"], x=s["x"], y=s["y"],
                                        route_m=s["route_m"], t_s=s["t_s"],
                                        stuck_s=round(s["n"] / env.fps, 1),
                                        stuck_can_s=round(s["can"] / env.fps, 1),
                                        # Ami a beragadas idejenek legnagyobb reszeben igaz volt.
                                        stuck_reason=max(s["why"], key=s["why"].get)))

                print(f"[{sc['name']:<16}] {end:<15} | {row['completion_pct']:5.1f}% "
                      f"{row['time_s']:6.1f} s {row['avg_speed_kmh']:4.1f} km/h | "
                      f"overtakes {row['overtakes']} (legal {row['legal_overtakes']}, "
                      f"legal no hit {row['legal_overtakes_no_hit']}, other {row['other_passes']}; "
                      f"attempts {row['overtake_attempts']}, success {row['overtake_success']} "
                      f"({row['overtake_success_pct'] or '-'}%); "
                      f"gap avg {row['pass_gap_avg_m'] or '-'} m, "
                      f"min {row['pass_gap_min_m'] or '-'} m) | hit cars {row['hit_cars']} | "
                      f"resets {row['resets']} (off {n_off_track}, stop {n_stopped}, fast {n_too_fast}, "
                      f"{row['resets_per_km']}/km) | "
                      f"lane center {row['lane_center_pct']:.1f}% | steer jerk {row['steer_jerk']:.4f} | "
                      f"behind {row['behind_s']} s (can {row['behind_can_s']}, blocked {row['behind_blocked_s']}), "
                      f"stuck {row['stuck']} ({row['stuck_s']} s) | "
                      f"reward {row['total_reward']:.0f} "
                      f"(mean {row['mean_reward']:.2f}, no pen {row['reward_no_penalty']:.0f})",
                      flush=True)

    except KeyboardInterrupt:
        print("[INFO] Evaluation interrupted")
    finally:
        print(f"[INFO] Results: {results_csv}")
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
