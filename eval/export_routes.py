"""A 10 scene utvonalanak kimentese az abrahoz.

Ugyanazzal a tervezovel szamolja ki az utvonalakat, mint az eval_scenes.py
(savvaltas-elek nelkul), igy a kirajzolt palya pontosan az, amit a modellek
vegigvezettek.

Eredmeny: results/routes.csv (group, scene, x, y)

Futtatas a repo gyokerebol:
  python eval/export_routes.py
"""
import csv
import json
import os
import sys
import time

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(EVAL_DIR))

import carla

from carla_env.navigation.planner import compute_route_waypoints
from carla_env.navigation.global_route_planner import GlobalRoutePlanner
from carla_env.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from config import CONFIG
from utils import launch_simulator

with open(os.path.join(EVAL_DIR, "scenes.json")) as f:
    cfg = json.load(f)

env_cfg = CONFIG["env"]
sim = None
if env_cfg["launch_sim"]:
    sim = launch_simulator(
        env_cfg["carla_root"],
        docker_image=env_cfg["carla_docker"] if env_cfg["is_carla_in_docker"] else None,
        container_name=env_cfg["carla_container"],
    )
    time.sleep(5.0)

try:
    client = carla.Client(env_cfg["host"], env_cfg["port"])
    client.set_timeout(60.0)
    world = client.load_world(cfg["town"])
    world_map = world.get_map()

    # A savvaltas-elek nelkuli tervezo, ugyanugy, mint az eval_scenes.py-ban:
    # kanyarban a belso savba atugrani mindig "rovidebb" lenne.
    grp = GlobalRoutePlanner(GlobalRoutePlannerDAO(world_map, 1.0))
    grp.setup()
    grp._graph.remove_edges_from([(u, v) for u, v, d in grp._graph.edges(data=True)
                                  if d["type"].name.startswith("CHANGELANE")])

    out = os.path.join(EVAL_DIR, "results", "routes.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "scene", "x", "y"])
        for group in ("highway", "city_to_highway"):
            for sc in cfg[group]:
                pts = [world_map.get_waypoint(carla.Location(*p)) for p in sc["points"]]
                route = []
                for a, b in zip(pts, pts[1:]):
                    seg = compute_route_waypoints(world_map, a, b, resolution=1.0, grp=grp)
                    route += [p for p in seg if not route
                              or p[0].transform.location.distance(
                                  route[-1][0].transform.location) > 0.5]
                for wp, _ in route:
                    loc = wp.transform.location
                    w.writerow([group, sc["name"], round(loc.x, 2), round(loc.y, 2)])
                print(f"[INFO] {sc['name']:<16} {len(route):5d} pont", flush=True)
    print(f"[INFO] Kesz: {out}")
finally:
    if sim is not None:
        sim.terminate()
        print("[INFO] Simulator closed")
