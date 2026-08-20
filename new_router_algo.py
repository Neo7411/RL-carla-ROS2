import carla, random, itertools, time
from carla_env.navigation.planner import compute_route_waypoints
from carla_env.navigation.local_planner import RoadOption

client = carla.Client("127.0.0.1", 2000)
client.set_timeout(60.0)
world = client.load_world("Town04")
carla_map = world.get_map()
sps = carla_map.get_spawn_points()

# async mód, különben nem tickel a szerver és nem látszik a rajz
s = world.get_settings()
s.synchronous_mode = False
s.fixed_delta_seconds = None
world.apply_settings(s)

MIN_LEN, MAX_LEN, MIN_TURNS, N_WANTED = 600, 10000, 5, 12


def count_turns(route):
    n, prev = 0, RoadOption.VOID
    for _, o in route:
        if o in (RoadOption.LEFT, RoadOption.RIGHT) and o != prev:
            n += 1
        prev = o
    return n


pairs = list(itertools.permutations(range(len(sps)), 2))
random.shuffle(pairs)

routes, drawn = [], []
for a, b in pairs:
    route = compute_route_waypoints(carla_map,
                                    carla_map.get_waypoint(sps[a].location),
                                    carla_map.get_waypoint(sps[b].location),
                                    resolution=1.0)
    if not (MIN_LEN <= len(route) <= MAX_LEN):
        continue
    if any(o.name.startswith("CHANGELANE") for _, o in route):
        continue
    t = count_turns(route)
    if t < MIN_TURNS:
        continue
    routes.append((a, b))
    drawn.append((a, b, route))
    print(f"({a}, {b})\thossz={len(route)}m\tkanyar={t}")
    if len(routes) >= N_WANTED:
        break

print("\nintersection_routes = itertools.cycle(" + str(routes) + ")")

Z = carla.Location(z=1.0)
PALETTE = [carla.Color(255, 0, 0), carla.Color(0, 255, 0), carla.Color(0, 128, 255),
           carla.Color(255, 255, 0), carla.Color(255, 0, 255), carla.Color(0, 255, 255)]

while True:
    for k, (a, b, route) in enumerate(drawn):
        c = PALETTE[k % len(PALETTE)]
        for wp, _ in route[::4]:
            world.debug.draw_point(wp.transform.location + Z, 0.08, c, 3.0)
        world.debug.draw_string(sps[a].location + Z, f"{k}:START {a}", False, c, 3.0)
        world.debug.draw_string(sps[b].location + Z, f"{k}:END {b}", False, c, 3.0)
    time.sleep(2.5)