import time
import gymnasium as gym
import pygame
import cv2
from pygame.locals import *

from carla_env.tools.hud import HUD
from carla_env.navigation.planner import (RoadOption, compute_route_waypoints,
                                          build_route_planner, path_has_lane_change)
from carla_env.wrappers import *

import carla
from collections import deque
import itertools




class CarlaRouteEnv(gym.Env):

    def __init__(self, host="127.0.0.1", port=2000,
                 viewer_res=(1920, 1080), obs_res=(160, 80),
                 town="Town04",
                 # === dinamikus route hossz ===
                 min_route_length=150,   # waypoint = meter (resolution=1.0 miatt)
                 max_route_length=400,
                 # === render optimalizacio ===
                 draw_path_lookahead=60,  # hany waypointot vetitsunk a kepre
                 reward_fn=None,
                 observation_space=None,
                 encode_state_fn=None,
                 fps=15,
                 action_smoothing=0.0,
                 action_space_type="continuous",
                 activate_spectator=True,
                 activate_lidar=False,
                 activate_render=True,
                 # === forgalom (minden reset-ben uj, lasd _spawn_traffic) ===
                 traffic_vehicles=0,             # max ennyi auto epizodonkent, 0 = nincs
                 traffic_start_m=30.0,           # az ego elott ennyitol kezdve [m]
                 traffic_gap_m=(20.0, 50.0),     # ket egymas utani auto kozt a route menten [m]
                 traffic_speed_kmh=(30.0, 60.0),
                 hybrid_radius=70.0,             # ezen belul fut valodi fizika [m]
                 tm_port=8000):
        self.town = town
        self.traffic_vehicles = traffic_vehicles
        self.traffic_start_m = traffic_start_m
        self.traffic_gap_m = traffic_gap_m
        self.traffic_speed_kmh = traffic_speed_kmh
        self.traffic_ids = []
        self.tm = None
        self.collision_with = None
        self.min_route_length = min_route_length
        self.max_route_length = max_route_length
        self.draw_path_lookahead = draw_path_lookahead

        width, height = viewer_res
        if obs_res is None:
            out_width, out_height = width, height
        else:
            out_width, out_height = obs_res
        self.activate_render = activate_render

        # Setup gym environment
        self.action_space_type = action_space_type
        if self.action_space_type == "continuous":
            self.action_space = gym.spaces.Box(np.array([-1, 0], dtype=np.float32), np.array([1, 1], dtype=np.float32), dtype=np.float32)  # steer, throttle
        self.observation_space = observation_space

        self.fps = fps
        self.action_smoothing = action_smoothing
        self.episode_idx = -2

        self.encode_state_fn = (lambda x: x) if not callable(encode_state_fn) else encode_state_fn

        self.reward_fn = (lambda x: 0) if not callable(reward_fn) else reward_fn
        self.max_distance = 30000  # m
        self.activate_spectator = activate_spectator
        self.activate_lidar = activate_lidar

        # A spectator kamera projekcios matrixa. A kameraparameterek nem
        # valtoznak futas kozben, ezert eleg egyszer kiszamolni - a regi kod
        # minden waypointra ujraepitette ugyanezt a 3x3 matrixot.
        self._proj_K = None

        self.world = None
        try:
            # Connect to carla
            self.client = carla.Client(host, port)
            self.client.set_timeout(60.0)
            # Create world wrapper
            self.world = World(self.client, self.town)
            self.map = self.world.get_map()

            settings = self.world.get_settings()
            settings.fixed_delta_seconds = 1 / self.fps
            settings.synchronous_mode = True
            self.world.apply_settings(settings)
            self.client.reload_world(False)  # reload map keeping the world settings

            # A route planner grafjat egyszer epitjuk fel. Korabban minden
            # route-probalkozas ujraepitette, es reset kozben a sim allt.
            self._grp = build_route_planner(self.world.map, resolution=1.0)

            # Forgalom: TrafficManager szinkron modban (a world.tick() lepteti),
            # hybrid physics-szel. A hero (az ego) hybrid_radius-an kivul a
            # forgalom fizika nelkul, teleporttal halad - igy a sok auto nem
            # terheli a szervert.
            if self.traffic_vehicles > 0:
                self.tm = self.client.get_trafficmanager(tm_port)
                self.tm.set_synchronous_mode(True)
                self.tm.set_hybrid_physics_mode(True)
                self.tm.set_hybrid_physics_radius(hybrid_radius)
                self.tm.set_global_distance_to_leading_vehicle(2.5)
                self._traffic_bps = [bp for bp in self.world.get_blueprint_library().filter("vehicle.*")
                                     if int(bp.get_attribute("number_of_wheels")) == 4]

            # self.world.set_weather(carla.WeatherParameters.MidRainyNoon)

            # Create vehicle and attach camera to it
            self.vehicle = Vehicle(self.world, self.world.map.get_spawn_points()[0],
                                   on_collision_fn=lambda e: self._on_collision(e),
                                   on_invasion_fn=lambda e: self._on_invasion(e))

            # Create hud and initialize pygame for visualization
            if self.activate_render:
                pygame.init()
                pygame.font.init()
                pygame.display.set_caption("CARLA RL Train HUD")
                self.display = pygame.display.set_mode((width, height), pygame.DOUBLEBUF)

                self.clock = pygame.time.Clock()
                self.hud = HUD(width, height)
                self.hud.set_vehicle(self.vehicle)
                self.world.on_tick(self.hud.on_world_tick)

            # A 'seg_camera' kulcs jelenlete kapcsolja be a szemantikus kamerat.
            # A CityScapesPalette a CARLA sajat, szerveroldali konverzioja -
            # mindig az adott verzio osztalykeszletet hasznalja, ezert nem tud
            # elavulni, mint egy kezzel irt palettatabla.
            seg_settings = {}
            if "seg_camera" in self.observation_space.keys():
                seg_settings.update({
                    'camera_type': "sensor.camera.semantic_segmentation",
                    'color_converter': carla.ColorConverter.CityScapesPalette
                })
            self.dashcam = Camera(self.world, out_width, out_height,
                                  transform=sensor_transforms["dashboard"],
                                  attach_to=self.vehicle,
                                  on_recv_image=lambda e, f: self._set_observation_image(e, f),
                                  **seg_settings)

            if self.activate_spectator:
                self.camera = Camera(self.world, width, height,
                                     transform=sensor_transforms["spectator"],
                                     attach_to=self.vehicle,
                                     on_recv_image=lambda e, f: self._set_viewer_image(e, f))
            if self.activate_lidar:
                # Ket kimenet, ket celra: a BEV kep CSAK a HUD-hoz kell (ezert
                # render nelkul be sem kerjuk), a nyers pontfelho pedig a
                # range image-en keresztul a halonak megy.
                self.lidar = Lidar(
                    self.world, transform=sensor_transforms["lidar"],
                    attach_to=self.vehicle,
                    on_recv_points=lambda p, f: self._set_lidar_points(p, f),
                    rotation_frequency=self.fps)
        except Exception as e:
            raise e
        # Reset env to set initial state
        self.reset()
    
    def get_vehicle_surroundings(self, radius=10.0, lead_range=40.0):
        """A forgalom az egohoz kepest. A step() hivja lepesenkent egyszer
        (self.surroundings), a HUD es a reward_fn is azt olvassa.

          vehicles      a HUD jelzoi: van-e auto radius-on belul elol / hatul /
                        balra / jobbra (az ego sajat iranyahoz kepest)
          left_marking, right_marking   a sajat savom felfestese, az EGO
                        menetiranyahoz kepest (a szembesavban megforditva)
          front_id, front_dist, front_speed   a sajat savomban elottem levo
                        legkozelebbi auto lead_range-en belul (a sebessege
                        km/h-ban), kulonben None
          left_free     a bal szomszed sav letezik, es -8..+20 m-en szabad
          lanes         a szomszed savok az ego iranyabol: +1 bal, +2 ketto
                        balra, -1 jobb, -2 ketto jobbra (ami nincs, az kimarad).
                        broken: odaig minden vonal szaggatott; free: -8..+20
                        m-en szabad (ide allhat be); cross: -8..+8 m-en szabad
                        (at lehet vagni rajta a tulso savba)
          lon           minden forgalmi auto hosszanti tavolsaga [m], + = elottem
        """
        tr = self.vehicle.get_transform()
        loc = tr.location
        fwd = tr.get_forward_vector()
        right = tr.get_right_vector()

        ego_wp = self.map.get_waypoint(loc)
        half_lane = ego_wp.lane_width / 2.0

        # A CARLA a bal/jobb oldalt es a next()-et a SAV iranyahoz adja. Ha a
        # szembesavban megyek (elozes ketsavos uton), a sav bal oldala nekem a
        # jobb, a next() pedig hatrafele visz.
        wp_fwd = ego_wp.transform.get_forward_vector()
        ego_along = wp_fwd.x * fwd.x + wp_fwd.y * fwd.y >= 0.0
        if ego_along:
            left_mark, right_mark = ego_wp.left_lane_marking, ego_wp.right_lane_marking
        else:
            left_mark, right_mark = ego_wp.right_lane_marking, ego_wp.left_lane_marking

        # Szomszed savok, mindket oldalon ketto. Masodikat csak azonos iranyu
        # savon lepunk; szembesavba csak egyet balra (elozes ketsavos uton).
        lanes = {}
        for side in (1, -1):
            wp, marks = ego_wp, []
            for n in (1, 2):
                # a SAV iranyahoz kepest bal-e (a szembesavban forditva)
                lane_left = (side > 0) == ego_along
                marks.append(str((wp.left_lane_marking if lane_left else wp.right_lane_marking).type))
                wp = wp.get_left_lane() if lane_left else wp.get_right_lane()
                if wp is None or wp.lane_type != carla.LaneType.Driving:
                    break
                same_dir = wp.lane_id * ego_wp.lane_id > 0
                if not same_dir and (side < 0 or n > 1):
                    break
                wf = wp.transform.get_forward_vector()
                lanes[side * n] = dict(wp=wp, along=wf.x * fwd.x + wf.y * fwd.y >= 0.0,
                                       broken=all(m == "Broken" for m in marks),
                                       free=True, cross=True)
                if not same_dir:
                    break

        flags = {"front": False, "behind": False, "left": False, "right": False}
        front_id, front_dist, front_actor = None, None, None
        lon_by_id = {}

        for other in (self.world.get_actors(self.traffic_ids) if self.traffic_ids else []):
            o = other.get_location()
            dx, dy = o.x - loc.x, o.y - loc.y
            lon = dx * fwd.x + dy * fwd.y      # + előre, - hátra
            lat = dx * right.x + dy * right.y  # + jobbra, - balra
            lon_by_id[other.id] = lon

            if loc.distance(o) <= radius:
                if abs(lat) < half_lane:       # a saját sávomban van
                    flags["front" if lon > 0 else "behind"] = True
                else:
                    flags["right" if lat > 0 else "left"] = True

            # Sajat sav: a savomat lon meterrel elore kovetve az auto a sav
            # kozepetol OLDALIRANYBAN fel savszelessegen belul van-e. Az ego
            # iranyahoz mert lat kanyarban 40 m-en mar meteres hibat ad, a sav
            # kovetese nem. Hosszanti iranyban 6 m tures: kanyarban a lon
            # (egyenes vetulet) rovidebb, mint a sav menti ut, a pont az auto
            # moge esik - merve 30-40 m-en 3.5 m-rel, es a sima tavolsaggal
            # a sajat savban elottem levo autok 75%-a kimaradt.
            if 0.1 < lon < lead_range and (front_dist is None or lon < front_dist):
                for w in (ego_wp.next(lon) if ego_along else ego_wp.previous(lon)):
                    wl, wf, wr = w.transform.location, w.transform.get_forward_vector(), w.transform.get_right_vector()
                    ddx, ddy = o.x - wl.x, o.y - wl.y
                    if abs(ddx * wr.x + ddy * wr.y) < half_lane and abs(ddx * wf.x + ddy * wf.y) < 6.0:
                        front_id, front_dist, front_actor = other.id, lon, other
            # Szomszed savok foglaltsaga, ugyanigy (elore vagy hatra). A 0.75
            # savszelesseg: a savhataron allo auto mindket savot foglalja.
            if -8.0 <= lon <= 20.0:
                for lane in lanes.values():
                    if not (lane["free"] or (lane["cross"] and lon <= 8.0)):
                        continue
                    lane_wp = lane["wp"]
                    if lon > 0.1:
                        pts = lane_wp.next(lon) if lane["along"] else lane_wp.previous(lon)
                    elif lon < -0.1:
                        pts = lane_wp.previous(-lon) if lane["along"] else lane_wp.next(-lon)
                    else:
                        pts = [lane_wp]
                    for w in pts:
                        wl, wf, wr = w.transform.location, w.transform.get_forward_vector(), w.transform.get_right_vector()
                        ddx, ddy = o.x - wl.x, o.y - wl.y
                        if abs(ddx * wr.x + ddy * wr.y) < 0.75 * w.lane_width and abs(ddx * wf.x + ddy * wf.y) < 6.0:
                            lane["free"] = False
                            lane["cross"] = lane["cross"] and lon > 8.0

        # Csak a lead autoe kell (a reward_fn kovetesi jutalma ehhez meri az
        # ego sebesseget) - a tobbiet nem kerdezzuk le.
        front_speed = None
        if front_actor is not None:
            v = front_actor.get_velocity()
            front_speed = 3.6 * np.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)

        return {"vehicles": flags,
                "left_marking": str(left_mark.type), "right_marking": str(right_mark.type),
                "lane_width": ego_wp.lane_width,
                "front_id": front_id, "front_dist": front_dist, "front_speed": front_speed,
                "left_free": lanes[1]["free"] if 1 in lanes else False,
                "lanes": {k: dict(broken=v["broken"], free=v["free"], cross=v["cross"])
                          for k, v in lanes.items()},
                "lon": lon_by_id}

    def reset(self, seed=None, options=None):
        # Create new route
        self.num_routes_completed = -1
        self.episode_idx += 1
        # Az elozo epizod forgalma a teleport ELOTT tunik el, kulonben az ego
        # akar egy ott allo autoba erkezhetne.
        self._destroy_traffic()
        self.new_route()
        self._spawn_traffic()

        # Two different variables to differ between success episode and fail episode
        self.terminal_state = False  # Set to True when we want to end episode
        self.success_state = False  # Set to True when we want to end episode.
        self.collision_with = None  # az elso utkozes masik szereplojenek type_id-je
        self.terminal_reason = ""   # mi zarta le az epizodot (a reward_fn allitja; "" = nem terminal)

        self.closed = False  # Set to True when ESC is pressed
        self.extra_info = []  # List of extra info shown on the HUD
        self.observation = self.observation_buffer = None  # Last received observation
        self.ae_reconstruction = None  # AE rekonstrukcio a HUD-hoz (encode_state_fn tolti)
        self.viewer_image = self.viewer_image_buffer = None  # Last received image to show in the viewer
        self.lidar_points = self.lidar_points_buffer = None
        # BEV a HUD-hoz: amit a halo kap / amit visszaad (encode_state_fn tolti)
        self.lidar_bev_input = self.lidar_bev_recon = None
        self.step_count = 0

        # Init metrics
        self.total_reward = 0.0
        self.previous_location = self.vehicle.get_transform().location
        self.distance_traveled = 0.0
        self.center_lane_deviation = 0.0
        self.speed_accum = 0.0
        self.routes_completed = 0.0
        # Elozes-metrikak a TensorBoardhoz (a reward_fn noveli oket)
        self.overtakes = 0             # kesz elozesek szama
        self.overtake_reward = 0.0     # elozes kozben kapott jutalom + bonusz
        self.blocked_time = 0.0        # ennyi ideig kovetett, mert nem elozhetett [s]
        self.world.tick()
        # Return initial observation. (Alvas nem kell: szinkron modban a
        # szerver ugysem lep a tick nelkul, a szenzoradatot pedig a step
        # frame szerint varja meg.)
        obs, _, _, _, info = self.step(None)
        return obs, info

    # ------------------------------------------------------------------ route

    def _sample_route(self, max_tries=2000):
        """Random spawn-part huz, amig a route hossza a [min, max] ablakba nem esik.
        A kanyarok / keresztezodesek szama igy epizodrol epizodra random lesz.

        A max_tries azert ilyen bo: Town04 tobbsavos autopalya, ott a random
        spawn-parok ~83%-a savvaltasos, es a megmarado route-oknak is csak a
        tizede esik a hossz-ablakba. 30 probabol ez rendszeresen elbukott
        ("Nem sikerult ervenyes route-ot generalni").

        Gyorsitas: a planner egyszer epul (self._grp), es a savvaltasos
        parokat mar a graf-uton kiszurjuk, a draga trace_route elott (a
        route-ok ugyanazok). Merve (Town04, 400-800 m): reset ~0.1 s,
        korabban ~1.6 s, amig a sim allt.

        Visszaad: (start_wp, end_wp, route_waypoints)
        """
        spawn_points = self.world.map.get_spawn_points()
        fallback = None  # utolso ervenyes route, ha a hossz-szures nem jon ossze

        for _ in range(max_tries):
            a, b = np.random.choice(len(spawn_points), 2, replace=False)
            start = self.world.map.get_waypoint(spawn_points[a].location)
            end = self.world.map.get_waypoint(spawn_points[b].location)

            # Savvaltasos par eldobasa MEG a trace_route elott (lasd lent).
            if path_has_lane_change(self._grp, start.transform.location, end.transform.location):
                continue

            route = compute_route_waypoints(self.world.map, start, end, resolution=1.0, grp=self._grp)

            # Ervenytelen / degeneralt route (start == end, vagy nincs osszekottetes)
            if len(route) <= 1:
                continue

            # Savvaltasos route kiszurese.
            # Ilyenkor a route waypointjai egy tick alatt atugranak a szomszedos
            # sav kozepvonalara -> a distance_from_center hirtelen ~3.5 m lesz,
            # mikozben az auto egyenesen megy. Az agens ok nelkul halna meg.
            if any(o.name.startswith("CHANGELANE") for _, o in route):
                continue

            # Csak ide er el, ami mar minden minosegi szuron atment, tehat a
            # fallback sem lehet savvaltasos.
            fallback = (start, end, route)

            if self.min_route_length <= len(route) <= self.max_route_length:
                return fallback

        if fallback is None:
            raise RuntimeError(
                "Nem sikerult ervenyes route-ot generalni {} probabol. "
                "Probald emelni a max_tries-t vagy tagitani a hossz-ablakot."
                .format(max_tries))
        return fallback

    def new_route(self):
        # Do a soft reset (teleport vehicle).
        # A fizika kikapcsolasa azert kell, mert teleportkor a motor megorizne a
        # sebesseget/impulzust, es az auto kiloone vagy felborulna az uj helyen.
        self.vehicle.control.steer = float(0.0)
        self.vehicle.control.throttle = float(0.0)
        self.vehicle.control.brake = float(0.0)
        self.vehicle.set_simulate_physics(False)

        self.start_wp, self.end_wp, self.route_waypoints = self._sample_route()

        self.distance_from_center_history = deque(maxlen=30)

        self.current_waypoint_index = 0
        self.num_routes_completed += 1

        # A route elso waypointjara teleportalunk, nem a start_wp-re.
        # A GlobalRoutePlanner a topologia-graf elei menten dolgozik, ezert a
        # route[0] gyakran nem azonos a start_wp-vel - tobbsavos uton (Town04)
        # akar masik savba is eshet. Ha a start_wp-re tennenk az autot, a
        # distance_from_center mar az elso ticktol egy savszelesseg lenne, es a
        # reward_fn azonnal terminalna.
        wp0 = self.route_waypoints[0][0]
        spawn_tf = carla.Transform(
            wp0.transform.location + carla.Location(z=0.5),  # ne essen az aszfaltba
            wp0.transform.rotation)                          # a route iranyaba nezzen
        self.vehicle.set_transform(spawn_tf)
        self.vehicle.set_simulate_physics(True)

        # A teleport tavolsaga ne szamitson bele a megtett utba. A cel-
        # transzformot vesszuk, mert szinkron modban a get_transform() a
        # kovetkezo tickig meg a teleport ELOTTI helyet adja.
        if hasattr(self, "previous_location"):
            self.previous_location = spawn_tf.location

    # -------------------------------------------------------------- forgalom

    def _traffic_slots(self):
        """Hova keruljon forgalom: waypointok a route menten, az ego elott.

        traffic_start_m-tol a route vegeig traffic_gap_m-enkent (veletlen) egy
        auto, a route savjaban vagy egy azonos iranyu szomszed savban - igy
        lepcsozetesen allnak, es van hol elmenni melluk. Keresztezodesbe nem
        rakunk (ott a TM ugyis keresztbe allna). A np.random-ot hasznalja,
        mint a route-sorsolas: ugyanaz a seed -> ugyanaz a forgalom.
        """
        slots = []
        d = self.traffic_start_m
        while len(slots) < self.traffic_vehicles and d < len(self.route_waypoints) - 1:
            wp = self.route_waypoints[int(d)][0]
            if not wp.is_junction:
                lanes = [wp] + [n for n in (wp.get_left_lane(), wp.get_right_lane())
                                if n is not None and n.lane_type == carla.LaneType.Driving
                                and n.lane_id * wp.lane_id > 0]   # azonos irany
                slots.append(lanes[np.random.randint(len(lanes))])
            d += np.random.uniform(*self.traffic_gap_m)
        return slots

    def _spawn_traffic(self):
        """Uj forgalom a most sorsolt route menten, az ego elott.

        A forgalom lassabb az egonal (traffic_speed_kmh), nem valt savot es
        nem all meg lampanal - az egonak kell kikerulnie oket.
        """
        if self.tm is None:
            return
        port = self.tm.get_port()
        batch = []
        for wp in self._traffic_slots():
            bp = self._traffic_bps[np.random.randint(len(self._traffic_bps))]
            if bp.has_attribute("color"):
                colors = bp.get_attribute("color").recommended_values
                bp.set_attribute("color", colors[np.random.randint(len(colors))])
            bp.set_attribute("role_name", "traffic")
            tf = carla.Transform(wp.transform.location + carla.Location(z=0.5), wp.transform.rotation)
            batch.append(carla.command.SpawnActor(bp, tf).then(
                carla.command.SetAutopilot(carla.command.FutureActor, True, port)))
        # A True egy tick-et is lep: ekkor all be az ego teleportja is.
        self.traffic_ids = [r.actor_id for r in self.client.apply_batch_sync(batch, True)
                            if not r.error]
        for actor in self.world.get_actors(self.traffic_ids):
            self.tm.set_desired_speed(actor, float(np.random.uniform(*self.traffic_speed_kmh)))
            self.tm.auto_lane_change(actor, False)
            self.tm.ignore_lights_percentage(actor, 100)

    def _destroy_traffic(self):
        if self.traffic_ids:
            self.client.apply_batch_sync(
                [carla.command.DestroyActor(i) for i in self.traffic_ids], False)
        self.traffic_ids = []

    def close(self):
        # A carla_process-t semmi nem allitja be (a szervert kivulrol inditjuk),
        # a sima self.carla_process AttributeError-t dobott.
        if getattr(self, "carla_process", None):
            self.carla_process.terminate()
        pygame.quit()
        self._destroy_traffic()
        if self.tm is not None:
            self.tm.set_synchronous_mode(False)
        if self.world is not None:
            self.world.destroy()
        self.closed = True

    def render(self, mode="human"):
        if mode == "rgb_array_no_hud":
            return self.viewer_image
        elif mode == "rgb_array":
            # Turn display surface into rgb_array
            return np.array(pygame.surfarray.array3d(self.display), dtype=np.uint8).transpose([1, 0, 2])
        elif mode == "state_pixels":
            return self.observation

        # Tick render clock
        self.clock.tick()
        self.hud.tick(self.world, self.clock)

        # Get maneuver name
        if self.current_road_maneuver == RoadOption.LANEFOLLOW:
            maneuver = "Follow Lane"
        elif self.current_road_maneuver == RoadOption.LEFT:
            maneuver = "Left"
        elif self.current_road_maneuver == RoadOption.RIGHT:
            maneuver = "Right"
        elif self.current_road_maneuver == RoadOption.STRAIGHT:
            maneuver = "Straight"
        else:
            maneuver = "INVALID"

        # Add metrics to HUD
        self.extra_info.extend([
            "Episode {}".format(self.episode_idx),
            "Reward: % 19.2f" % self.last_reward,
            "",
            "Maneuver:        % 11s" % maneuver,
            "Route length:      % 7d m" % len(self.route_waypoints),
            "Routes completed:    % 7.2f" % self.routes_completed,
            "Distance traveled: % 7d m" % self.distance_traveled,
            "Center deviance:   % 7.2f m" % self.distance_from_center,
            "Avg center dev:    % 7.2f m" % (self.center_lane_deviation / self.step_count),
            "Avg speed:      % 7.2f km/h" % (self.speed_accum / self.step_count),
            "Total reward:        % 7.2f" % self.total_reward,
        ])

        # Kornyezet: jarmu 30 m-en belul (a HUD a (cimke, bool) part
        # jelolonegyzetkent rajzolja, teli = van ott auto), es a sav felfestese.
        # A step() szamolja, ugyanazt latja a reward_fn is.
        surr = self.surroundings
        veh = surr["vehicles"]
        self.extra_info.extend([
            "",
            "Vehicles within 30 m:",
            ("Front:", veh["front"]),
            ("Behind:", veh["behind"]),
            ("Left:", veh["left"]),
            ("Right:", veh["right"]),
            "Left line:  % 16s" % surr["left_marking"],
            "Right line: % 16s" % surr["right_marking"],
            "Lead car:   %s" % ("% 14.1f m" % surr["front_dist"] if surr["front_id"] is not None else "             -"),
            ("Left lane free:", surr["left_free"]),
        ])
        if self.activate_spectator:
            # Blit image from spectator camera
            self.viewer_image = self._draw_path(self.camera, self.viewer_image)
            self._blit_rgb(self.viewer_image, (0, 0))

        # Jobb oldali oszlop: amit az agent lat. Soronkent egy cim, alatta a
        # kep(ek). A lidar ket kepe egy sorban: bemenet es rekonstrukcio.
        PAD, GAP, TITLE = 10, 8, 18
        rows = [("camera", [self.observation]),
                ("camera AE recon", [self.ae_reconstruction]),
                ("lidar BEV  (in / recon)", [self.lidar_bev_input, self.lidar_bev_recon])]
        rows = [(t, [i for i in imgs if i is not None]) for t, imgs in rows]
        rows = [(t, imgs) for t, imgs in rows if imgs]

        if rows:
            # A panel szelessege a legszelesebb sorhoz igazodik, igy minden
            # elem ugyanabba az oszlopba kerul.
            panel_w = max(sum(i.shape[1] for i in imgs) + GAP * (len(imgs) - 1)
                          for _, imgs in rows) + 2 * PAD
            panel_h = sum(TITLE + max(i.shape[0] for i in imgs) for _, imgs in rows) \
                + GAP * (len(rows) - 1) + 2 * PAD
            panel_x = self.display.get_size()[0] - panel_w - PAD

            # Felatlatszo hatter, mint a bal oldali HUD-sav - kulonben a
            # vilagos kamerakepen a feher felirat olvashatatlan.
            bg = pygame.Surface((panel_w, panel_h))
            bg.set_alpha(100)
            self.display.blit(bg, (panel_x, PAD))

            y = 2 * PAD
            for title, imgs in rows:
                self.display.blit(
                    self.hud.font_mono.render(title, True, (255, 255, 255)),
                    (panel_x + PAD, y))
                y += TITLE
                x = panel_x + PAD
                for img in imgs:
                    self._blit_rgb(img, (x, y))
                    x += img.shape[1] + GAP
                y += max(i.shape[0] for i in imgs) + GAP

        # Render HUD
        self.hud.render(self.display, extra_info=self.extra_info)
        self.extra_info = []  # Reset extra info list

        # Render to screen
        pygame.display.flip()

    def _blit_rgb(self, img, pos):
        """(H, W, 3) uint8 RGB kep kirajzolasa masolas nelkul.

        A make_surface(img.swapaxes(0, 1)) atrendezve lemasolta a kepet
        (1280x720-on 1.5 ms); a frombuffer kozvetlenul a tomb memoriajat
        olvassa. Ehhez folytonos tomb kell - a kamerakep mar az, a kis
        panelkepeknel az ascontiguousarray olcso.
        """
        img = np.ascontiguousarray(img)
        self.display.blit(pygame.image.frombuffer(img, (img.shape[1], img.shape[0]), "RGB"), pos)

    def step(self, action):
        if self.closed:
            raise Exception("CarlaEnv.step() called after the environment was closed." +
                            "Check for info[\"closed\"] == True in the learning loop.")
        # Take action
        # (A route vegen mar nem itt teleportalunk uj route-ra: az epizod
        # "truncated"-del er veget, es az uj route a reset()-ben jon - lasd lent.)

        # action=None: a reset() vegen futo step(None) - csak tick, akcio nelkul.
        if action is not None:
            # Nincs fek-akcio: a fek (control.brake) mindig 0, lassitani a
            # gaz elvetelevel lehet.
            steer, throttle = [float(a) for a in action]
            self.vehicle.control.steer = smooth_action(self.vehicle.control.steer, steer, self.action_smoothing)
            self.vehicle.control.throttle = smooth_action(self.vehicle.control.throttle, throttle,
                                                          self.action_smoothing)
        # Tick game
        _t0 = time.perf_counter()
        frame = self.world.tick()
        _t1 = time.perf_counter()

        # Az EBBEN a tickben keszult observation. A spectator kepet csak a
        # kodolas utan varjuk meg (lent), mert az agensnek nem kell.
        self.observation = self._get_observation(frame)


        self.lidar_points = self._get_lidar_points(frame)
        _t2 = time.perf_counter()

        # Get vehicle transform
        transform = self.vehicle.get_transform()

        # Keep track of closest waypoint on the route
        self.prev_waypoint_index = self.current_waypoint_index
        waypoint_index = self.current_waypoint_index
        # Az index a route utolso pontjanal megall. Korabban "% len" volt itt:
        # a route vegen a ciklus a route ELEJEN folytatta a vizsgalatot, es egy
        # step alatt akar 2*len-ig futott (routes_completed ugrott, a
        # current_waypoint pedig egy tavoli, route eleji pont lehetett).
        for _ in range(len(self.route_waypoints) - 1 - waypoint_index):
            # Check if we passed the next waypoint along the route
            next_waypoint_index = waypoint_index + 1
            wp, _ = self.route_waypoints[next_waypoint_index]
            dot = np.dot(vector(wp.transform.get_forward_vector())[:2],
                         vector(transform.location - wp.transform.location)[:2])
            if dot > 0.0:  # Did we pass the waypoint?
                waypoint_index += 1  # Go to next waypoint
            else:
                break
        self.current_waypoint_index = waypoint_index

        # Check for route completion
        if self.current_waypoint_index < len(self.route_waypoints) - 1:
            self.next_waypoint, self.next_road_maneuver = self.route_waypoints[
                (self.current_waypoint_index + 1) % len(self.route_waypoints)]

        self.current_waypoint, self.current_road_maneuver = self.route_waypoints[
            self.current_waypoint_index % len(self.route_waypoints)]
        self.routes_completed = self.num_routes_completed + (self.current_waypoint_index + 1) / len(
            self.route_waypoints)
        route_done = self.current_waypoint_index >= len(self.route_waypoints) - 1

        # Calculate deviation from center of the lane
        self.distance_from_center = distance_to_line(vector(self.current_waypoint.transform.location),
                                                     vector(self.next_waypoint.transform.location),
                                                     vector(transform.location))
        self.center_lane_deviation += self.distance_from_center

        # Calculate distance traveled
        if action is not None:
            self.distance_traveled += self.previous_location.distance(transform.location)
        self.previous_location = transform.location

        # Accumulate speed
        self.speed_accum += self.vehicle.get_speed()

        # Terminal on max distance
        if self.distance_traveled >= self.max_distance:
            self.success_state = True

        self.distance_from_center_history.append(self.distance_from_center)

        # Forgalom az egohoz kepest - lepesenkent egyszer, a reward es a HUD is ezt olvassa
        self.surroundings = self.get_vehicle_surroundings(radius=30.0)

        # Call external reward fn
        self.last_reward = self.reward_fn(self)
        self.total_reward += self.last_reward

        # Encode the state
        encoded_state = self.encode_state_fn(self)
        self.step_count += 1

        # A 1280x720-as spectator kep a szerveren ~12 ms-mal a dashcam utan
        # er ide. Ha a kodolas elott varnank ra, ez az ido hozzaadodna a
        # step-hez; igy a kodolas alatt erkezik meg.
        if self.activate_spectator:
            self.viewer_image = self._get_viewer_image(frame)

        # DEBUG: Draw path
        # self._draw_path_server(life_time=1.0, skip=8)

        # Check for ESC press
        if self.activate_render:
            pygame.event.pump()
            if pygame.key.get_pressed()[K_ESCAPE]:
                self.terminal_state = True

            self.render()

        # === IDOMERES ===
        # tick   = a CARLA szerver mennyit dolgozott (szerveroldali terheles)
        # obs    = mennyit vartunk a dashcam kepre es a lidarra
        # render = reward, kodolas, spectator kep, pygame (blit, draw_path, flip)
        # Csak a 80 ms folotti stepeket logolja, hogy ne arassza el a konzolt.
        _t3 = time.perf_counter()
        if (_t3 - _t0) > 0.08:
            print("[SLOW] tick={:.0f} obs={:.0f} render={:.0f} ms  (osszesen {:.0f})".format(
                1000 * (_t1 - _t0),
                1000 * (_t2 - _t1),
                1000 * (_t3 - _t2),
                1000 * (_t3 - _t0)))

        info = {
            "closed": self.closed,
            'total_reward': self.total_reward,
            'routes_completed': self.routes_completed,
            'total_distance': self.distance_traveled,
            'avg_center_dev': (self.center_lane_deviation / self.step_count),
            'avg_speed': (self.speed_accum / self.step_count),
            'mean_reward': (self.total_reward / self.step_count),
            'route_length': len(self.route_waypoints),
            'overtakes': self.overtakes,
            'overtake_reward': self.overtake_reward,
            'blocked_time': self.blocked_time,
            'terminal_reason': self.terminal_reason,
        }
        terminated = self.terminal_state
        truncated = (route_done or self.success_state) and not terminated
        return encoded_state, self.last_reward, terminated, truncated, info

    def _draw_path_server(self, life_time=60.0, skip=0):
        """
            Draw a connected path from start of route to end.
            Green node = start
            Red node   = point along path
            Blue node  = destination
        """
        for i in range(0, len(self.route_waypoints) - 1, skip + 1):
            z = 30.25
            w0 = self.route_waypoints[i][0]
            w1 = self.route_waypoints[i + 1][0]
            self.world.debug.draw_line(
                w0.transform.location + carla.Location(z=z),
                w1.transform.location + carla.Location(z=z),
                thickness=0.1, color=carla.Color(255, 0, 0),
                life_time=life_time, persistent_lines=False)
            self.world.debug.draw_point(
                w0.transform.location + carla.Location(z=z), 0.1,
                carla.Color(0, 255, 0) if i == 0 else carla.Color(255, 0, 0),
                life_time, False)
        self.world.debug.draw_point(
            self.route_waypoints[-1][0].transform.location + carla.Location(z=z), 0.1,
            carla.Color(0, 0, 255),
            life_time, False)

    def _draw_path(self, camera, image):
        vehicle_vector = vector(self.vehicle.get_transform().location)
        # Get the world to camera matrix
        world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

        # A kameraparameterek nem valtoznak futas kozben -> egyszer olvassuk be
        # es egyszer epitjuk fel a projekcios matrixot.
        if self._proj_K is None:
            image_w = int(camera.actor.attributes['image_size_x'])
            image_h = int(camera.actor.attributes['image_size_y'])
            fov = float(camera.actor.attributes['fov'])
            self._proj_K = build_projection_matrix(image_w, image_h, fov)
        K = self._proj_K

        # A tavolsagszuro amugy is csak 50 m-en belulit rajzol, ezert felesleges
        # az egesz hatralevo route-on vegigmenni.
        last_index = len(self.route_waypoints) - 1
        end = min(len(self.route_waypoints),
                  self.current_waypoint_index + self.draw_path_lookahead)

        for i in range(self.current_waypoint_index, end):
            waypoint_location = self.route_waypoints[i][0].transform.location + carla.Location(z=1.25)
            waypoint_vector = vector(waypoint_location)
            if not (2 < abs(np.linalg.norm(vehicle_vector - waypoint_vector)) < 50):
                continue
            x, y = get_image_point(waypoint_location, K, world_2_camera)
            if i == last_index:
                color = (255, 0, 0)
            else:
                color = (0, 0, 255)
            image = cv2.circle(image, (int(np.int32(x)), int(np.int32(y))), radius=3, color=color, thickness=-1)
        return image

    def _wait_for_frame(self, buffer_name, frame, timeout=2.0):
        """A `frame` tickhez tartozo szenzoradat. A buffer (frame, adat) part tart.

        Szinkron modban a world.tick() visszater, mielott a szenzorok adata
        megerkezne - a callbackek kulon szalon, kesve jonnek. A regi getter a
        bufferben talalt elso adatot vitte, ami merve a kameraknal MINDIG az
        elozo tick kepe volt, a lidarnal ~60%-ban. Az obs igy egy tickkel
        kesett a sebesseghez/waypointokhoz kepest, es a kamera meg a lidar
        gyakran mas pillanatot mutatott.

        Busy-wait helyett rovid alvas: az ures ciklus 100% CPU-t eszik, es
        pont a CARLA kliens szalaival versenyez, amik az adatot szallitjak.
        """
        deadline = time.perf_counter() + timeout
        while True:
            item = getattr(self, buffer_name)
            if item is not None and item[0] >= frame:
                return item[1]
            if item is not None and time.perf_counter() > deadline:
                # Ne akassza meg a tanitast: a legutolso adattal megyunk tovabb.
                print("[WARN] {}: {} s alatt nem jott adat a {}. frame-hez, "
                      "a {}. frame-et hasznalom".format(buffer_name, timeout, frame, item[0]))
                return item[1]
            time.sleep(0.001)

    # Masolas nem kell: a Camera callback minden kephez uj tombot ad.
    def _get_observation(self, frame):
        return self._wait_for_frame("observation_buffer", frame)

    def _get_viewer_image(self, frame):
        return self._wait_for_frame("viewer_image_buffer", frame)

    def _get_lidar_points(self, frame):
        return self._wait_for_frame("lidar_points_buffer", frame)

    def _on_collision(self, event):
        if get_actor_display_name(event.other_actor) != "Road":
            self.terminal_state = True
            # A reward ebbol tudja, hogy jarmuvel (vehicle.*) vagy massal utkozott.
            if self.collision_with is None:
                self.collision_with = event.other_actor.type_id
        if self.activate_render:
            self.hud.notification("Collision with {}".format(get_actor_display_name(event.other_actor)))

    def _on_invasion(self, event):
        # FIGYELEM: ez NEM terminal. Csak HUD uzenetet ir ki, es azt is csak
        # activate_render mellett. A savelhagyasos halalt a reward_fn okozza a
        # distance_from_center alapjan.
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ["%r" % str(x).split()[-1] for x in lane_types]
        if self.activate_render:
            self.hud.notification("Crossed line %s" % " and ".join(text))

    def _set_observation_image(self, image, frame):
        self.observation_buffer = (frame, image)

    def _set_viewer_image(self, image, frame):
        self.viewer_image_buffer = (frame, image)

    def _set_lidar_points(self, points, frame):
        self.lidar_points_buffer = (frame, points)
        