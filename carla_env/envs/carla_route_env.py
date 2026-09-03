import os
import subprocess
import sys
import glob
import time
import gymnasium as gym
import pygame
import cv2
from pygame.locals import *

from carla_env.tools.hud import HUD
from carla_env.navigation.planner import RoadOption, compute_route_waypoints
from carla_env.wrappers import *

import carla
from collections import deque
import itertools

# Az intersection_routes lista kikerult - helyette a _sample_route() general
# dinamikusan, a min/max_route_length parameterek alapjan.
eval_routes = itertools.cycle([(48, 21), (0, 72), (28, 83), (61, 39)])

discrete_actions = {
    0: [-1, 1], 1: [0, 1], 2: [1, 1], 3: [0, 0],
}


class CarlaRouteEnv(gym.Env):

    def __init__(self, host="127.0.0.1", port=2000,
                 viewer_res=(1920, 1080), obs_res=(160, 80),
                 town="Town04",
                 # === dinamikus route hossz ===
                 min_route_length=150,   # waypoint = meter (resolution=1.0 miatt)
                 max_route_length=400,
                 # === akadaly-erzekeles hatarai ===
                 obstacle_near=15.0,     # ettol kozelebb: teljes savvalto szabadsag
                 obstacle_far=30.0,      # ennel tavolabb: nincs kedvezmeny
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
                 eval=False,
                 activate_render=True):
        self.town = town
        self.min_route_length = min_route_length
        self.max_route_length = max_route_length
        self.obstacle_near = obstacle_near
        self.obstacle_far = obstacle_far
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
        self.eval = eval

        # Akadaly-jelzes. 0.0 = szabad ut elottem, 1.0 = kozel van valami.
        self.obstacle_ahead = 0.0

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

            settings = self.world.get_settings()
            settings.fixed_delta_seconds = 1 / self.fps
            settings.synchronous_mode = True
            self.world.apply_settings(settings)
            self.client.reload_world(False)  # reload map keeping the world settings

            # self.world.set_weather(carla.WeatherParameters.MidRainyNoon)

            # Create vehicle and attach camera to it
            self.vehicle = Vehicle(self.world, self.world.map.get_spawn_points()[0],
                                   on_collision_fn=lambda e: self._on_collision(e),
                                   on_invasion_fn=lambda e: self._on_invasion(e))

            # Create hud and initialize pygame for visualization
            if self.activate_render:
                pygame.init()
                pygame.font.init()
                # HWSURFACE nelkul: modern rendszereken a szoftveres surface
                # jellemzoen gyorsabb, es nem varja be a kepfrissitest.
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
                                  attach_to=self.vehicle, on_recv_image=lambda e: self._set_observation_image(e),
                                  **seg_settings)

            if self.activate_spectator:
                self.camera = Camera(self.world, width, height,
                                     transform=sensor_transforms["spectator"],
                                     attach_to=self.vehicle, on_recv_image=lambda e: self._set_viewer_image(e))
            if self.activate_lidar:
                self.lidar = Lidar(self.world, transform=sensor_transforms["lidar"],
                                   attach_to=self.vehicle, on_recv_image=lambda e: self._set_lidar_data(e))
        except Exception as e:
            raise e
        # Reset env to set initial state
        self.reset()

    def reset(self, seed=None, options=None):
        # Create new route
        self.num_routes_completed = -1
        self.episode_idx += 1
        self.new_route()

        # Two different variables to differ between success episode and fail episode
        self.terminal_state = False  # Set to True when we want to end episode
        self.success_state = False  # Set to True when we want to end episode.

        self.closed = False  # Set to True when ESC is pressed
        self.extra_info = []  # List of extra info shown on the HUD
        self.observation = self.observation_buffer = None  # Last received observation
        self.ae_reconstruction = None  # AE rekonstrukcio a HUD-hoz (encode_state_fn tolti)
        self.viewer_image = self.viewer_image_buffer = None  # Last received image to show in the viewer
        self.lidar_data = self.lidar_data_buffer = None
        self.step_count = 0

        # Init metrics
        self.total_reward = 0.0
        self.previous_location = self.vehicle.get_transform().location
        self.distance_traveled = 0.0
        self.center_lane_deviation = 0.0
        self.speed_accum = 0.0
        self.routes_completed = 0.0
        self.obstacle_ahead = 0.0
        self.world.tick()
        # Return initial observation
        time.sleep(0.2)
        obs, _, _, _, info = self.step(None)
        time.sleep(0.2)
        return obs, info

    # ------------------------------------------------------------------ route

    def _actor_id(self):
        """A Vehicle wrapper hol .actor.id-t, hol .id-t ad. Mindkettot kezeljuk,
        kulonben a sajat autonkat is akadalykent latnank."""
        v = self.vehicle
        if hasattr(v, "actor") and hasattr(v.actor, "id"):
            return v.actor.id
        return v.id

    def _sample_route(self, max_tries=2000):
        """Random spawn-part huz, amig a route hossza a [min, max] ablakba nem esik.
        A kanyarok / keresztezodesek szama igy epizodrol epizodra random lesz.

        A max_tries azert ilyen bo: Town04 tobbsavos autopalya, ott a random
        spawn-parok ~83%-a savvaltasos, es a megmarado route-oknak is csak a
        tizede esik a hossz-ablakba. 30 probabol ez rendszeresen elbukott
        ("Nem sikerult ervenyes route-ot generalni"). Merve ~2.6 s epizodonkent,
        ami epizodonkent egyszer fut - megeri.

        Visszaad: (start_wp, end_wp, route_waypoints)
        """
        spawn_points = self.world.map.get_spawn_points()
        fallback = None  # utolso ervenyes route, ha a hossz-szures nem jon ossze

        for _ in range(max_tries):
            a, b = np.random.choice(len(spawn_points), 2, replace=False)
            start = self.world.map.get_waypoint(spawn_points[a].location)
            end = self.world.map.get_waypoint(spawn_points[b].location)
            route = compute_route_waypoints(self.world.map, start, end, resolution=1.0)

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
        self.vehicle.set_simulate_physics(False)

        if self.eval:
            # Eval modban fix, reprodukalhato utvonalak.
            spawn_points_list = [self.world.map.get_spawn_points()[i] for i in next(eval_routes)]
            self.start_wp, self.end_wp = [self.world.map.get_waypoint(sp.location)
                                          for sp in spawn_points_list]
            self.route_waypoints = compute_route_waypoints(
                self.world.map, self.start_wp, self.end_wp, resolution=1.0)
        else:
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

        time.sleep(0.2)
        self.vehicle.set_simulate_physics(True)

        # A teleport tavolsaga ne szamitson bele a megtett utba.
        if hasattr(self, "previous_location"):
            self.previous_location = self.vehicle.get_transform().location

    # -------------------------------------------------------------- akadaly

    def _obstacle_ahead(self):
        """Folytonos akadaly-jelzes a sajat savomban, elottem.

        0.0 = nincs semmi obstacle_far meteren belul
        1.0 = van valami obstacle_near-en belul
        koztuk linearis atmenet

        Miert folytonos es nem bool: SAC-nal egy kemeny kuszob azt jelentene,
        hogy ket szinte azonos observation ket kulonbozo rewardot kap. A critic
        ezt nem tudja megjosolni -> magas TD error, instabil tanulas.
        """
        me = self.vehicle.get_transform()
        my_wp = self.world.map.get_waypoint(me.location)
        fwd = me.get_forward_vector()
        my_id = self._actor_id()

        nearest = None
        for actor in self.world.get_actors().filter("vehicle.*"):
            if actor.id == my_id:
                continue

            loc = actor.get_transform().location
            d = me.location.distance(loc)
            if d > self.obstacle_far:
                continue

            # Elottem van-e? (skalarszorzat elojele a haladasi iranyra)
            to_actor = loc - me.location
            if fwd.x * to_actor.x + fwd.y * to_actor.y <= 0.0:
                continue

            # Ugyanabban a savban van-e? A road_id + lane_id par azonositja
            # egyertelmuen a savot; a puszta tavolsag nem lenne eleg, mert a
            # szomszed savban levo auto is lehet 10 m-re.
            wp = self.world.map.get_waypoint(loc)
            if wp is None or wp.road_id != my_wp.road_id or wp.lane_id != my_wp.lane_id:
                continue

            nearest = d if nearest is None else min(nearest, d)

        if nearest is None:
            return 0.0
        span = max(self.obstacle_far - self.obstacle_near, 1e-6)
        return float(np.clip((self.obstacle_far - nearest) / span, 0.0, 1.0))

    def close(self):
        if self.carla_process:
            self.carla_process.terminate()
        pygame.quit()
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
            "Obstacle ahead:      % 7.2f" % self.obstacle_ahead,
            "Routes completed:    % 7.2f" % self.routes_completed,
            "Distance traveled: % 7d m" % self.distance_traveled,
            "Center deviance:   % 7.2f m" % self.distance_from_center,
            "Avg center dev:    % 7.2f m" % (self.center_lane_deviation / self.step_count),
            "Avg speed:      % 7.2f km/h" % (self.speed_accum / self.step_count),
            "Total reward:        % 7.2f" % self.total_reward,
        ])
        if self.activate_spectator:
            # Blit image from spectator camera
            self.viewer_image = self._draw_path(self.camera, self.viewer_image)
            self.display.blit(pygame.surfarray.make_surface(self.viewer_image.swapaxes(0, 1)), (0, 0))

        # Superimpose current observation into top-right corner
        obs_h, obs_w = self.observation.shape[:2]
        pos_observation = (self.display.get_size()[0] - obs_w - 10, 10)
        self.display.blit(pygame.surfarray.make_surface(self.observation.swapaxes(0, 1)), pos_observation)

        # Alatta az AE rekonstrukcioja: ez mutatja, mit tart meg a latent,
        # vagyis mit "lat" tenylegesen az agent.
        if getattr(self, "ae_reconstruction", None) is not None:
            pos_recon = (self.display.get_size()[0] - obs_w - 10, 10 + obs_h + 24)
            self.display.blit(
                pygame.surfarray.make_surface(self.ae_reconstruction.swapaxes(0, 1)),
                pos_recon)
            self.display.blit(
                self.hud.font_mono.render("AE recon", True, (255, 255, 255)),
                (pos_recon[0], pos_recon[1] - 18))

        if self.activate_lidar:
            lidar_h, lidar_w = self.lidar_data.shape[:2]
            pos_lidar = (self.display.get_size()[0] - obs_w - 10, 100)
            self.display.blit(pygame.surfarray.make_surface(self.lidar_data.swapaxes(0, 1)), pos_lidar)

        # Render HUD
        self.hud.render(self.display, extra_info=self.extra_info)
        self.extra_info = []  # Reset extra info list

        # Render to screen
        pygame.display.flip()

    def step(self, action):
        if self.closed:
            raise Exception("CarlaEnv.step() called after the environment was closed." +
                            "Check for info[\"closed\"] == True in the learning loop.")
        # Take action
        if action is not None:
            # Create new route on route completion
            if self.current_waypoint_index >= len(self.route_waypoints) - 1:
                if not self.eval:
                    self.new_route()
                else:
                    self.success_state = True

            if self.action_space_type == "continuous":
                steer, throttle = [float(a) for a in action]
            elif self.action_space_type == "discrete":
                steer, throttle = discrete_actions[action]

            self.vehicle.control.steer = smooth_action(self.vehicle.control.steer, steer, self.action_smoothing)
            self.vehicle.control.throttle = smooth_action(self.vehicle.control.throttle, throttle,
                                                          self.action_smoothing)
        # Tick game
        _t0 = time.perf_counter()
        self.world.tick()
        _t1 = time.perf_counter()

        # Get most recent observation and viewer image
        self.observation = self._get_observation()
        if self.activate_spectator:
            self.viewer_image = self._get_viewer_image()

        if self.activate_lidar:
            self.lidar_data = self._get_lidar_data()
        _t2 = time.perf_counter()

        # Get vehicle transform
        transform = self.vehicle.get_transform()

        # Keep track of closest waypoint on the route
        self.prev_waypoint_index = self.current_waypoint_index
        waypoint_index = self.current_waypoint_index
        for _ in range(len(self.route_waypoints)):
            # Check if we passed the next waypoint along the route
            next_waypoint_index = waypoint_index + 1
            wp, _ = self.route_waypoints[next_waypoint_index % len(self.route_waypoints)]
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

        # === akadaly-tudatos saveltere-meres ===
        self.obstacle_ahead = self._obstacle_ahead()

        # (a) Eltres a ROUTE savjatol - ez az eredeti metrika.
        route_dev = distance_to_line(vector(self.current_waypoint.transform.location),
                                     vector(self.next_waypoint.transform.location),
                                     vector(transform.location))

        # (b) Eltres attol a savtol, amiben EPPEN vagyok. A get_waypoint() a
        #     legkozelebbi driving sav kozepvonalara vetit, tehat ez akkor is
        #     kicsi, ha atmentem a szomszed savba - felteve hogy ott a sav
        #     kozepen vagyok.
        lane_wp = self.world.map.get_waypoint(transform.location)
        lane_dev = (transform.location.distance(lane_wp.transform.location)
                    if lane_wp is not None else route_dev)

        # Sulyozott keveres. w = 0 -> pontosan az eredeti viselkedes.
        # w = 1 -> barmelyik sav kozepvonalan lenni rendben van (min()), tehat
        # a savvaltas nem buntetodik, de a savon beluli kanyargas igen.
        w = self.obstacle_ahead
        self.distance_from_center = (1.0 - w) * route_dev + w * min(route_dev, lane_dev)
        self.center_lane_deviation += self.distance_from_center

        # Calculate distance traveled
        if action is not None:
            self.distance_traveled += self.previous_location.distance(transform.location)
        self.previous_location = transform.location

        # Accumulate speed
        self.speed_accum += self.vehicle.get_speed()

        # Terminal on max distance
        if self.distance_traveled >= self.max_distance and not self.eval:
            self.success_state = True

        self.distance_from_center_history.append(self.distance_from_center)

        # Call external reward fn
        self.last_reward = self.reward_fn(self)
        self.total_reward += self.last_reward

        # Encode the state
        encoded_state = self.encode_state_fn(self)
        self.step_count += 1

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
        # obs    = mennyit vartunk a kamerakepekre (szenzor atvitel)
        # render = a pygame oldal (blit, draw_path, flip)
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
            'obstacle_ahead': self.obstacle_ahead,
        }
        done = self.terminal_state or self.success_state
        return encoded_state, self.last_reward, done, False, info

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
        """
            Draw a connected path from start of route to end using homography.

        Optimalizalva: a K projekcios matrix egyszer szamolodik ki (nem minden
        waypointra ujra), es a ciklus csak draw_path_lookahead waypointig megy.
        A regi verzio 800 hosszu route-nal 800-szor futott le tickenkent, ami
        fps=15-nel 12000 felesleges iteracio masodpercenkent.
        """
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

    # A ket busy-wait ciklus helyett rovid alvas. Az ures 'while: pass' 100%
    # CPU-t eszik egy magon, es pont azokert a magokert versenyez, amelyeken a
    # CARLA kliens szalai a kepet szallitanak - vagyis lassitja azt, amire var.
    def _get_observation(self):
        while self.observation_buffer is None:
            time.sleep(0.001)
        obs = self.observation_buffer.copy()
        self.observation_buffer = None
        return obs

    def _get_viewer_image(self):
        while self.viewer_image_buffer is None:
            time.sleep(0.001)
        image = self.viewer_image_buffer.copy()
        self.viewer_image_buffer = None
        return image

    def _get_lidar_data(self):
        while self.lidar_data_buffer is None:
            time.sleep(0.001)
        image = self.lidar_data_buffer.copy()
        self.lidar_data_buffer = None
        return image

    def _on_collision(self, event):
        if get_actor_display_name(event.other_actor) != "Road":
            self.terminal_state = True
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

    def _set_observation_image(self, image):
        self.observation_buffer = image

    def _set_viewer_image(self, image):
        self.viewer_image_buffer = image

    def _set_lidar_data(self, image):
        self.lidar_data_buffer = image