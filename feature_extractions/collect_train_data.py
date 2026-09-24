import json
import os
import queue
import random
import sys
import time

import numpy as np
import pygame
from pygame.locals import K_BACKSPACE, K_ESCAPE, K_c, K_p, K_r, KMOD_SHIFT

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import carla  # noqa: E402

from feature_extractions.manual_control import VehicleController  # noqa: E402

HOST = "127.0.0.1"
PORT = 2000
TOWN = "Town04"

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
CAMERA_DIR = os.path.join(OUT_DIR, "camera")
LIDAR_DIR = os.path.join(OUT_DIR, "lidar")

FIXED_DELTA = 0.05
SENSOR_HZ = 1.0 / FIXED_DELTA

CAM_WIDTH, CAM_HEIGHT = 640, 320
SAVE_WIDTH, SAVE_HEIGHT = 160, 80
CAM_FOV = 110


LIDAR_CHANNELS = 64
LIDAR_RANGE = 50.0
LIDAR_UPPER_FOV = 10.0
LIDAR_LOWER_FOV = -25.0

LIDAR_ROTATION_HZ = SENSOR_HZ
LIDAR_POINTS_PER_SEC = int(64 * 1024 * LIDAR_ROTATION_HZ)


VOXEL_SIZE = 0

# Csak megjelenites, nem kerul mentesre.
BEV_SIZE = 256
BEV_Z_MIN, BEV_Z_MAX = -2.4, 2.0
# A lidar_ae.py range image parameterei - a HUD ezzel mutatja, mit kapna a halo.
RANGE_SIZE = (64, 1024)
RANGE_FOV = (LIDAR_UPPER_FOV, LIDAR_LOWER_FOV)

SENSOR_TRANSFORMS = {
    "dashboard": carla.Transform(carla.Location(x=1.6, z=1.7)),
    "lidar": carla.Transform(carla.Location(x=0.0, z=2.4)),
    "spectator": carla.Transform(carla.Location(x=-5.5, z=2.8),
                                 carla.Rotation(pitch=-15)),
}

NUM_TRAFFIC = 200

HYBRID_PHYSICS_RADIUS = 70.0
EGO_SPEED_KMH = 90.0
START_WITH_AUTOPILOT = True

START_RECORDING = False
MAX_FRAMES = None


EVERY_N_TICK = 4

# === Sav-kozeptol eltero kepek (csak autopilot mellett) ===
# Az autopilot mindig a sav kozepen, a sav iranyaban megy: a regi adatban
# szinte nincs oldalra csuszott vagy ferde kep, ezert az AE latense alig
# kodolja, hol all az auto a savban - pedig az RL policy-nek pont ez kell,
# amikor lesodrodik. Ket forras:
#   - a TM oldaleltolasa folyamatosan hullamzik (szinusz): az auto a savban
#     jobbra-balra kanyarog, kozben ferden is all. Minden teljes hullam utan
#     uj veletlen amplitudo (LANE_OFFSET_AMP) es periodus (LANE_OFFSET_PERIOD).
#     90 km/h-n 1.5 m / 3 s ~7 fokos szoget ad.
#   - PERTURB_EVERY-nkent par tizedmasodpercre mi kormanyzunk egy veletlen
#     kiteressel, utana az autopilot visszahozza (DART-szeruen). 90 km/h-n a
#     0.05-os kormany 0.4 s alatt ~12 fokot fordit.
# WANDER = False: a regi, sav-kozepes autopilot.
WANDER = True
LANE_OFFSET_AMP = (0.5, 1.5)       # m
LANE_OFFSET_PERIOD = (3.0, 8.0)    # s, egy teljes jobbra-balra hullam
PERTURB_EVERY = (2.0, 5.0)         # s
PERTURB_STEER = (0.02, 0.05)
PERTURB_TICKS = (4, 8)             # 20 Hz-en 0.2-0.4 s


DRAW_EVERY = 1


REALTIME = True

WINDOW_WIDTH, WINDOW_HEIGHT = 1920, 1080


def carla_image_to_rgb(image):
    bgra = np.reshape(np.frombuffer(image.raw_data, dtype=np.uint8),
                      (image.height, image.width, 4))
    return bgra[:, :, :3][:, :, ::-1].copy()


def carla_lidar_to_xyz(measurement):
    """Nyers (N,3) XYZ. Az y elojelet forditjuk: a CARLA bal kezes, a
    lidar_ae.pcd2range jobb kezes rendszert var."""
    pts = np.frombuffer(measurement.raw_data, dtype=np.float32)
    xyz = np.reshape(pts, (-1, 4))[:, :3].copy()
    xyz[:, 1] = -xyz[:, 1]
    return xyz


def voxel_downsample(xyz, voxel=VOXEL_SIZE):
    """Kockankent egy pont. Terben egyenletesen ritkit: a kozeli suru reszek
    ritkulnak, a tavoli ritkak erintetlenul maradnak."""
    if voxel <= 0 or len(xyz) == 0:
        return xyz
    # A racsindexekbol egy 1D kulcsot keszitunk, es kulcsonkent az elso
    # pontot tartjuk meg. A minimum levonasa a negativ indexek miatt kell.
    keys = np.floor(xyz / voxel).astype(np.int64)
    keys -= keys.min(axis=0)
    dims = keys.max(axis=0) + 1
    flat = (keys[:, 0] * dims[1] + keys[:, 1]) * dims[2] + keys[:, 2]
    _, idx = np.unique(flat, return_index=True)
    return xyz[np.sort(idx)]


def xyz_to_range_image(xyz, size=RANGE_SIZE, fov=RANGE_FOV,
                       depth_range=(1.0, LIDAR_RANGE)):
    """Gombi projekcio, a lidar_ae.pcd2range numpy-only masa (torch nelkul).
    Ahol nincs pont, ott -1. EZEN tanul majd a halo, ezert ezt erdemes nezni
    a ritkitas hangolasakor - nem a BEV-et."""
    if len(xyz) == 0:
        return np.full(size, -1, dtype=np.float32)

    fov_up, fov_down = fov[0] / 180.0 * np.pi, fov[1] / 180.0 * np.pi
    fov_range = abs(fov_down) + abs(fov_up)

    depth = np.linalg.norm(xyz, 2, axis=1)
    mask = (depth > depth_range[0]) & (depth < depth_range[1])
    depth, pcd = depth[mask], xyz[mask]
    if len(depth) == 0:
        return np.full(size, -1, dtype=np.float32)

    yaw = -np.arctan2(pcd[:, 1], pcd[:, 0])
    pitch = np.arcsin(pcd[:, 2] / depth)

    proj_x = 0.5 * (yaw / np.pi + 1.0) * size[1]
    proj_y = (1.0 - (pitch + abs(fov_down)) / fov_range) * size[0]
    proj_x = np.maximum(0, np.minimum(size[1] - 1, np.floor(proj_x))).astype(np.int32)
    proj_y = np.maximum(0, np.minimum(size[0] - 1, np.floor(proj_y))).astype(np.int32)

    # Csokkeno tavolsag szerint: a kozelebbi pont irja felul a tavolabbit.
    order = np.argsort(depth)[::-1]
    out = np.full(size, -1, dtype=np.float32)
    out[proj_y[order], proj_x[order]] = depth[order]
    return out


def range_image_to_rgb(ri, max_range=LIDAR_RANGE):
    """Range image szinezve: kozel piros, tavol kek, ures cella fekete."""
    v = np.clip(ri / max_range, 0.0, 1.0)
    rgb = np.zeros((ri.shape[0], ri.shape[1], 3), dtype=np.uint8)
    rgb[:, :, 0] = ((1.0 - v) * 255).astype(np.uint8)
    rgb[:, :, 2] = (v * 255).astype(np.uint8)
    rgb[ri < 0] = 0
    return rgb


def xyz_to_bev_rgb(xyz, size=BEV_SIZE, max_range=LIDAR_RANGE,
                   z_min=BEV_Z_MIN, z_max=BEV_Z_MAX):
    """Felulnezeti kep a kepernyore. A szin a magassag: zold -> sarga -> piros."""
    if len(xyz) == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)

    scale = size / (2.0 * max_range)
    px = xyz[:, 0] * scale + 0.5 * size
    py = xyz[:, 1] * scale + 0.5 * size

    inside = (px >= 0) & (px < size) & (py >= 0) & (py < size)
    px = px[inside].astype(np.int32)
    py = py[inside].astype(np.int32)
    z_norm = np.clip((xyz[inside, 2] - z_min) / (z_max - z_min), 0.0, 1.0)

    # maximum.at: egy cellaba tobb pont eshet, a legmagasabbat tartjuk.
    # A +1e-3 elvalasztja a legalso pontot az ures cellatol.
    bev = np.zeros((size, size), dtype=np.float32)
    np.maximum.at(bev, (py, px), z_norm + 1e-3)

    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    rgb[:, :, 0] = np.clip(bev * 2.0, 0, 1) * 255
    rgb[:, :, 1] = np.clip(2.0 - bev * 2.0, 0, 1) * 255
    rgb[bev <= 0.0] = 0
    c = size // 2
    rgb[c - 2:c + 3, c - 1:c + 2] = 255      # az ego helye
    return rgb


def _drain_until(q, frame, timeout):
    """A queue-bol olvas, amig el nem eri a kert frame-et.

    A korabbi mereseket eldobja (azok mar elavultak), a kesobbieket nem
    tudja visszatenni - de szinkron modban ilyen nincs, mert framenkent
    pontosan egy meres keletkezik.
    """
    while True:
        data = q.get(timeout=timeout)
        if data.frame >= frame:
            return data


def frame_name(frame):
    """A fajlnev a SZIMULACIOS FRAME sorszama.

    Korabban a snapshot ideje volt a nev, ami ket bajt okozott: a mentett
    kep/felho nem feltetlenul ahhoz a pillanathoz tartozott, es az azonos
    idobelyeg utkozhetett. A frame-sorszam egyertelmu es szigoruan novekvo.
    """
    return f"{frame:09d}"


def count_existing(directory, ext):
    if not os.path.isdir(directory):
        return 0
    return sum(1 for f in os.listdir(directory) if f.endswith(ext))


def spawn_at_free_point(world, bp, spawn_points):
    for point in random.sample(spawn_points, len(spawn_points)):
        tf = carla.Transform(point.location + carla.Location(z=0.3), point.rotation)
        actor = world.try_spawn_actor(bp, tf)
        if actor is not None:
            return actor
    return None


def spawn_traffic(client, world, count):
    tm = client.get_trafficmanager()
    tm.set_global_distance_to_leading_vehicle(2.5)
    tm.set_hybrid_physics_mode(True)
    tm.set_hybrid_physics_radius(HYBRID_PHYSICS_RADIUS)

    blueprints = [bp for bp in world.get_blueprint_library().filter("vehicle.*")
                  if int(bp.get_attribute("number_of_wheels")) == 4]
    spawn_points = world.get_map().get_spawn_points()
    points = random.sample(spawn_points, len(spawn_points))

    def make_batch(pts):
        batch = []
        for point in pts:
            bp = random.choice(blueprints)
            if bp.has_attribute("color"):
                bp.set_attribute("color", random.choice(
                    bp.get_attribute("color").recommended_values))
            tf = carla.Transform(point.location + carla.Location(z=0.3), point.rotation)
            batch.append(carla.command.SpawnActor(bp, tf).then(
                carla.command.SetAutopilot(carla.command.FutureActor, True,
                                           tm.get_port())))
        return batch

    # Adagokban kuldjuk: a foglalt pontok hibat adnak, ezert `count` parancs
    # kevesebb autot eredmenyezne a kertnel.
    vehicles = []
    remaining = list(points)
    while remaining and len(vehicles) < count:
        take = min(count - len(vehicles), len(remaining))
        chunk, remaining = remaining[:take], remaining[take:]
        for response in client.apply_batch_sync(make_batch(chunk), True):
            if not response.error:
                vehicles.append(world.get_actor(response.actor_id))
    return vehicles


def set_autopilot(vehicle, tm, enabled):
    vehicle.set_autopilot(enabled, tm.get_port())
    if not enabled:
        return
    tm.vehicle_percentage_speed_difference(vehicle, 0.0)
    tm.set_desired_speed(vehicle, EGO_SPEED_KMH)
    tm.auto_lane_change(vehicle, True)
    # Lampak atlepese: allo autonal a kepek szinte azonosak lennenek.
    tm.ignore_lights_percentage(vehicle, 100)
    tm.distance_to_leading_vehicle(vehicle, 2.5)


class LaneWander(object):
    """Autopilot mellett: hullamzo oldaleltolas es idonkenti kiteres.
    Lasd LANE_OFFSET_* es PERTURB_*."""

    def __init__(self, tm):
        self.tm = tm
        self.reset(0.0)

    def reset(self, t):
        """Respawn vagy autopilot-valtas utan: tiszta lap."""
        self.offset = 0.0
        self.amp = 0.0
        self.period = 1.0
        self.wave_start_t = self.wave_end_t = t
        self.next_perturb_t = t + random.uniform(*PERTURB_EVERY)
        self.perturb_left = 0
        self.perturb_steer = 0.0

    @property
    def perturbing(self):
        return self.perturb_left > 0

    def step(self, vehicle, t):
        # Egy hullam 0-bol indul es 0-ban er veget, igy az uj amplitudora
        # valtas nem ugrik. Az elojel is veletlen: balra vagy jobbra kezd.
        # Kiteres alatt is halad, hogy utana ne egy regi ertekre alljon vissza.
        if t >= self.wave_end_t:
            self.amp = random.choice((-1.0, 1.0)) * random.uniform(*LANE_OFFSET_AMP)
            self.period = random.uniform(*LANE_OFFSET_PERIOD)
            self.wave_start_t = t
            self.wave_end_t = t + self.period
        self.offset = self.amp * np.sin(2.0 * np.pi * (t - self.wave_start_t) / self.period)

        if self.perturb_left > 0:
            self.perturb_left -= 1
            if self.perturb_left == 0:
                # Vissza az autopilotnak. A set_autopilot a TM beallitasait
                # is ujra rakja, az eltolast biztonsagbol mi is.
                set_autopilot(vehicle, self.tm, True)
                self.tm.vehicle_lane_offset(vehicle, self.offset)
            else:
                ctrl = vehicle.get_control()
                ctrl.steer = self.perturb_steer
                vehicle.apply_control(ctrl)
            return

        self.tm.vehicle_lane_offset(vehicle, self.offset)

        if t >= self.next_perturb_t:
            # A gazt az autopilot utolso parancsabol tartjuk, csak a kormany
            # a miénk.
            ctrl = vehicle.get_control()
            vehicle.set_autopilot(False, self.tm.get_port())
            self.perturb_steer = random.choice((-1.0, 1.0)) * random.uniform(*PERTURB_STEER)
            self.perturb_left = random.randint(*PERTURB_TICKS)
            ctrl.steer = self.perturb_steer
            vehicle.apply_control(ctrl)
            self.next_perturb_t = t + random.uniform(*PERTURB_EVERY)


class Ego(object):
    def __init__(self, world):
        self.world = world
        self.player = None
        self.camera = None
        self.lidar = None
        self.viewer = None
        self.rgb = None
        self.xyz = None
        self.raw_points = 0
        self.viewer_image = None
        # Melyik szimulacios frame-bol szarmazik a jelenlegi rgb/xyz par.
        self.frame = -1

        # A szenzorok ide kuldenek, a collect() innen szedi ossze a KERT
        # frame-hez tartozo merest. Igy a kamera es a lidar garantaltan
        # ugyanazt a pillanatot latja - a listen()-be kotott callbackekkel ez
        # nem volt garantalt, es a mintak fele duplikatum lett tole.
        self.camera_queue = queue.Queue()
        self.lidar_queue = queue.Queue()
        # A kulso nezet csak a kepernyore megy, ott nem szamit a szinkron.
        self.viewer_queue = queue.Queue()
        # Respawn utan az elso tick meg adat nelkul jon - nem hiba.
        self.just_spawned = False

        self.weather_presets = [
            (getattr(carla.WeatherParameters, n), n)
            for n in dir(carla.WeatherParameters)
            if n[0].isupper() and not n.startswith("_")
        ]
        self.weather_index = 0
        self.spawn()

    def spawn(self):
        self.destroy()
        # Az elozo auto adatai nem keveredhetnek az ujeval.
        self.rgb = self.xyz = self.viewer_image = None
        self.frame = -1
        # Az uj szenzorok csak a KOVETKEZO ticktol mernek, tehat az aktualis
        # tickre nem lesz adat. Ez nem hiba - a fociklus ezt a jelzot nezi,
        # hogy ne szamolja elveszettnek.
        self.just_spawned = True
        for q in (self.camera_queue, self.lidar_queue, self.viewer_queue):
            while not q.empty():
                q.get_nowait()

        bp = self.world.get_blueprint_library().find("vehicle.tesla.model3")
        bp.set_attribute("role_name", "hero")

        spawn_points = self.world.get_map().get_spawn_points()
        self.player = spawn_at_free_point(self.world, bp, spawn_points)
        for _ in range(5):
            if self.player is not None:
                break
            time.sleep(0.4)
            self.player = spawn_at_free_point(self.world, bp, spawn_points)
        if self.player is None:
            raise RuntimeError("Nem sikerult lerakni az egot - csokkentsd a "
                               "NUM_TRAFFIC erteket.")
        self._spawn_sensors()

    def _spawn_sensors(self):
        bl = self.world.get_blueprint_library()

        cam_bp = bl.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(CAM_WIDTH))
        cam_bp.set_attribute("image_size_y", str(CAM_HEIGHT))
        cam_bp.set_attribute("fov", str(CAM_FOV))
        self.camera = self.world.spawn_actor(
            cam_bp, SENSOR_TRANSFORMS["dashboard"], attach_to=self.player)
        self.camera.listen(self.camera_queue.put)

        lidar_bp = bl.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("channels", str(LIDAR_CHANNELS))
        lidar_bp.set_attribute("range", str(LIDAR_RANGE))
        lidar_bp.set_attribute("upper_fov", str(LIDAR_UPPER_FOV))
        lidar_bp.set_attribute("lower_fov", str(LIDAR_LOWER_FOV))
        lidar_bp.set_attribute("horizontal_fov", "360")
        lidar_bp.set_attribute("points_per_second", str(LIDAR_POINTS_PER_SEC))
        lidar_bp.set_attribute("rotation_frequency", str(LIDAR_ROTATION_HZ))
        # NINCS sensor_tick: a szenzorok minden tickben mernek, es a ritkitast
        # a fociklus vegzi (EVERY_N_TICK). A sensor_tick sajat, a szenzor
        # letrehozasatol szamolt fazissal fut, ami respawn utan elcsuszik a
        # fociklus ritmusatol - onnantol minden masodik meres elveszett.
        self.lidar = self.world.spawn_actor(
            lidar_bp, SENSOR_TRANSFORMS["lidar"], attach_to=self.player)
        self.lidar.listen(self.lidar_queue.put)

        # Kulso nezet, csak a kepernyore.
        view_bp = bl.find("sensor.camera.rgb")
        view_bp.set_attribute("image_size_x", str(WINDOW_WIDTH))
        view_bp.set_attribute("image_size_y", str(WINDOW_HEIGHT))
        self.viewer = self.world.spawn_actor(
            view_bp, SENSOR_TRANSFORMS["spectator"], attach_to=self.player)
        self.viewer.listen(self.viewer_queue.put)

    def collect(self, frame, timeout=0.2):
        """Megvarja a KERT frame-hez tartozo kamera- es lidaradatot.

        Szinkron modban a `world.tick()` utan mindket szenzor pontosan egy
        merest kuld, de NEM azonnal es nem sorrendben (halozaton jon at),
        ezert olvasunk a queue-bol a keresett frame-ig.

        Vissza: True, ha mindketto megerkezett.
        """
        try:
            image = _drain_until(self.camera_queue, frame, timeout)
            measurement = _drain_until(self.lidar_queue, frame, timeout)

            # A ket szenzor sensor_tick-je respawn utan elcsuszhat egymastol
            # (mindegyik a sajat letrehozasatol szamol), ezert a kesobbihez
            # igazitjuk a masikat.
            while image.frame < measurement.frame:
                image = _drain_until(self.camera_queue, measurement.frame, timeout)
            while measurement.frame < image.frame:
                measurement = _drain_until(self.lidar_queue, image.frame, timeout)
        except queue.Empty:
            return False

        self.rgb = carla_image_to_rgb(image)
        raw = carla_lidar_to_xyz(measurement)
        # A ritkitott felho megy MENTESRE es a BEV-re is: amit a kepernyon
        # latsz, pontosan az kerul a fajlba.
        self.xyz = voxel_downsample(raw)
        self.raw_points = len(raw)
        self.frame = measurement.frame

        return True

    def drain_viewer(self):
        """A kulso nezet legfrissebb kepe. Nem varunk ra: ha nincs uj, marad
        a regi. Kulonben a queue korlatlanul nooene."""
        while not self.viewer_queue.empty():
            self.viewer_image = carla_image_to_rgb(self.viewer_queue.get_nowait())

    def next_weather(self, reverse=False):
        step = -1 if reverse else 1
        self.weather_index = (self.weather_index + step) % len(self.weather_presets)
        preset, name = self.weather_presets[self.weather_index]
        self.world.set_weather(preset)
        return name

    def destroy(self):
        for sensor in (self.camera, self.lidar, self.viewer):
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        self.camera = self.lidar = self.viewer = None
        if self.player is not None:
            self.player.destroy()
            self.player = None


def main():
    os.makedirs(CAMERA_DIR, exist_ok=True)
    os.makedirs(LIDAR_DIR, exist_ok=True)
    existing = count_existing(CAMERA_DIR, ".png")
    if existing:
        print(f"[collect] mar van {existing} frame - az uj melle kerul")
    print(f"[collect] sav-eltolas {'BE' if WANDER else 'KI'}")

    pygame.init()
    pygame.font.init()
    display = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.DOUBLEBUF)
    pygame.display.set_caption("CARLA adatgyujtes - R: rogzites, P: autopilot, ESC: kilepes")
    font = pygame.font.Font(pygame.font.match_font("mono"), 18)

    client = carla.Client(HOST, PORT)
    client.set_timeout(20.0)
    world = client.load_world(TOWN)
    print(f"[collect] palya: {TOWN}")

    original_settings = world.get_settings()
    ego = None
    traffic = []
    saved_this_run = 0

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DELTA
        world.apply_settings(settings)

        tm = client.get_trafficmanager()
        tm.set_synchronous_mode(True)
        wander = LaneWander(tm) if WANDER else None

        # Az ego megy le eloszor: a hybrid physics kore epul, es a forgalom
        # kulonben elfoglalna minden spawn pontot.
        ego = Ego(world)

        print(f"[collect] forgalom lerakasa ({NUM_TRAFFIC} auto)...")
        traffic = spawn_traffic(client, world, NUM_TRAFFIC)
        print(f"[collect] {len(traffic)} auto elindult")

        autopilot = START_WITH_AUTOPILOT
        set_autopilot(ego.player, tm, autopilot)

        controller = VehicleController()
        throttle = brake = steer = 0.0

        recording = START_RECORDING
        tick_count = 0
        skipped = 0
        # A HUD panelek kepei. Csak uj szenzoradatnal szamoljuk ujra oket.
        range_img = range_rgb = bev_rgb = None
        range_fill = 0.0
        clock = pygame.time.Clock()
        last_report = time.time()
        running = True

        print(f"[collect] kesz. R = rogzites "
              f"({'FUT' if recording else 'SZUNET'})")

        while running:
            frame = world.tick()
            tick_count += 1

            # Minden tickben beolvassuk a szenzorokat - igy a queue-k nem
            # nonek, es a kep is folyamatos. Menteni csak EVERY_N_TICK-enkent
            # mentunk, lentebb.
            got_data = ego.collect(frame)
            if not got_data and not ego.just_spawned:
                skipped += 1
            if got_data:
                ego.just_spawned = False

            for event in pygame.event.get():
                controller.handle_event(event)
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYUP:
                    if event.key == K_ESCAPE:
                        running = False
                    elif event.key == K_r:
                        recording = not recording
                        print(f"\n[collect] rogzites "
                              f"{'INDUL' if recording else 'SZUNET'} "
                              f"({existing + saved_this_run} frame)")
                    elif event.key == K_p:
                        autopilot = not autopilot
                        set_autopilot(ego.player, tm, autopilot)
                        if wander is not None:
                            wander.reset(tick_count * FIXED_DELTA)
                        print(f"\n[collect] autopilot {'BE' if autopilot else 'KI'}")
                    elif event.key == K_c:
                        reverse = pygame.key.get_mods() & KMOD_SHIFT
                        print(f"\n[collect] idojaras: {ego.next_weather(reverse)}")
                    elif event.key == K_BACKSPACE:
                        ego.spawn()
                        set_autopilot(ego.player, tm, autopilot)
                        if wander is not None:
                            wander.reset(tick_count * FIXED_DELTA)
                        print("\n[collect] uj auto, uj pozicio")

            if not autopilot:
                throttle, brake, steer = controller.apply(ego.player, FIXED_DELTA)
            elif wander is not None:
                wander.step(ego.player, tick_count * FIXED_DELTA)

            # A `got_data` a tick elejen keszult, a respawn viszont utana is
            # johet (Backspace) - olyankor az ego adatai mar torolve vannak.
            has_data = ego.rgb is not None and ego.xyz is not None
            if recording and has_data and tick_count % EVERY_N_TICK == 0:
                # A nev a szimulacios frame sorszama, tehat a png es az npy
                # garantaltan ugyanahhoz a pillanathoz tartozik.
                name = frame_name(ego.frame)

                surface = pygame.surfarray.make_surface(ego.rgb.swapaxes(0, 1))
                surface = pygame.transform.smoothscale(surface,
                                                       (SAVE_WIDTH, SAVE_HEIGHT))
                pygame.image.save(surface, os.path.join(CAMERA_DIR, name + ".png"))
                np.save(os.path.join(LIDAR_DIR, name + ".npy"), ego.xyz)

                saved_this_run += 1

                if MAX_FRAMES is not None and saved_this_run >= MAX_FRAMES:
                    print(f"\n[collect] elertuk a {MAX_FRAMES} frame-et, leallas")
                    running = False

            # --- rajzolas ---
            # Csak minden DRAW_EVERY. tickben: a kepernyo tartalma a mentett
            # adatra nincs hatassal, a range image / BEV kiszamitasa viszont
            # ritkitas nelkuli felhon (65536 pont) ~6 ms, a teljes rajzolas
            # pedig ennek a tobbszorose. 4-es ertekkel a HUD 5 Hz-en frissul,
            # ami szemnek boven eleg.
            if recording and time.time() - last_report > 2.0:
                last_report = time.time()
                print(f"\r[collect] {existing + saved_this_run} frame | "
                      f"{saved_this_run} most | {clock.get_fps():.1f} FPS | "
                      f"{skipped} elveszett",
                      end="", flush=True)

            if tick_count % DRAW_EVERY:
                clock.tick(SENSOR_HZ if REALTIME else 0)
                continue

            ego.drain_viewer()

            # A range image csak akkor valtozik, ha uj lidaradat jott - a
            # koztes tickekben ujraszamolni felesleges (65536 ponton ~6 ms).
            # Respawn utan a panelek eltunnek, amig az uj auto adata meg nem
            # jon (ezert nullazzuk oket, ha nincs felho).
            if not has_data:
                range_img = range_rgb = bev_rgb = None
                range_fill = 0.0
            elif got_data or range_rgb is None:
                range_img = xyz_to_range_image(ego.xyz)
                range_fill = 100.0 * (range_img >= 0).mean()
                bev_rgb = xyz_to_bev_rgb(ego.xyz)
                range_rgb = range_image_to_rgb(range_img)

            if ego.viewer_image is not None:
                display.blit(pygame.surfarray.make_surface(
                    ego.viewer_image.swapaxes(0, 1)), (0, 0))
            else:
                display.fill((0, 0, 0))

            # A panelsav az ablak szelessegehez igazodik, hogy kisebb
            # felbontason se nojon ki a kepernyorol.
            pw = WINDOW_WIDTH // 6
            px = WINDOW_WIDTH - pw - 10
            py = 10
            border = (255, 60, 60) if recording else (90, 90, 90)

            def draw_panel(surf, label, height, ypos):
                display.blit(pygame.transform.scale(surf, (pw, height)), (px, ypos))
                pygame.draw.rect(display, border, (px, ypos, pw, height), 2)
                display.blit(font.render(label, True, (255, 255, 255)),
                             (px, ypos + height + 2))
                return ypos + height + 24

            if has_data:
                # Le-, majd visszanagyitva: igy azt latod, ami a fajlba kerul.
                small = pygame.transform.smoothscale(
                    pygame.surfarray.make_surface(ego.rgb.swapaxes(0, 1)),
                    (SAVE_WIDTH, SAVE_HEIGHT))
                py = draw_panel(small, f"mentett kamera ({SAVE_WIDTH}x{SAVE_HEIGHT})",
                                int(pw * SAVE_HEIGHT / SAVE_WIDTH), py)

            if has_data:
                label = f"lidar BEV - MENTETT ({len(ego.xyz)} pont"
                label += f", voxel {VOXEL_SIZE} m)" if VOXEL_SIZE > 0 else ")"
                py = draw_panel(pygame.surfarray.make_surface(bev_rgb.swapaxes(0, 1)),
                                label, pw, py)

                # A range image: EZT kapna a halo. A ritkitas hatasa ITT
                # latszik igazan - a BEV durvabb cellai elfedik a lyukakat.
                # A 16:1 arany miatt fuggolegesen nyujtva rajzoljuk.
                py = draw_panel(
                    pygame.surfarray.make_surface(range_rgb.swapaxes(0, 1)),
                    f"range image - EZEN TANUL ({range_fill:.0f}% kitoltott)",
                    90, py)

            speed = ego.player.get_velocity()
            kmh = 3.6 * (speed.x ** 2 + speed.y ** 2 + speed.z ** 2) ** 0.5
            lines = [
                "ROGZITES: FUT" if recording else "rogzites: szunet (R = indit)",
                f"Frame:        {existing + saved_this_run}  (+{saved_this_run} most)",
                f"Pontok:       {0 if ego.xyz is None else len(ego.xyz)}"
                + (f" / {ego.raw_points}  ({100.0 * len(ego.xyz) / ego.raw_points:.0f}%"
                   f", voxel {VOXEL_SIZE} m)"
                   if VOXEL_SIZE > 0 and ego.raw_points else ""),
                f"Meret:        {0 if ego.xyz is None else len(ego.xyz) * 12 / 1e6:.2f} MB/frame",
                f"Range kitolt: {range_fill:5.1f}%  (95% = ritkitas nelkul)",
                f"Elveszett:    {skipped:5d} tick  (szenzor timeout - "
                f"0 a normalis)",
                f"Sebesseg:     {kmh:5.1f} km/h",
                f"Vezetes:      {'autopilot' if autopilot else 'KEZI'}  (P = valt)",
                (f"Sav-eltolas:  {wander.offset:+.2f} m"
                 + ("  KITERES" if wander.perturbing else "")
                 if wander is not None and autopilot else "Sav-eltolas:  ki"),
                f"Frekvencia:   {SENSOR_HZ:.0f} Hz (sync)",
                f"Valos FPS:    {clock.get_fps():5.1f}",
            ]
            if not autopilot:
                lines += ["", f"Gaz {throttle:4.2f}  Fek {brake:4.2f}  "
                              f"Kormany {steer:+5.2f}"]
            for i, line in enumerate(lines):
                text = font.render(line, True,
                                   (255, 80, 80) if (i == 0 and recording)
                                   else (255, 255, 255))
                bg = pygame.Surface((text.get_width() + 8, text.get_height()))
                bg.set_alpha(140)
                bg.fill((0, 0, 0))
                display.blit(bg, (8, 8 + i * 22))
                display.blit(text, (12, 8 + i * 22))

            pygame.display.flip()
            clock.tick(SENSOR_HZ if REALTIME else 0)

    except KeyboardInterrupt:
        print("\n[collect] leallitas (Ctrl+C)")

    finally:
        meta = {
            "town": TOWN,
            "frames": count_existing(CAMERA_DIR, ".png"),
            # A MENTES frekvenciaja, nem a szenzore: minden EVERY_N_TICK-edik
            # tickben mentunk.
            "fps": SENSOR_HZ / EVERY_N_TICK,
            "sensor_hz": SENSOR_HZ,
            "every_n_tick": EVERY_N_TICK,
            "fixed_delta_seconds": FIXED_DELTA,
            "naming": "a fajlnev a szimulacios ido; az azonos nevu png es npy "
                      "egy pillanatban keszult",
            "camera": {
                "dir": "camera",
                "width": SAVE_WIDTH, "height": SAVE_HEIGHT,
                "rendered": [CAM_WIDTH, CAM_HEIGHT],
                "fov": CAM_FOV,
                "transform": "dashboard (x=1.6, z=1.7)",
                "format": "PNG RGB uint8",
            },
            "lidar": {
                "dir": "lidar",
                "channels": LIDAR_CHANNELS,
                "range": LIDAR_RANGE,
                "upper_fov": LIDAR_UPPER_FOV,
                "lower_fov": LIDAR_LOWER_FOV,
                "points_per_second": LIDAR_POINTS_PER_SEC,
                "rotation_frequency": LIDAR_ROTATION_HZ,
                "transform": "x=0.0, z=2.4",
                "format": "npy (N,3) float32 XYZ, meter",
                "voxel_size": VOXEL_SIZE,
                # Mar megforditva: kozvetlenul atadhato a
                # lidar_ae.points_to_range_image()-nek.
                "y_axis_flipped": True,
            },
            "wander": None if not WANDER else {
                "lane_offset_amp_m": LANE_OFFSET_AMP,
                "lane_offset_period_s": LANE_OFFSET_PERIOD,
                "perturb_every_s": PERTURB_EVERY,
                "perturb_steer": PERTURB_STEER,
                "perturb_ticks": PERTURB_TICKS,
            },
        }
        with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        if ego is not None:
            ego.destroy()
        if traffic:
            client.apply_batch([carla.command.DestroyActor(v) for v in traffic])
            print(f"\n[collect] {len(traffic)} forgalmi auto torolve")

        # Kotelezo: kulonben a szerver a tickjeinkre varna es minden mas
        # szkript lefagyna rajta.
        world.apply_settings(original_settings)
        client.get_trafficmanager().set_synchronous_mode(False)

        pygame.quit()
        print(f"[collect] kesz: +{saved_this_run} uj frame, osszesen "
              f"{count_existing(CAMERA_DIR, '.png')} a {OUT_DIR} mappaban")


if __name__ == "__main__":
    main()
