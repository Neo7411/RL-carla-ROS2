"""CARLA dataset gyujto v2 - ciklikus, ego-koruli forgalommal.

Kulonbseg a v1-hez kepest: ott a forgalom egyszer, a palya EGESZ teruleten
szetszorva kerult le, es az ego onalloan indult el. Egy nagy palyan (Town04)
ez azt jelentette, hogy az ego sokszor teljesen ures uton haladt - a dataset
tele lett ugyanolyan "csak aszfalt es korlat" kepekkel.

A v2 ciklusokban dolgozik. Egy ciklus:

    1. az ego lerakasa egy veletlen spawn pontba
    2. NUM_TRAFFIC auto lerakasa az EGO KORE (SPAWN_RADIUS-on belul)
    3. CYCLE_SECONDS masodperc vezetes autopilotban, kozben felvetel
    4. minden actor torlese, majd ujra az 1. ponttol egy MASIK helyen

Igy minden ciklus elejen garantaltan forgalom van a kamera elott, es
CYCLE_SECONDS-onkent uj kornyezetbe kerul az auto (varos, autopalya, alagut,
rezsu), ahelyett hogy percekig ugyanazon a szakaszon korozne.

A kepek MIND ugyanabba a session mappaba kerulnek, folyamatos szamlaloval -
a tanito notebook glob-ja igy valtozatlanul mukodik.

Billentyuk:
    W/A/S/D vagy nyilak - kezi vezetes (csak autopilot nelkul)
    SPACE - kezifek,  Q - hatramenet
    P - autopilot be/ki
    R - felvetel szunet/inditas
    C / Shift+C - idojaras
    N - a jelenlegi ciklus lezarasa, ugras a kovetkezore
    ESC - kilepes
"""

import os
import queue
import random
import threading
import time
import weakref

import carla
import cv2
import numpy as np
import pygame
from pygame.locals import (
    K_ESCAPE, K_SPACE, K_DOWN, K_LEFT, K_RIGHT, K_UP,
    K_a, K_c, K_d, K_n, K_p, K_q, K_r, K_s, K_w, KMOD_SHIFT,
)

HOST = "127.0.0.1"
PORT = 2000
TOWN = "Town04"

DATASET_ROOT = os.path.join(os.path.dirname(__file__), "dataset")
TARGET_FRAMES = 20000
SAVES_PER_SECOND = 5.0

# --- ciklus beallitasok -----------------------------------------------------
CYCLE_SECONDS = 120.0     # ennyi ideig fut egy ciklus (2 perc)
NUM_TRAFFIC = 60          # ennyi autot probalunk lerakni az ego kore

# Az ego koruli spawn gyuru. A belso sugar azert kell, hogy ne pont az ego
# nyakaba rakjunk autot (attol koccanas es beragadas lenne); a kulso azert,
# hogy a forgalom tenyleg a kamera latoterebe essen, ne a palya masik vegen.
SPAWN_RADIUS_MIN = 15.0
SPAWN_RADIUS_MAX = 120.0

# Tartalek sugar: ha a szuk gyuruben nem jott ossze a NUM_TRAFFIC letszam,
# ennyire tagitunk. Town04-en a gyuruben atlagosan ~56 spawn pont van, de a
# ritkabb szakaszokon (autopalya, alagut) ennel jóval kevesebb - ott e nelkul
# ures uton menne a felvetel.
SPAWN_RADIUS_FALLBACK = 250.0

# Ciklusonkent sorsoljon-e idojarast.
#
# ALAPBOL KI. A CARLA presetek kozott van HardRain es sur­u kod is - azokban
# a dashcam kep annyira kimosodik, hogy nem latszik sem az ut szele, sem a
# forgalom. Ilyen kepeken az AE-nek nincs mit tanulnia.
#
# Kikapcsolva a szerver AKTUALIS idojarasat hagyjuk bekeen: amit a CARLA
# inditasakor vagy kezzel (C gomb) beallitottal, az marad vegig.
RANDOM_WEATHER_PER_CYCLE = False

EGO_SPEED_KMH = 40.0
START_WITH_AUTOPILOT = True

RENDER_FPS = 30
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
PREVIEW_SCALE = 2

# Szinkron mod: a szimulacio NEM sajat tempoban fut, hanem minden lepest mi
# kerunk a world.tick()-kel. Ugyanaz a felallas, mint az RL env-ben
# (carla_env/envs/carla_route_env.py).
#
# Miert kell: aszinkron modban a szerver a sajat tempojaban lep, az idokoz
# ingadozik, es a kamera egy-egy hosszabb lepes alatt tobbet "lat" - ez maga
# is elmosast okoz, a motion bluron felul. Szinkronban minden lepes pontosan
# SIM_DELTA masodperc, tehat a kepek egyenletesek.
SYNCHRONOUS = True
SIM_FPS = 30.0
SIM_DELTA = 1.0 / SIM_FPS

# Ezeknek egyeznie kell az env kamerajaval (config.py OBS_RES,
# carla_env/wrappers.py sensor_transforms), kulonben az AE mas kepen tanul,
# mint amit eles futasban latni fog.
OBS_WIDTH, OBS_HEIGHT = 160, 80
DASHCAM_TRANSFORM = carla.Transform(carla.Location(x=1.6, z=1.7))
DASHCAM_FOV = 110
SPECTATOR_TRANSFORM = carla.Transform(carla.Location(x=-5.5, z=2.8),
                                      carla.Rotation(pitch=-15))


class Camera(object):
    """RGB kamera. A kepet a CARLA sajat szalan kapjuk meg."""

    def __init__(self, parent, width, height, transform, fov=None,
                 output_dir=None, fps=None):
        self.rgb = None
        self.output_dir = output_dir     # ha None, csak a kepernyore megy
        self.recording = True
        self.frame_count = 0
        self.frame_id = -1               # a legutobb kapott kep sorszama

        # A mentes NEM a szenzor szalan tortenik: a PNG tomorites kepenkent
        # 5-15 ms, es amig tart, a CARLA nem tud uj kepet leadni - ettol
        # szaggatna a kep. Helyette egy KORLATOS sorba tesszuk, amit egy kulon
        # szal ir ki. A korlat a lenyeg: ha a lemez nem birja, a sor megtelik
        # es eldobjuk a kepet. Igy a memoria nem tud elszallni.
        self._queue = None
        self._writer = None
        if output_dir is not None:
            self._queue = queue.Queue(maxsize=128)
            self._writer = threading.Thread(target=self._write_loop, daemon=True)
            self._writer.start()

        world = parent.get_world()
        bp = world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(width))
        bp.set_attribute("image_size_y", str(height))
        if fov is not None:
            bp.set_attribute("fov", str(fov))

        # Motion blur KI. A CARLA RGB kameraja alapertelmezetten elmossa a
        # kepet (motion_blur_intensity=0.45) - 70 km/h-nal ez keni el a sav
        # szelet es a tavoli autokat. Az elmosas MERTEKE a sebessegtol fugg,
        # tehat ugyanaz a hely mas kepet ad allva es haladva; az AE-nek ez
        # csak zaj, amit meg kell tanulnia atlagolni.
        #
        # UGYANEZ be van allitva a carla_env/wrappers.py Camera osztalyaban is.
        # A kettonek egyeznie KELL, kulonben az AE mas kepen tanul, mint amit
        # eles futasban latni fog.
        for attr, value in (("motion_blur_intensity", "0.0"),
                            ("motion_blur_max_distortion", "0.0"),
                            ("motion_blur_min_object_screen_size", "0.0")):
            if bp.has_attribute(attr):
                bp.set_attribute(attr, value)

        if fps is not None:
            # sensor_tick = ennyi masodpercenkent kuldjon egy kepet. Ez
            # SZERVEROLDALI ritkitas: amit igy nem kerunk, azt a szerver el
            # sem keszíti es at sem kuldi.
            bp.set_attribute("sensor_tick", str(1.0 / fps))

        self.sensor = world.spawn_actor(bp, transform, attach_to=parent)
        # weakref: eros referenciaval a szenzor sosem szabadulna fel.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: Camera._on_image(weak_self, image))

    def _write_loop(self):
        """A mentoszal: a sorbol veszi a kepeket es kiirja oket."""
        while True:
            item = self._queue.get()
            if item is None:        # leallitas jelzese
                return
            path, bgr = item
            cv2.imwrite(path, bgr)
            self.frame_count += 1

    @staticmethod
    def _on_image(weak_self, image):
        self = weak_self()
        if self is None:
            return

        # A CARLA bufferje BGRA. A copy() kell: a szerver ujrahasznositja a
        # kepbuffereket, tehat masolas nelkul a tartalom barmikor felulirodhat
        # alattunk - epp rajzolas kozben is.
        bgra = np.reshape(np.frombuffer(image.raw_data, dtype=np.uint8),
                          (image.height, image.width, 4))
        bgr = bgra[:, :, :3].copy()
        self.rgb = bgr[:, :, ::-1]
        # A fo ciklus ebbol tudja, hogy erkezett-e uj kep az elozo rajzolas ota.
        self.frame_id = image.frame

        if self._queue is None or not self.recording:
            return

        path = os.path.join(self.output_dir, "%08d.png" % image.frame)
        try:
            self._queue.put_nowait((path, bgr))
        except queue.Full:
            pass    # a lemez nem birja - inkabb kihagyjuk, mint hogy gyuljon

    def get_surface(self):
        """A legutobbi kep pygame surface-kent, vagy None."""
        rgb = self.rgb
        if rgb is None:
            return None
        # A pygame (W,H)-ben varja, a numpy (H,W)-ben tarolja.
        return pygame.surfarray.make_surface(rgb.swapaxes(0, 1))

    def destroy(self):
        self.sensor.stop()
        self.sensor.destroy()
        if self._queue is not None:
            # Megvarjuk, amig a mar sorban allo kepek kiirodnak.
            self._queue.put(None)
            self._writer.join(timeout=10.0)


def spawn_ego(world, spawn_points):
    """Lerakja az egot egy veletlen szabad pontra. None, ha mind foglalt."""
    bp = world.get_blueprint_library().find("vehicle.tesla.model3")
    bp.set_attribute("role_name", "hero")
    if bp.has_attribute("color"):
        bp.set_attribute("color", random.choice(
            bp.get_attribute("color").recommended_values))

    for point in random.sample(spawn_points, len(spawn_points)):
        # A spawn pont SAJAT z-jet hasznaljuk, csak egy kis rahagyassal.
        #
        # NE ird at fix ertekre: Town04 nem sik - a feluljarokon es a rampakon
        # a spawn pontok 10+ meteren vannak. Fix z=0.5-tel az auto a terep ALA
        # kerul, es a fizika kilokni probalja - ettol rangatozik es remeg.
        tf = carla.Transform(
            point.location + carla.Location(z=0.3), point.rotation)
        actor = world.try_spawn_actor(bp, tf)
        if actor is not None:
            return actor
    return None


def spawn_traffic_around(client, world, center, count):
    """`count` autot rak le a `center` korul, autopilotban.

    A palya SAJAT spawn pontjaibol valogatunk, nem talalomra sorsolt
    koordinatakbol: egy spawn pont garantaltan az uttesten van, helyes
    iranyba nezve. Egy magunk szamolt pont konnyen a szalagkorlaton kivul
    vagy szembe iranyba kerulne.
    """
    tm = client.get_trafficmanager()
    tm.set_global_distance_to_leading_vehicle(2.5)

    # Csak negykerekuek: a biciklik/motorok lassuak es feltorlodnak.
    blueprints = [bp for bp in world.get_blueprint_library().filter("vehicle.*")
                  if int(bp.get_attribute("number_of_wheels")) == 4]

    spawn_points = world.get_map().get_spawn_points()

    def ring(radius_max):
        """A `center` koruli gyuruben levo pontok, tavolsag szerint rendezve.

        Eloszor a kozelieket toltjuk fel, mert azok latszanak a kameran.
        """
        found = [(p.location.distance(center), p) for p in spawn_points
                 if SPAWN_RADIUS_MIN <= p.location.distance(center) <= radius_max]
        found.sort(key=lambda item: item[0])
        return found

    vehicles = []

    def fill_from(points):
        for _, point in points:
            if len(vehicles) >= count:
                return
            bp = random.choice(blueprints)
            if bp.has_attribute("color"):
                bp.set_attribute("color", random.choice(
                    bp.get_attribute("color").recommended_values))
            tf = carla.Transform(
                point.location + carla.Location(z=0.3), point.rotation)
            v = world.try_spawn_actor(bp, tf)
            if v is not None:
                v.set_autopilot(True, tm.get_port())
                vehicles.append(v)

    fill_from(ring(SPAWN_RADIUS_MAX))

    # Ha a szuk gyuruben nem jott ossze a letszam (a palya ritkabb szakaszain
    # kevesebb spawn pont van), tagitunk egyet. Inkabb legyen tavolabb par
    # auto, mint hogy ures uton menjen a felvetel.
    if len(vehicles) < count:
        fill_from(ring(SPAWN_RADIUS_FALLBACK))

    return vehicles


class Ego(object):
    """Az ego auto es a ket kameraja.

    A kamerak a ciklusok kozott UJRA lesznek hozva (uj autora kell felszerelni
    oket), a frame_count viszont atoroklodik - a szamlalo a teljes session-re
    vonatkozik, nem ciklusonkent indul ujra.
    """

    def __init__(self, world, frames_dir):
        self.world = world
        self.frames_dir = frames_dir
        self.player = None
        self.dashcam = None
        self.spectator = None
        self.frames_so_far = 0      # a korabbi ciklusokban mentett kepek
        self.recording = True       # atmentodik a ciklusok kozott

    def spawn(self):
        """Uj ego egy veletlen ponton, uj kamerakkal."""
        self.destroy()

        self.player = spawn_ego(self.world, self.world.get_map().get_spawn_points())
        if self.player is None:
            raise RuntimeError("Nem sikerult lerakni az egot - minden spawn pont foglalt.")

        self.dashcam = Camera(self.player, OBS_WIDTH, OBS_HEIGHT,
                              DASHCAM_TRANSFORM, fov=DASHCAM_FOV,
                              output_dir=self.frames_dir,
                              fps=SAVES_PER_SECOND)
        self.dashcam.frame_count = self.frames_so_far
        self.dashcam.recording = self.recording
        # A spectatoron NINCS sensor_tick: a kepernyokep akkor sima, ha minden
        # keszult kepet megkapunk. Ritkitva a pygame ugyanazt a regi kepet
        # rajzolna ujra, mikozben az auto mar elmozdult - ettol szaggat.
        self.spectator = Camera(self.player, WINDOW_WIDTH, WINDOW_HEIGHT,
                                SPECTATOR_TRANSFORM)
        return self.player.get_location()

    def get_speed(self):
        """Sebesseg km/h-ban - ugyanaz a keplet, mint a wrappers.py-ban."""
        v = self.player.get_velocity()
        return 3.6 * (v.x ** 2 + v.y ** 2 + v.z ** 2) ** 0.5

    @property
    def frame_count(self):
        return self.dashcam.frame_count if self.dashcam else self.frames_so_far

    def render(self, display):
        """Spectator kep a hatterben, a mentett dashcam kep jobb felul."""
        surface = self.spectator.get_surface()
        if surface is not None:
            display.blit(surface, (0, 0))
        else:
            display.fill((0, 0, 0))

        preview = self.dashcam.get_surface()
        if preview is not None:
            w, h = OBS_WIDTH * PREVIEW_SCALE, OBS_HEIGHT * PREVIEW_SCALE
            pos = (display.get_size()[0] - w - 10, 10)
            display.blit(pygame.transform.scale(preview, (w, h)), pos)
            border = (220, 40, 40) if self.dashcam.recording else (120, 120, 120)
            pygame.draw.rect(display, border, (pos[0] - 2, pos[1] - 2, w + 4, h + 4), 2)

    def destroy(self):
        # A szamlalot a kamera torlese ELOTT mentjuk at, de a frame_count-ot
        # a destroy() UTAN olvassuk: a mentoszal addig meg kiirja, ami a
        # soraban maradt, es azok a kepek is a datasetben vannak.
        if self.dashcam is not None:
            self.recording = self.dashcam.recording
        for cam in (self.dashcam, self.spectator):
            if cam is not None:
                cam.destroy()
        if self.dashcam is not None:
            self.frames_so_far = self.dashcam.frame_count
        self.dashcam = self.spectator = None
        if self.player is not None:
            self.player.destroy()
            self.player = None


class KeyboardControl(object):
    def __init__(self, tm):
        self._control = carla.VehicleControl()
        self._steer_cache = 0.0
        self._tm = tm
        self.autopilot = START_WITH_AUTOPILOT
        self.next_cycle = False     # az N gomb allitja be

    def set_autopilot(self, ego):
        """Minden uj ciklusban meg kell hivni: uj auto, uj beallitasok."""
        v = ego.player
        v.set_autopilot(self.autopilot, self._tm.get_port())
        if not self.autopilot:
            return
        # A 0 kioltja a TM globalis sebessegbeallitasat az egon, kulonben az
        # kerulhet a set_desired_speed ele.
        self._tm.vehicle_percentage_speed_difference(v, 0.0)
        # set_desired_speed km/h-t var, NEM m/s-ot.
        self._tm.set_desired_speed(v, EGO_SPEED_KMH)
        self._tm.auto_lane_change(v, True)
        # A lampakat atlepjuk, hogy ne alljon sokat. A tablakat es a tobbi
        # autot NEM: ignore_vehicles=100 mellett belehajt masokba, es Town04
        # rezsuin egy koccanas is levisz az uttestrol.
        self._tm.ignore_lights_percentage(v, 100)
        self._tm.distance_to_leading_vehicle(v, 2.5)

    def parse_events(self, ego, clock, weather):
        """True-t ad vissza, ha ki kell lepni."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type != pygame.KEYUP:
                continue

            if event.key == K_ESCAPE:
                return True

            elif event.key == K_p:
                self.autopilot = not self.autopilot
                self.set_autopilot(ego)
                print(f"\n[collect] autopilot {'BE' if self.autopilot else 'KI'}")

            elif event.key == K_r:
                ego.dashcam.recording = not ego.dashcam.recording
                print(f"\n[collect] felvetel {'FUT' if ego.dashcam.recording else 'SZUNET'}"
                      f" ({ego.dashcam.frame_count} kep)")

            elif event.key == K_c:
                reverse = pygame.key.get_mods() & KMOD_SHIFT
                print(f"\n[collect] idojaras: {weather.step(reverse)}")

            elif event.key == K_n:
                # A ciklust nem itt zarjuk le: a fo ciklus kezeli, hogy a
                # takaritas es az uj spawn egy helyen tortenjen.
                self.next_cycle = True
                print("\n[collect] ugras a kovetkezo ciklusra")

            elif event.key == K_q:
                self._control.gear = 1 if self._control.reverse else -1
                self._control.reverse = self._control.gear < 0

        # Autopilot alatt a TM vezet - a billentyuk felulirnak a parancsait.
        if not self.autopilot:
            self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
            ego.player.apply_control(self._control)
        return False

    def _parse_vehicle_keys(self, keys, milliseconds):
        self._control.throttle = min(self._control.throttle + 0.1, 1.0) \
            if (keys[K_UP] or keys[K_w]) else 0.0
        self._control.brake = min(self._control.brake + 0.2, 1.0) \
            if (keys[K_DOWN] or keys[K_s]) else 0.0

        # A _steer_cache "emlekszik" az elozo allapotra: folyamatosan fordul,
        # amig nyomva tartod, es visszaall kozepre, ha elengeded. A lepeskoz
        # masodperc-alapu, hogy fuggetlen legyen a ciklus sebessegetol.
        step = milliseconds / 1000.0
        if keys[K_LEFT] or keys[K_a]:
            self._steer_cache = 0.0 if self._steer_cache > 0 else self._steer_cache - step
        elif keys[K_RIGHT] or keys[K_d]:
            self._steer_cache = 0.0 if self._steer_cache < 0 else self._steer_cache + step
        else:
            self._steer_cache = 0.0

        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = self._steer_cache
        self._control.hand_brake = keys[K_SPACE]


class Weather(object):
    """Idojaras presetek - lepteteshez (C gomb) es veletlen valasztashoz."""

    def __init__(self, world):
        self.world = world
        self.presets = [
            (getattr(carla.WeatherParameters, n), n)
            for n in dir(carla.WeatherParameters)
            if n[0].isupper() and not n.startswith("_")
        ]
        self.index = 0
        # A szerver aktualis idojarasat NEM olvassuk ki es nem irjuk felul.
        # A `touched` jelzi, nyultunk-e hozza egyaltalan: amig False, a HUD
        # es a log nem ir ki idojaras nevet - nem tudnank, mi van beallitva.
        self.name = "szerver szerint"
        self.touched = False

    def step(self, reverse=False):
        self.index = (self.index + (-1 if reverse else 1)) % len(self.presets)
        return self._apply(self.index)

    def randomize(self):
        return self._apply(random.randrange(len(self.presets)))

    def _apply(self, index):
        self.index = index
        preset, name = self.presets[index]
        self.world.set_weather(preset)
        self.name = name
        self.touched = True
        return name


def main():
    frames_dir = os.path.join(
        DATASET_ROOT, "session_" + time.strftime("%Y%m%d_%H%M%S"), "frames")
    os.makedirs(frames_dir)

    pygame.init()
    pygame.font.init()
    display = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.DOUBLEBUF)
    pygame.display.set_caption(
        "CARLA dataset gyujto v2 - R: felvetel, P: autopilot, N: uj ciklus, ESC: kilepes")
    font = pygame.font.Font(pygame.font.get_default_font(), 18)

    client = carla.Client(HOST, PORT)
    client.set_timeout(20.0)
    world = client.load_world(TOWN)
    print(f"[collect] palya: {TOWN}")

    # Az EREDETI beallitasokat elmentjuk, mert a finally-ben vissza kell
    # allitani. Szinkron modban hagyott szerverhez a kovetkezo program (pl. a
    # CARLA sajat manual_control-ja) mar nem tudna csatlakozni: varna a
    # tick()-re, amit senki nem kuld, es ugy tunne, mintha lefagyott volna.
    original_settings = world.get_settings()
    traffic_manager = client.get_trafficmanager()

    if SYNCHRONOUS:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = SIM_DELTA
        world.apply_settings(settings)
        # A TM-et is szinkronba kell tenni, kulonben a sajat szalan probalna
        # vezetni az autokat, mikozben a vilag a mi tick()-unkre var - ettol
        # az autok rangatoznak vagy meg sem indulnak.
        traffic_manager.set_synchronous_mode(True)
        print(f"[collect] szinkron mod: {SIM_FPS:.0f} FPS ({SIM_DELTA*1000:.1f} ms/lepes)")

    # Elore None/ures, mert a finally akkor is hivatkozik rajuk, ha a try mar
    # az elso soron elszall.
    ego = None
    traffic = []
    try:
        weather = Weather(world)
        ego = Ego(world, frames_dir)
        controller = KeyboardControl(client.get_trafficmanager())

        print(f"[collect] session: {frames_dir}")
        print(f"[collect] cel: {TARGET_FRAMES} kep, {SAVES_PER_SECOND} kep/mp")
        print(f"[collect] ciklus: {CYCLE_SECONDS:.0f} mp, {NUM_TRAFFIC} auto az ego kore")
        print("[collect] Ctrl+C vagy ESC a leallitashoz\n")

        clock = pygame.time.Clock()
        label = font.render("indul...", True, (255, 255, 255), (0, 0, 0))
        last_report = time.time()
        last_drawn = -1

        cycle = 0
        cycle_deadline = 0.0
        quit_requested = False

        while ego.frame_count < TARGET_FRAMES and not quit_requested:
            # ---------------- uj ciklus indul ----------------
            cycle += 1
            # A regi forgalom torlese. Ezt a ciklus ELEJEN tesszuk, nem a
            # vegen: igy a kilepesi agakon (ESC, hiba) is csak egy helyen
            # kell takaritani - a finally-ben.
            #
            # A destroy() SZINKRON hivas (megvarja a szervert), az apply_batch
            # viszont csak bekuldi a parancsokat. Ez itt nem mindegy: ha a
            # regi autok meg a palyan allnak, amikor az ujakat probaljuk
            # lerakni, ELFOGLALJAK a spawn pontokat, es a try_spawn_actor
            # None-t ad vissza. Pont ettol lett "csak neha" forgalom az ego
            # korul. Ezert egyenkent, bevarva toroljuk oket.
            for v in traffic:
                if v.is_alive:
                    v.destroy()
            traffic = []

            center = ego.spawn()
            controller.set_autopilot(ego)
            if RANDOM_WEATHER_PER_CYCLE:
                weather.randomize()

            # Szinkron modban egy tick kell ahhoz, hogy a torles es az ego uj
            # pozicioja tenylegesen ervenyre jusson - csak utana szabad a
            # kornyezo pontokat szabadnak tekinteni.
            if SYNCHRONOUS:
                world.tick()

            traffic = spawn_traffic_around(client, world, center, NUM_TRAFFIC)

            # Szinkron modban a spawn csak a kovetkezo tick-en lep eletbe.
            # Tick nelkul az elso kepek meg az ELOZO allapotot mutatnak (ures
            # ut, vagy a regi pozicio), es azok is a datasetbe kerulnenek.
            # Par lepes arra is kell, hogy a fizika leultesse az autokat a
            # kerekeikre, es a TM elinditsa oket.
            if SYNCHRONOUS:
                for _ in range(10):
                    world.tick()

            shortfall = "" if len(traffic) >= NUM_TRAFFIC else f" (cel: {NUM_TRAFFIC})"
            print(f"\n[collect] {cycle}. ciklus | {len(traffic)} auto az ego kore{shortfall}"
                  + (f" | idojaras: {weather.name}" if weather.touched else ""))

            controller.next_cycle = False
            cycle_deadline = time.time() + CYCLE_SECONDS

            # ---------------- a ciklus fut ----------------
            while (time.time() < cycle_deadline
                   and not controller.next_cycle
                   and ego.frame_count < TARGET_FRAMES):
                if SYNCHRONOUS:
                    # Szinkron modban MI leptetjuk a vilagot, es a tick()
                    # blokkol, amig a szerver kesz - ez maga az utemezes,
                    # nincs szukseg kulon varakozasra. A clock.tick(0) csak
                    # az FPS-szamlalot frissiti.
                    world.tick()
                    clock.tick()
                else:
                    # Aszinkron modban a ciklus surubben porog, mint ahogy a
                    # kepek jonnek (RENDER_FPS ketszerese), es a rajzolast az
                    # uj kep erkezese utemezi.
                    #
                    # Ha itt allna a RENDER_FPS korlat, ket gat lenne egymason:
                    # a tick elaludna 33 ms-ot, es ha a kep epp az alvas utan
                    # erkezett, meg egy kort varna - az effektiv frissites igy
                    # a fele lenne. Ettol szaggatott.
                    clock.tick(RENDER_FPS * 2)

                if controller.parse_events(ego, clock, weather):
                    quit_requested = True
                    break

                # Csak akkor rajzolunk ujra, ha tenyleg erkezett uj kep - a
                # felesleges blit csak CPU-t enne, a kep ugyanaz maradna.
                if ego.spectator.frame_id != last_drawn:
                    last_drawn = ego.spectator.frame_id
                    ego.render(display)
                    display.blit(label, (16, WINDOW_HEIGHT - 34))
                    pygame.display.flip()

                # A HUD feliratot csak masodpercenkent frissitjuk. A
                # font.render() es a get_speed() (ez utobbi RPC hivas a
                # szerverhez) minden kepnel lefutva feleslegesen lassitana.
                if time.time() - last_report > 1.0:
                    last_report = time.time()
                    status = "REC" if ego.dashcam.recording else "SZUNET"
                    mode = "autopilot" if controller.autopilot else "kezi"
                    left = max(0.0, cycle_deadline - time.time())
                    label = font.render(
                        f"{status} | {ego.frame_count}/{TARGET_FRAMES} kep | "
                        f"{cycle}. ciklus {left:4.0f} mp | {len(traffic)} auto | "
                        f"{mode} | {ego.get_speed():5.1f} km/h | "
                        f"{weather.name} | {clock.get_fps():.0f} FPS",
                        True, (255, 255, 255), (0, 0, 0))
                    print(f"\r[collect] {ego.frame_count}/{TARGET_FRAMES} kep | "
                          f"{cycle}. ciklus, {left:4.0f} mp",
                          end="", flush=True)

    except KeyboardInterrupt:
        print("\n[collect] leallitas (Ctrl+C)")

    finally:
        if ego is not None:
            ego.destroy()
        if traffic:
            client.apply_batch([carla.command.DestroyActor(v) for v in traffic])
            print(f"\n[collect] {len(traffic)} forgalmi auto torolve")

        # A szerver visszaallitasa aszinkron modba. E NELKUL a szerver
        # szinkron modban ragad: a kovetkezo program (vagy a CARLA sajat
        # manual_control-ja) a tick()-re varna, amit senki nem kuld, es
        # lefagyottnak tunne. Ezert van a torles UTAN, de a pygame.quit()
        # ELOTT - hogy meg egy kesobbi hiba se hagyja ki.
        if SYNCHRONOUS:
            traffic_manager.set_synchronous_mode(False)
            world.apply_settings(original_settings)
            print("[collect] szerver visszaallitva aszinkron modba")

        pygame.quit()
        print(f"\n[collect] {len(os.listdir(frames_dir))} kep mentve ide: {frames_dir}")


if __name__ == "__main__":
    main()
