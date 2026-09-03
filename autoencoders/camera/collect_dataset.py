"""CARLA dataset gyujto az autoencoder tanitasahoz.

Lerak forgalmat + egy ego autot, es masodpercenkent SAVES_PER_SECOND kepet
ment a dashcamrol.

Billentyuk:
    W/A/S/D vagy nyilak - kezi vezetes (csak autopilot nelkul)
    SPACE - kezifek,  Q - hatramenet
    P - autopilot be/ki
    R - felvetel szunet/inditas
    C / Shift+C - idojaras
    BACKSPACE - uj auto, uj pozicio
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
    K_ESCAPE, K_SPACE, K_BACKSPACE, K_DOWN, K_LEFT, K_RIGHT, K_UP,
    K_a, K_c, K_d, K_p, K_q, K_r, K_s, K_w, KMOD_SHIFT,
)

HOST = "127.0.0.1"
PORT = 2000
TOWN = "Town04"

DATASET_ROOT = os.path.join(os.path.dirname(__file__), "dataset")
TARGET_FRAMES = 20000
SAVES_PER_SECOND = 5.0

NUM_TRAFFIC = 50
EGO_SPEED_KMH = 70.0
START_WITH_AUTOPILOT = True

RENDER_FPS = 30
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
PREVIEW_SCALE = 2

# Ezeknek egyeznie kell az env kamerajaval (config.py OBS_RES,
# carla_env/wrappers.py sensor_transforms), kulonben az AE mas kepen tanul,
# mint amit eles futasban latni fog.
OBS_WIDTH, OBS_HEIGHT = 160, 80
DASHCAM_TRANSFORM = carla.Transform(carla.Location(x=1.6, z=1.7))
DASHCAM_FOV = 110
SPECTATOR_TRANSFORM = carla.Transform(carla.Location(x=-5.5, z=2.8),
                                      carla.Rotation(pitch=-15))


def spawn_at_free_point(world, bp, spawn_points):
    """Lerakja a bp-t az elso szabad pontra. None, ha mind foglalt."""
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
        # es eldobjuk a kepet. Igy a memoria nem tud elszallni - a regi
        # save_to_disk pont azert evett meg 26 GB-ot, mert a sora vegtelen volt.
        self._queue = None
        self._writer = None
        if output_dir is not None:
            # maxsize: a 160x80-as kepek 38 kB-osak, 128 db igy is csak ~5 MB.
            # A korlat a lenyeg, nem a merete - ezzel a memoria nem tud nonni.
            self._queue = queue.Queue(maxsize=128)
            self._writer = threading.Thread(target=self._write_loop, daemon=True)
            self._writer.start()

        world = parent.get_world()
        bp = world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(width))
        bp.set_attribute("image_size_y", str(height))
        if fov is not None:
            bp.set_attribute("fov", str(fov))
        if fps is not None:
            # sensor_tick = ennyi masodpercenkent kuldjon egy kepet. Ez
            # SZERVEROLDALI ritkitas: amit igy nem kerunk, azt a szerver el
            # sem kesziti es at sem kuldi.
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


def spawn_traffic(client, world, count):
    """Lerak `count` autot autopilotban. Visszaadja a listajukat."""
    tm = client.get_trafficmanager()
    tm.set_global_distance_to_leading_vehicle(2.5)

    # Csak negykerekuek: a biciklik/motorok lassuak es feltorlodnak.
    blueprints = [bp for bp in world.get_blueprint_library().filter("vehicle.*")
                  if int(bp.get_attribute("number_of_wheels")) == 4]
    spawn_points = world.get_map().get_spawn_points()

    vehicles = []
    for _ in range(count):
        bp = random.choice(blueprints)
        if bp.has_attribute("color"):
            bp.set_attribute("color", random.choice(
                bp.get_attribute("color").recommended_values))
        v = spawn_at_free_point(world, bp, spawn_points)
        if v is None:
            break
        v.set_autopilot(True, tm.get_port())
        vehicles.append(v)
    return vehicles


class Ego(object):
    """Az ego auto es a ket kameraja."""

    def __init__(self, world, frames_dir):
        self.world = world
        self.frames_dir = frames_dir
        self.player = None
        self.dashcam = None
        self.spectator = None

        self.weather_presets = [
            (getattr(carla.WeatherParameters, n), n)
            for n in dir(carla.WeatherParameters)
            if n[0].isupper() and not n.startswith("_")
        ]
        self.weather_index = 0

        self.spawn()

    def spawn(self):
        """Auto letrehozasa/ujraletrehozasa veletlen szabad spawn pontban."""
        bp = self.world.get_blueprint_library().find("vehicle.tesla.model3")
        bp.set_attribute("role_name", "hero")
        if bp.has_attribute("color"):
            bp.set_attribute("color", random.choice(
                bp.get_attribute("color").recommended_values))

        # A felvetel allapotat atmentjuk, kulonben a respawn ujrainditana a
        # szuneteltetett felvetelt.
        was_recording = self.dashcam.recording if self.dashcam else True
        old_dashcam = self.dashcam
        self.destroy()
        # A szamlalot a destroy() UTAN olvassuk: a mentoszal addig meg kiirja,
        # ami a soraban maradt, es azok a kepek is a datasetben vannak.
        frames_so_far = old_dashcam.frame_count if old_dashcam else 0

        self.player = spawn_at_free_point(
            self.world, bp, self.world.get_map().get_spawn_points())
        if self.player is None:
            raise RuntimeError("Nem sikerult lerakni az egot - minden spawn pont foglalt.")

        self.dashcam = Camera(self.player, OBS_WIDTH, OBS_HEIGHT,
                              DASHCAM_TRANSFORM, fov=DASHCAM_FOV,
                              output_dir=self.frames_dir,
                              fps=SAVES_PER_SECOND)
        self.dashcam.frame_count = frames_so_far
        self.dashcam.recording = was_recording
        # A spectatoron NINCS sensor_tick: a kepernyokep akkor sima, ha minden
        # keszult kepet megkapunk. Ritkitva a pygame ugyanazt a regi kepet
        # rajzolna ujra, mikozben az auto mar elmozdult - ettol szaggat.
        self.spectator = Camera(self.player, WINDOW_WIDTH, WINDOW_HEIGHT,
                                SPECTATOR_TRANSFORM)

    def next_weather(self, reverse=False):
        self.weather_index = (self.weather_index + (-1 if reverse else 1)) % len(self.weather_presets)
        preset = self.weather_presets[self.weather_index]
        self.world.set_weather(preset[0])
        return preset[1]

    def get_speed(self):
        """Sebesseg km/h-ban - ugyanaz a keplet, mint a wrappers.py-ban."""
        v = self.player.get_velocity()
        return 3.6 * (v.x ** 2 + v.y ** 2 + v.z ** 2) ** 0.5

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
        for cam in (self.dashcam, self.spectator):
            if cam is not None:
                cam.destroy()
        self.dashcam = self.spectator = None
        if self.player is not None:
            self.player.destroy()
            self.player = None


class KeyboardControl(object):
    def __init__(self, ego, tm):
        self._control = carla.VehicleControl()
        self._steer_cache = 0.0
        self._tm = tm                  # a respawnolt egot is be kell allitani
        self.autopilot = START_WITH_AUTOPILOT
        self.set_autopilot(ego)

    def set_autopilot(self, ego):
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

    def parse_events(self, ego, clock):
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
                print(f"\n[collect] idojaras: {ego.next_weather(reverse)}")

            elif event.key == K_BACKSPACE:
                ego.spawn()
                self.set_autopilot(ego)
                print("\n[collect] uj auto, uj pozicio")

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


def main():
    frames_dir = os.path.join(
        DATASET_ROOT, "session_" + time.strftime("%Y%m%d_%H%M%S"), "frames")
    os.makedirs(frames_dir)

    pygame.init()
    pygame.font.init()
    display = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.DOUBLEBUF)
    pygame.display.set_caption("CARLA AE dataset gyujto - R: felvetel, P: autopilot, ESC: kilepes")
    font = pygame.font.Font(pygame.font.get_default_font(), 18)

    client = carla.Client(HOST, PORT)
    client.set_timeout(20.0)
    world = client.load_world(TOWN)
    print(f"[collect] palya: {TOWN}")

    # Elore None/ures, mert a finally akkor is hivatkozik rajuk, ha a try mar
    # az elso soron elszall.
    ego = None
    traffic = []
    try:
        print(f"[collect] forgalom lerakasa ({NUM_TRAFFIC} auto)...")
        traffic = spawn_traffic(client, world, NUM_TRAFFIC)
        print(f"[collect] {len(traffic)} auto elindult autopilotban")

        ego = Ego(world, frames_dir)
        controller = KeyboardControl(ego, client.get_trafficmanager())
        print(f"[collect] session: {frames_dir}")
        print(f"[collect] cel: {TARGET_FRAMES} kep, {SAVES_PER_SECOND} kep/mp")
        print("[collect] Ctrl+C vagy ESC a leallitashoz\n")

        clock = pygame.time.Clock()
        last_report = time.time()
        label = font.render("indul...", True, (255, 255, 255), (0, 0, 0))

        last_drawn = -1
        while ego.dashcam.frame_count < TARGET_FRAMES:
            # A ciklus surubben porog, mint ahogy a kepek jonnek (RENDER_FPS
            # ketszerese), es a rajzolast az uj kep erkezese uttemezi - lentebb.
            #
            # Ha itt allna a RENDER_FPS korlat, ket gat lenne egymason: a tick
            # elaludna 33 ms-ot, es ha a kep epp az alvas utan erkezett, meg egy
            # kort varna - az effektiv frissites igy a fele lenne. Ettol
            # szaggatott.
            clock.tick(RENDER_FPS * 2)

            if controller.parse_events(ego, clock):
                break

            # Csak akkor rajzolunk ujra, ha tenyleg erkezett uj kep - a felesleges
            # blit csak CPU-t enne, a kep ugyanaz maradna.
            if ego.spectator.frame_id != last_drawn:
                last_drawn = ego.spectator.frame_id
                ego.render(display)

                display.blit(label, (16, WINDOW_HEIGHT - 34))
                pygame.display.flip()

            # A HUD feliratot csak masodpercenkent frissitjuk. A font.render()
            # es a get_speed() (ez utobbi RPC hivas a szerverhez) minden kepnel
            # lefutva feleslegesen lassitana a rajzolast.
            if time.time() - last_report > 1.0:
                last_report = time.time()
                status = "REC" if ego.dashcam.recording else "SZUNET"
                mode = "autopilot" if controller.autopilot else "kezi"
                label = font.render(
                    f"{status} | {ego.dashcam.frame_count}/{TARGET_FRAMES} kep | "
                    f"{mode} | {ego.get_speed():5.1f} km/h | "
                    f"{clock.get_fps():.0f} FPS", True, (255, 255, 255), (0, 0, 0))
                print(f"\r[collect] {ego.dashcam.frame_count}/{TARGET_FRAMES} kep",
                      end="", flush=True)

    except KeyboardInterrupt:
        print("\n[collect] leallitas (Ctrl+C)")

    finally:
        if ego is not None:
            ego.destroy()
        if traffic:
            client.apply_batch([carla.command.DestroyActor(v) for v in traffic])
            print(f"[collect] {len(traffic)} forgalmi auto torolve")
        pygame.quit()
        print(f"\n[collect] {len(os.listdir(frames_dir))} kep mentve ide: {frames_dir}")


if __name__ == "__main__":
    main()
