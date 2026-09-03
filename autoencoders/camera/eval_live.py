"""Elo AE kiertekeles CARLA-ban.

Ugyanaz a felallas, mint a collect_dataset.py-ban (forgalom + ego autopilotban
korbemegy a palyan), csak itt nem mentunk kepet: minden dashcam kepet
atengedunk a betanitott autoencoderen, es egymas mellett mutatjuk az
EREDETI es a REKONSTRUALT kepet. Igy latszik, hogy eles futasban - mozgo
forgalomban, valtozo idojarasban - mit tart meg es mit veszit el a modell.

Ez a becsuletes teszt: a notebook validacios kepei ugyanabbol a gyujtesbol
valok, itt viszont friss, sosem latott kepeken fut a halo.

Billentyuk:
    W/A/S/D vagy nyilak - kezi vezetes (csak autopilot nelkul)
    SPACE - kezifek,  Q - hatramenet
    P - autopilot be/ki
    C / Shift+C - idojaras (ezzel lehet a legjobban "megtorni" az AE-t)
    V - nezet: egymas mellett / kulonbsegkep
    BACKSPACE - uj auto, uj pozicio
    ESC - kilepes
"""

import argparse
import os
import random
import time
import weakref
from collections import deque

import carla
import numpy as np
import pygame
import torch
from pygame.locals import (
    K_ESCAPE, K_SPACE, K_BACKSPACE, K_DOWN, K_LEFT, K_RIGHT, K_UP,
    K_a, K_c, K_d, K_p, K_q, K_s, K_v, K_w, KMOD_SHIFT,
)

from ae import Autoencoder

HOST = "127.0.0.1"
PORT = 2000
TOWN = "Town04"

DEFAULT_CKPT = os.path.join(os.path.dirname(__file__), "camera_ae.ckpt")

NUM_TRAFFIC = 50
EGO_SPEED_KMH = 80.0
START_WITH_AUTOPILOT = True

RENDER_FPS = 30
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# Ezeknek egyeznie kell azzal, amin az AE tanult (collect_dataset.py), kulonben
# nem azt merjuk, amit hiszunk: mas FOV/felbontas mellett a halo mas eloszlast
# lat, mint amire tanult.
OBS_WIDTH, OBS_HEIGHT = 160, 80
DASHCAM_TRANSFORM = carla.Transform(carla.Location(x=1.6, z=1.7))
DASHCAM_FOV = 110
SPECTATOR_TRANSFORM = carla.Transform(carla.Location(x=-5.5, z=2.8),
                                      carla.Rotation(pitch=-15))

# Az AE-t 30 FPS-en futtatni felesleges: az RL is ~10 Hz-en dontene, es igy
# marad GPU a szimulaciora.
AE_FPS = 10.0
MSE_HISTORY = 100        # ennyi utolso kep MSE-jebol megy a gorbe es az atlag

# A ket panel meretet az ablakbol szamoljuk, nem fix nagyitassal: fix 4x-nel
# a ket 160 pixeles kep a koztuk levo resszel egyutt 1304 pixel lenne, ami
# kilogna az 1280-as ablakbol.
PANEL_MARGIN = 20        # bal/jobb szel
PANEL_GAP = 24           # a ket panel kozott
PANEL_W = (WINDOW_WIDTH - 2 * PANEL_MARGIN - PANEL_GAP) // 2
PANEL_H = PANEL_W * OBS_HEIGHT // OBS_WIDTH       # az eredeti kepaanyt tartva
STRIP_W = 2 * PANEL_W + PANEL_GAP                 # az MSE gorbe / latent sav


def spawn_at_free_point(world, bp, spawn_points):
    """Lerakja a bp-t az elso szabad pontra. None, ha mind foglalt."""
    for point in random.sample(spawn_points, len(spawn_points)):
        # A spawn pont SAJAT z-jet hasznaljuk, csak egy kis rahagyassal - a
        # Town04 nem sik, fix z-vel a feluljarokon a terep ala kerulne az auto.
        tf = carla.Transform(
            point.location + carla.Location(z=0.3), point.rotation)
        actor = world.try_spawn_actor(bp, tf)
        if actor is not None:
            return actor
    return None


class Camera(object):
    """RGB kamera. A kepet a CARLA sajat szalan kapjuk meg."""

    def __init__(self, parent, width, height, transform, fov=None, fps=None):
        self.rgb = None
        self.frame_id = -1               # a legutobb kapott kep sorszama

        world = parent.get_world()
        bp = world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(width))
        bp.set_attribute("image_size_y", str(height))
        if fov is not None:
            bp.set_attribute("fov", str(fov))
        if fps is not None:
            # sensor_tick: szerveroldali ritkitas - amit igy nem kerunk, azt a
            # szerver el sem keszíti es at sem kuldi.
            bp.set_attribute("sensor_tick", str(1.0 / fps))

        self.sensor = world.spawn_actor(bp, transform, attach_to=parent)
        # weakref: eros referenciaval a szenzor sosem szabadulna fel.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: Camera._on_image(weak_self, image))

    @staticmethod
    def _on_image(weak_self, image):
        self = weak_self()
        if self is None:
            return
        # A CARLA bufferje BGRA. A copy() kell: a szerver ujrahasznositja a
        # kepbuffereket, tehat masolas nelkul a tartalom felulirodhat alattunk.
        bgra = np.reshape(np.frombuffer(image.raw_data, dtype=np.uint8),
                          (image.height, image.width, 4))
        self.rgb = bgra[:, :, :3][:, :, ::-1].copy()
        self.frame_id = image.frame

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


class AERunner(object):
    """A betanitott autoencoder: kep -> latent -> rekonstrukcio."""

    def __init__(self, ckpt_path, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # map_location: ha CUDA-n tanult es most CPU-n futtatnank, e nelkul elszall.
        self.model = Autoencoder.load_from_checkpoint(
            ckpt_path, map_location=self.device)
        # eval(): a BatchNorm maskepp viselkedik tanitas es kiertekeles kozben.
        # E nelkul egyetlen kepbol szamolna batch-statisztikat - az eredmeny
        # lathatoan rosszabb lenne, mint amit a halo tenylegesen tud.
        self.model.eval().to(self.device)

        self.latent_dim = self.model.hparams.latent_dim
        self.last_latent = None

    @torch.no_grad()
    def reconstruct(self, rgb_uint8):
        """(H,W,3) uint8 -> (rekonstrukcio uint8, MSE float).

        A normalizalas ugyanaz, mint a tanitasban: uint8/255, (H,W,C)->(C,H,W).
        Ha ez elcsuszna, a szam amit merunk nem lenne osszehasonlithato a
        notebook val_loss-aval.
        """
        x = torch.from_numpy(rgb_uint8.copy()).permute(2, 0, 1).float().div_(255.0)
        x = x.unsqueeze(0).to(self.device)          # (1, 3, H, W)

        z = self.model.encode(x)
        recon = self.model.decode(z)

        # Ugyanaz az MSE, amit a tanitas optimalizalt - igy a kiirt szam
        # kozvetlenul osszevetheto a checkpoint val_loss ertekevel.
        mse = float(torch.nn.functional.mse_loss(recon, x))

        self.last_latent = z[0].cpu().numpy()
        out = (recon[0].clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
        return out, mse


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

    def __init__(self, world):
        self.world = world
        self.player = None
        self.dashcam = None
        self.spectator = None

        self.weather_presets = [
            (getattr(carla.WeatherParameters, n), n)
            for n in dir(carla.WeatherParameters)
            if n[0].isupper() and not n.startswith("_")
        ]
        self.weather_index = 0
        self.weather_name = "alap"

        self.spawn()

    def spawn(self):
        """Auto letrehozasa/ujraletrehozasa veletlen szabad spawn pontban."""
        bp = self.world.get_blueprint_library().find("vehicle.tesla.model3")
        bp.set_attribute("role_name", "hero")
        if bp.has_attribute("color"):
            bp.set_attribute("color", random.choice(
                bp.get_attribute("color").recommended_values))

        self.destroy()
        self.player = spawn_at_free_point(
            self.world, bp, self.world.get_map().get_spawn_points())
        if self.player is None:
            raise RuntimeError("Nem sikerult lerakni az egot - minden spawn pont foglalt.")

        self.dashcam = Camera(self.player, OBS_WIDTH, OBS_HEIGHT,
                              DASHCAM_TRANSFORM, fov=DASHCAM_FOV, fps=AE_FPS)
        # A spectatoron NINCS sensor_tick: a kepernyokep akkor sima, ha minden
        # keszult kepet megkapunk.
        self.spectator = Camera(self.player, WINDOW_WIDTH, WINDOW_HEIGHT,
                                SPECTATOR_TRANSFORM)

    def next_weather(self, reverse=False):
        self.weather_index = (self.weather_index + (-1 if reverse else 1)) % len(self.weather_presets)
        preset = self.weather_presets[self.weather_index]
        self.world.set_weather(preset[0])
        self.weather_name = preset[1]
        return preset[1]

    def get_speed(self):
        """Sebesseg km/h-ban - ugyanaz a keplet, mint a wrappers.py-ban."""
        v = self.player.get_velocity()
        return 3.6 * (v.x ** 2 + v.y ** 2 + v.z ** 2) ** 0.5

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
        self._tm.set_desired_speed(v, EGO_SPEED_KMH)   # km/h, NEM m/s
        self._tm.auto_lane_change(v, True)
        # A lampakat atlepjuk, hogy ne alljon sokat - allo autonal mindig
        # ugyanazt a kepet latnank, abbol nem derul ki semmi az AE-rol.
        self._tm.ignore_lights_percentage(v, 100)
        self._tm.distance_to_leading_vehicle(v, 2.5)

    def parse_events(self, ego, hud, clock):
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
                print(f"\n[eval] autopilot {'BE' if self.autopilot else 'KI'}")

            elif event.key == K_c:
                reverse = pygame.key.get_mods() & KMOD_SHIFT
                name = ego.next_weather(reverse)
                # Az idojaras valtas eloszlas-valtas: a regi MSE ertekek mar
                # mas kepekre vonatkoznak, ezert tiszta lappal merunk tovabb.
                hud.reset_stats()
                print(f"\n[eval] idojaras: {name} (statisztika nullazva)")

            elif event.key == K_v:
                hud.show_diff = not hud.show_diff
                print(f"\n[eval] nezet: {'kulonbsegkep' if hud.show_diff else 'egymas mellett'}")

            elif event.key == K_BACKSPACE:
                ego.spawn()
                self.set_autopilot(ego)
                hud.reset_stats()
                print("\n[eval] uj auto, uj pozicio")

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
        # amig nyomva tartod. A lepeskoz masodperc-alapu, hogy fuggetlen legyen
        # a ciklus sebessegetol.
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


class HUD(object):
    """A kiertekeles megjelenitese: kepparok, MSE gorbe, latent savok."""

    def __init__(self, font, small_font):
        self.font = font
        self.small = small_font
        self.show_diff = False

        self.original = None      # (H,W,3) uint8
        self.recon = None
        self.latent = None
        self.mse_history = deque(maxlen=MSE_HISTORY)
        self.frames = 0
        self.mse_sum = 0.0        # a TELJES futasra vett atlaghoz
        self.ae_ms = 0.0

    def reset_stats(self):
        self.mse_history.clear()
        self.mse_sum = 0.0
        self.frames = 0

    def update(self, original, recon, latent, mse, ae_ms):
        self.original, self.recon, self.latent = original, recon, latent
        self.mse_history.append(mse)
        self.mse_sum += mse
        self.frames += 1
        self.ae_ms = ae_ms

    @staticmethod
    def _blit_image(display, rgb, pos, size, label, font, border):
        surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
        # scale (nem smoothscale): igy pixelesen latszik, mit adott vissza a
        # halo - a simitas eltakarna a rekonstrukcio hibait.
        display.blit(pygame.transform.scale(surface, size), pos)
        pygame.draw.rect(display, border, (pos[0] - 2, pos[1] - 2,
                                           size[0] + 4, size[1] + 4), 2)
        display.blit(font.render(label, True, (255, 255, 255), (0, 0, 0)),
                     (pos[0], pos[1] - 24))

    def _draw_panels(self, display):
        """Eredeti + rekonstrukcio (vagy kulonbsegkep) egymas mellett."""
        if self.original is None:
            return 0

        w, h = PANEL_W, PANEL_H
        top = 40
        left = PANEL_MARGIN

        self._blit_image(display, self.original, (left, top), (w, h),
                         "EREDETI (amit a kamera lat)", self.small, (80, 160, 240))

        if self.show_diff:
            # |eredeti - rekonstrukcio|, 4x felerositve. Igy latszik, HOL
            # hibazik a halo: jellemzoen a tavoli reszleteknel es az eleknel.
            diff = np.abs(self.original.astype(np.int16) -
                          self.recon.astype(np.int16))
            img = np.clip(diff * 4, 0, 255).astype(np.uint8)
            label, border = "KULONBSEG (|eredeti - rekon| x4)", (240, 200, 60)
        else:
            img = self.recon
            label, border = "REKONSTRUKCIO (amit az AE megtart)", (240, 120, 80)

        self._blit_image(display, img, (left + w + PANEL_GAP, top), (w, h),
                         label, self.small, border)
        return top + h

    def _draw_mse_curve(self, display, top):
        """Az utolso MSE_HISTORY kep hibaja idoben."""
        if len(self.mse_history) < 2:
            return top

        w, h = STRIP_W, 90
        left, top = PANEL_MARGIN, top + 46
        rect = pygame.Rect(left, top, w, h)
        pygame.draw.rect(display, (18, 18, 18), rect)
        pygame.draw.rect(display, (90, 90, 90), rect, 1)

        values = list(self.mse_history)
        # A skala mindig a lathato ablak maximumahoz igazodik - fix skalan a
        # 0.0005 koruli ertekek egy lapos vonalla olvadnanak.
        top_val = max(max(values), 1e-6)
        points = []
        for i, v in enumerate(values):
            px = int(left + i * w / (len(values) - 1))
            py = int(top + h - (float(v) / top_val) * (h - 8) - 4)
            points.append((px, py))
        pygame.draw.lines(display, (240, 120, 80), False, points, 2)

        display.blit(self.small.render(
            f"MSE / kep (max {top_val:.5f})", True, (200, 200, 200), (18, 18, 18)),
            (left + 6, top + 4))
        return top + h

    def _draw_latent(self, display, top):
        """A latent vektor komponensei savkent - ez megy majd az SAC-ba."""
        if self.latent is None:
            return

        w, h = STRIP_W, 70
        left, top = PANEL_MARGIN, top + 40
        rect = pygame.Rect(left, top, w, h)
        pygame.draw.rect(display, (18, 18, 18), rect)
        pygame.draw.rect(display, (90, 90, 90), rect, 1)

        z = self.latent
        mid = top + h // 2
        scale = max(float(np.abs(z).max()), 1e-6)
        bar_w = w / len(z)
        # A pygame.draw.rect nem fogad el numpy skalarokat a rect-ben (a z
        # float32 tomb), ezert minden koordinata explicit int.
        for i, v in enumerate(z):
            bar_h = int(round((float(v) / scale) * (h / 2 - 6)))
            color = (90, 190, 250) if bar_h >= 0 else (250, 130, 130)
            pygame.draw.rect(display, color, pygame.Rect(
                int(left + i * bar_w) + 1,
                mid - max(bar_h, 0),
                max(int(bar_w) - 1, 1),
                max(abs(bar_h), 1)))
        pygame.draw.line(display, (120, 120, 120), (left, mid), (left + w, mid), 1)
        display.blit(self.small.render(
            f"latent ({len(z)} dim, |max| {scale:.2f})", True,
            (200, 200, 200), (18, 18, 18)), (left + 6, top + 4))

    def draw(self, display, ego, controller, clock):
        bottom = self._draw_panels(display)
        bottom = self._draw_mse_curve(display, bottom)
        self._draw_latent(display, bottom)

        recent = np.mean(self.mse_history) if self.mse_history else 0.0
        overall = self.mse_sum / self.frames if self.frames else 0.0
        lines = [
            f"kep: {self.frames} | MSE most: {recent:.5f} | osszes atlag: {overall:.5f}",
            f"AE: {self.ae_ms:5.1f} ms/kep | {ego.get_speed():5.1f} km/h | "
            f"{'autopilot' if controller.autopilot else 'kezi'} | "
            f"idojaras: {ego.weather_name} | {clock.get_fps():.0f} FPS",
            "P: autopilot  C: idojaras  V: nezet  BACKSPACE: uj pozicio  ESC: kilepes",
        ]
        for i, line in enumerate(lines):
            display.blit(self.font.render(line, True, (255, 255, 255), (0, 0, 0)),
                         (20, WINDOW_HEIGHT - 76 + i * 24))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default=DEFAULT_CKPT,
                        help="a betanitott AE checkpointja")
    parser.add_argument("--town", default=TOWN)
    parser.add_argument("--traffic", type=int, default=NUM_TRAFFIC)
    parser.add_argument("--device", default=None, help="cuda vagy cpu")
    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        raise SystemExit(f"nincs ilyen checkpoint: {args.ckpt}")

    # Az AE-t a CARLA-hoz kapcsolodas ELOTT toltjuk be: ha a checkpoint hibas,
    # ne rakjunk le elotte 50 autot, amit aztan takaritani kell.
    print(f"[eval] modell betoltese: {args.ckpt}")
    ae = AERunner(args.ckpt, args.device)
    print(f"[eval] eszkoz: {ae.device} | latent_dim: {ae.latent_dim}")

    pygame.init()
    pygame.font.init()
    display = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.DOUBLEBUF)
    pygame.display.set_caption("CARLA - AE elo kiertekeles")
    font = pygame.font.Font(pygame.font.get_default_font(), 16)
    small = pygame.font.Font(pygame.font.get_default_font(), 13)

    client = carla.Client(HOST, PORT)
    client.set_timeout(20.0)
    world = client.load_world(args.town)
    print(f"[eval] palya: {args.town}")

    # Elore None/ures, mert a finally akkor is hivatkozik rajuk, ha a try mar
    # az elso soron elszall.
    ego = None
    traffic = []
    try:
        print(f"[eval] forgalom lerakasa ({args.traffic} auto)...")
        traffic = spawn_traffic(client, world, args.traffic)
        print(f"[eval] {len(traffic)} auto elindult autopilotban")

        ego = Ego(world)
        controller = KeyboardControl(ego, client.get_trafficmanager())
        hud = HUD(font, small)
        print("[eval] fut - ESC vagy Ctrl+C a leallitashoz\n")

        clock = pygame.time.Clock()
        last_ae_frame = -1
        last_report = time.time()

        while True:
            # A ciklus surubben porog, mint ahogy a kepek jonnek, hogy a
            # billentyuk azonnal reagaljanak.
            clock.tick(RENDER_FPS * 2)

            if controller.parse_events(ego, hud, clock):
                break

            # Csak akkor engedjuk at a halon, ha tenyleg UJ kep erkezett -
            # ugyanazt a kepet ujra kodolni csak GPU-t enne, es a statisztikat
            # is elhuzna (ugyanaz az MSE szazszor beszamitva).
            dash = ego.dashcam
            if dash.rgb is not None and dash.frame_id != last_ae_frame:
                last_ae_frame = dash.frame_id
                frame = dash.rgb
                t0 = time.perf_counter()
                recon, mse = ae.reconstruct(frame)
                hud.update(frame, recon, ae.last_latent, mse,
                           (time.perf_counter() - t0) * 1000.0)

            spectator = ego.spectator.get_surface()
            if spectator is not None:
                display.blit(spectator, (0, 0))
            else:
                display.fill((0, 0, 0))
            hud.draw(display, ego, controller, clock)
            pygame.display.flip()

            if time.time() - last_report > 2.0:
                last_report = time.time()
                recent = np.mean(hud.mse_history) if hud.mse_history else 0.0
                print(f"\r[eval] {hud.frames} kep | MSE most {recent:.5f} | "
                      f"atlag {hud.mse_sum / max(hud.frames, 1):.5f} | "
                      f"{hud.ae_ms:.1f} ms/kep", end="", flush=True)

    except KeyboardInterrupt:
        print("\n[eval] leallitas (Ctrl+C)")

    finally:
        if ego is not None:
            ego.destroy()
        if traffic:
            client.apply_batch([carla.command.DestroyActor(v) for v in traffic])
            print(f"\n[eval] {len(traffic)} forgalmi auto torolve")
        pygame.quit()


if __name__ == "__main__":
    main()
