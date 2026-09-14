"""Kamera autoencoder tanitasa ELO CARLA adaton.

A feature_extractions/lidar/train.py mintajara. Ugyanaz a felepites: a CARLA
sajat utemben fut, az ego autopilotban jar, a kepek egy replay bufferbe
gyulnek, es egy KULON SZAL tanit beloluk.

Billentyuk:
    P     - autopilot be/ki (kikapcsolva TE vezetsz)
    T     - tanitas szunet/inditas
    C     - idojaras (Shift+C visszafele)
    BACKSPACE - uj auto, uj pozicio
    S     - checkpoint mentese most
    ESC   - kilepes (mentessel)

Kezi vezetes (P-vel kikapcsolt autopilot mellett):
    W/A/S/D vagy nyilak, SPACE kezifek, Q hatramenet
    PS5 kontroller: R2 gaz, L2 fek, bal kar kormany, X kezifek, kor hatramenet
    Reszletek: feature_extractions/manual_control.py
"""

import os
import random
import sys
import threading
import time
from collections import deque

import lightning as pl
import numpy as np
import pygame
import torch
from pygame.locals import (
    K_BACKSPACE, K_ESCAPE, K_c, K_p, K_s, K_t, KMOD_SHIFT,
)

# A repo gyokere a path-ra, hogy a carla_env importalhato legyen innen is.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import carla  # noqa: E402

from carla_env.tools.hud import HUD  # noqa: E402
from carla_env.wrappers import Camera, sensor_transforms  # noqa: E402

from camera_ae import CameraAutoEncoder  # noqa: E402
from feature_extractions.manual_control import VehicleController  # noqa: E402

HOST = "127.0.0.1"
PORT = 2000
TOWN = "Town04"

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

NUM_TRAFFIC = 60

# Hybrid physics: ezen a sugaron belul (meter) van teljes kerekfizika a hero
# korul, azon kivul a TM olcso "teleportalos" modban mozgatja az autokat.
HYBRID_PHYSICS_RADIUS = 70.0
EGO_SPEED_KMH = 60.0
START_WITH_AUTOPILOT = True

# ---------------------------------------------------------------------------
# A kep parameterei - EZEN tanul a halo
# ---------------------------------------------------------------------------
# Ennek PONTOSAN egyeznie kell a config.py OBS_RES ertekevel, kulonben az RL
# mas meretu kepet adna az AE-nek, mint amin tanult. A CameraAutoEncoder
# felepitese is erre a meretre van meretezve (4 db felezo conv: 80x160 ->
# 5x10), mas felbontasnal a fc_encode bemenete nem stimmelne.
OBS_WIDTH = 160
OBS_HEIGHT = 80

# A kamera pozicioja. A "dashboard" a menetirany szerinti elore nezo kep -
# ugyanez megy az RL-ben is (carla_route_env.py).
CAMERA_TRANSFORM = "dashboard"

# ---------------------------------------------------------------------------
# Tanitas
# ---------------------------------------------------------------------------
# A kamera AE lenyegesen olcsobb a lidarenal (nincs k-NN graf), ezert nagyobb
# batch is elfer: 32 kep 80x160-on par szaz MB VRAM.
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

# A latent merete. Ez kerul az RL observationbe, ezert a config.py LSIZE
# ertekevel kell egyeznie.
LATENT_DIM = 64
BASE_CHANNELS = 32
# A latens hatara: az encode() vegen tanh * latent_scale all, tehat a kimenet
# garantaltan -latent_scale .. +latent_scale. Az RL obs_space-ben ugyanez a
# hatar szerepel (train_rl.py: low=-4, high=4).
LATENT_SCALE = 4.0

BUFFER_SIZE = 2048      # hany kepet tartunk a memoriaban
LEARNING_STARTS = 256   # ennyi kep alatt meg csak gyujtunk, nem tanulunk
SAVE_EVERY_STEPS = 1000
CKPT_PATH = os.path.join(os.path.dirname(__file__), "camera_new_ae.ckpt")


def spawn_at_free_point(world, bp, spawn_points):
    """Lerakja a bp-t az elso szabad pontra. None, ha mind foglalt."""
    for point in random.sample(spawn_points, len(spawn_points)):
        tf = carla.Transform(point.location + carla.Location(z=0.3), point.rotation)
        actor = world.try_spawn_actor(bp, tf)
        if actor is not None:
            return actor
    return None


def spawn_traffic(client, world, count=None):
    tm = client.get_trafficmanager()
    tm.set_global_distance_to_leading_vehicle(2.5)

    tm.set_hybrid_physics_mode(True)
    tm.set_hybrid_physics_radius(HYBRID_PHYSICS_RADIUS)

    # Csak negykerekuek: a biciklik/motorok lassuak es feltorlodnak.
    blueprints = [bp for bp in world.get_blueprint_library().filter("vehicle.*")
                  if int(bp.get_attribute("number_of_wheels")) == 4]
    spawn_points = world.get_map().get_spawn_points()
    if count is None:
        count = len(spawn_points)
    points = random.sample(spawn_points, len(spawn_points))

    # Egy batchben kuldjuk a spawnt: sok kulon RPC hivas percekig tartana.
    batch = []
    for point, _ in zip(points, range(count)):
        bp = random.choice(blueprints)
        if bp.has_attribute("color"):
            bp.set_attribute("color", random.choice(
                bp.get_attribute("color").recommended_values))
        tf = carla.Transform(point.location + carla.Location(z=0.3), point.rotation)
        batch.append(carla.command.SpawnActor(bp, tf).then(
            carla.command.SetAutopilot(carla.command.FutureActor, True, tm.get_port())))

    vehicles = []
    for response in client.apply_batch_sync(batch, True):
        # A foglalt pontok hibat adnak vissza - ezeket csendben atlepjuk.
        if not response.error:
            vehicles.append(world.get_actor(response.actor_id))
    return vehicles


def image_to_tensor_array(image):
    """
    CARLA kamerakep -> (3, H, W) float32, [0, 1].

    A normalizalasnak EGYEZNIE kell a train_rl.py encode_state fuggvenyevel,
    kulonben az AE mas eloszlasu kepet latna elesben, mint tanitas kozben.
    Ott ez all: uint8 / 255, majd (H,W,C) -> (C,H,W).
    """
    arr = np.ascontiguousarray(image, dtype=np.uint8)
    return arr.transpose(2, 0, 1).astype(np.float32) / 255.0


def image_to_surface(chw, width, height):
    """(3, H, W) float [0,1] -> pygame surface."""
    rgb = (np.clip(chw, 0.0, 1.0) * 255).astype(np.uint8).transpose(1, 2, 0)
    surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
    return pygame.transform.scale(surface, (width, height))


class Ego(object):
    """Az ego auto, a tanitokameraja es a spectator kameraja."""

    def __init__(self, world, on_image):
        self.world = world
        self.on_image = on_image     # a kis felbontasu kep -> ezen tanul a halo
        self.player = None
        self.camera = None           # tanito kamera (OBS_WIDTH x OBS_HEIGHT)
        self.viewer_camera = None    # nagy kep a kepernyore
        self.viewer_image = None

        self.weather_presets = [
            (getattr(carla.WeatherParameters, n), n)
            for n in dir(carla.WeatherParameters)
            if n[0].isupper() and not n.startswith("_")
        ]
        self.weather_index = 0

        self.spawn()

    def spawn(self):
        """Auto letrehozasa/ujraletrehozasa veletlen szabad spawn pontban."""
        self.destroy()

        bp = self.world.get_blueprint_library().find("vehicle.tesla.model3")
        bp.set_attribute("role_name", "hero")

        self.player = spawn_at_free_point(
            self.world, bp, self.world.get_map().get_spawn_points())

        if self.player is None:
            for _ in range(5):
                time.sleep(0.4)
                self.player = spawn_at_free_point(
                    self.world, bp, self.world.get_map().get_spawn_points())
                if self.player is not None:
                    break

        if self.player is None:
            raise RuntimeError(
                "Nem sikerult lerakni az egot - minden spawn pont foglalt. "
                "Csokkentsd a NUM_TRAFFIC erteket.")

        # A tanito kamera a wrappers.Camera-n keresztul megy, nem kozvetlenul
        # a CARLA API-n: igy a motion blur kikapcsolasa es a BGR->RGB fordulat
        # PONTOSAN ugyanugy tortenik, mint az RL futasakor. Ha ezt itt kezzel
        # csinalnank, a ket ut elcsuszhatna egymastol.
        self.camera = Camera(self.world, OBS_WIDTH, OBS_HEIGHT,
                             transform=sensor_transforms[CAMERA_TRANSFORM],
                             attach_to=self,
                             on_recv_image=self.on_image)

        viewer_bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        viewer_bp.set_attribute("image_size_x", str(WINDOW_WIDTH))
        viewer_bp.set_attribute("image_size_y", str(WINDOW_HEIGHT))
        self.viewer_camera = self.world.spawn_actor(
            viewer_bp, sensor_transforms["spectator"], attach_to=self.player)
        self.viewer_camera.listen(self._on_viewer_image)

    def _on_viewer_image(self, image):
        bgra = np.reshape(np.frombuffer(image.raw_data, dtype=np.uint8),
                          (image.height, image.width, 4))
        self.viewer_image = bgra[:, :, :3][:, :, ::-1].copy()

    def get_carla_actor(self):
        return self.player

    def next_weather(self, reverse=False):
        self.weather_index = (self.weather_index + (-1 if reverse else 1)) % len(self.weather_presets)
        preset = self.weather_presets[self.weather_index]
        self.world.set_weather(preset[0])
        return preset[1]

    def destroy(self):
        if self.viewer_camera is not None:
            self.viewer_camera.stop()
            self.viewer_camera.destroy()
            self.viewer_camera = None
        if self.camera is not None:
            self.camera.destroy()
            self.camera = None
        if self.player is not None:
            self.player.destroy()
            self.player = None


class WorldShim(object):
    def __init__(self, carla_world):
        self._world = carla_world
        self.actor_list = []
        self.map = carla_world.get_map()

    def __getattr__(self, name):
        return getattr(self._world, name)


class Trainer(object):
    """
    Replay buffer + a halo tanitasa KULON SZALON.

    Ugyanaz a felepites, mint a lidar tanitoban: a fo szal csak rajzol es a
    pygame esemenyeket kezeli, a gradiens lepesek a hatterben futnak. A GIL
    nem gond, mert a PyTorch a CUDA hivasok idejere elengedi.
    """

    def __init__(self, device):
        self.device = device
        self.model = CameraAutoEncoder(latent_dim=LATENT_DIM,
                                       base_channels=BASE_CHANNELS,
                                       lr=LEARNING_RATE,
                                       latent_scale=LATENT_SCALE).to(device)
        self.optimizer = self.model.configure_optimizers()

        if os.path.exists(CKPT_PATH):
            self.load(CKPT_PATH)

        self.buffer = deque(maxlen=BUFFER_SIZE)
        self.lock = threading.Lock()
        self.save_lock = threading.Lock()

        self.enabled = True
        self.steps = 0
        self.last_loss = float("nan")
        self.frames_seen = 0
        self.last_input = None
        self.last_recon = None

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def load(self, path):
        """
        Folytatas checkpointbol.

        A Lightning sajat formatumat is elfogadja (load_from_checkpoint), de
        itt eleg a state_dict: a hiperparametereket a fenti konstansok adjak.
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            print(f"[train] hianyzo kulcsok: {missing}")
        if unexpected:
            print(f"[train] varatlan kulcsok: {unexpected}")
        print(f"[train] folytatas innen: {path}")

    def add(self, image):
        with self.lock:
            self.buffer.append(image)
        self.frames_seen += 1
        self.last_input = image

    def _loop(self):
        """A tanito szal: amig van mibol, tanul; kulonben var."""
        while not self._stop.is_set():
            if not self.enabled:
                time.sleep(0.05)
                continue

            with self.lock:
                ready = len(self.buffer) >= LEARNING_STARTS
                batch = (random.sample(self.buffer, min(BATCH_SIZE, len(self.buffer)))
                         if ready else None)
            if batch is None:
                # Meg gyujtunk. A sleep nelkul ez a szal feleslegesen porogne.
                time.sleep(0.05)
                continue

            self._train_step(batch)

            # Rekonstrukcio a HUD-hoz. Ugyanezen a szalon fut, mert a modellt
            # nem hasznalhatja kozben a fo szal: a train()/eval() mod valtas
            # kulonben osszeakadna.
            if self.last_input is not None:
                self.last_recon = self._reconstruct(self.last_input)

    def _train_step(self, batch):
        """Egy gradiens lepes a bufferbol vett veletlen batch-en."""
        x = torch.from_numpy(np.stack(batch)).to(self.device)

        self.model.train()
        x_rec = self.model(x)

        # MSE, ahogy a CameraAutoEncoder _shared_step-je is hasznalja. Itt
        # nincs ertelme L1-re valtani: a kamerakep suru es folytonos, nem
        # ritka, mint a BEV volt.
        loss = torch.nn.functional.mse_loss(x_rec, x)

        self.optimizer.zero_grad()
        loss.backward()
        with self.save_lock:
            self.optimizer.step()

        self.steps += 1
        self.last_loss = float(loss.detach())

    @torch.no_grad()
    def _reconstruct(self, image):
        """
        Egy kep atengedese a halon, megjelenitesre.

        eval() mod: a BatchNorm maskepp viselkedik tanitas es kiertekeles
        kozben, es egy elemu batch-nel tanito modban ertelmetlen statisztikat
        szamolna.
        """
        self.model.eval()
        x = torch.from_numpy(image).unsqueeze(0).to(self.device)
        return self.model(x)[0].cpu().numpy()

    @torch.no_grad()
    def latent_stats(self):
        """
        A latens tartomanya az utolso kepen. Ezt erdemes szemmel tartani: az
        RL obs_space -LATENT_SCALE..+LATENT_SCALE hatart hirdet, es a tanh
        miatt ezt nem is lehet tullepni. Ha viszont a latens VEGIG a hatar
        kozeleben all, az azt jelenti, hogy a tanh telitesben van, es a halo
        gyakorlatilag binarizalta a latenst - olyankor a LATENT_SCALE-t kell
        emelni.
        """
        if self.last_input is None:
            return float("nan"), float("nan")
        self.model.eval()
        x = torch.from_numpy(self.last_input).unsqueeze(0).to(self.device)
        z = self.model.encode(x)
        return float(z.min()), float(z.max())

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5.0)

    def save(self):
        # A save_lock a tanito szal sulyfrissiteset zarja ki a mentes idejere.
        # Nelkule a state_dict() masolas kozben futna egy optimizer.step(), es
        # a checkpointba fel-frissitett allapot kerulne (egyes retegek a lepes
        # elotti, masok a lepes utani sulyokkal).
        with self.save_lock:
            state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        # A train_rl.py a Lightning load_from_checkpoint-jat hasznalja, az
        # pedig HAROM kulcsot var. A "pytorch-lightning_version" nelkul
        # KeyError-ral szall el (a migracios logika keresi eloszor), ezert
        # nem eleg a state_dict + hyper_parameters.
        torch.save({"state_dict": state,
                    "hyper_parameters": dict(self.model.hparams),
                    "pytorch-lightning_version": pl.__version__,
                    "steps": self.steps}, CKPT_PATH)
        print(f"\n[train] mentve: {CKPT_PATH} ({self.steps} lepes)")


def set_autopilot(ego, tm, enabled):
    v = ego.player
    v.set_autopilot(enabled, tm.get_port())
    if not enabled:
        return
    # A 0 kioltja a TM globalis sebessegbeallitasat az egon, kulonben az
    # kerulhet a set_desired_speed ele.
    tm.vehicle_percentage_speed_difference(v, 0.0)
    tm.set_desired_speed(v, EGO_SPEED_KMH)
    tm.auto_lane_change(v, True)
    # A lampakat atlepjuk, hogy ne alljon sokat egy helyben - allo autonal a
    # kepek szinte azonosak lennenek, az pedig hasznalhatatlan tanitoadat.
    tm.ignore_lights_percentage(v, 100)
    tm.distance_to_leading_vehicle(v, 2.5)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] eszkoz: {device}")

    pygame.init()
    pygame.font.init()
    display = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.DOUBLEBUF)
    pygame.display.set_caption("Kamera AE tanitas - T: tanitas, P: autopilot, S: mentes, ESC: kilepes")

    trainer = Trainer(device)

    client = carla.Client(HOST, PORT)
    client.set_timeout(20.0)
    world_raw = client.load_world(TOWN)
    world = WorldShim(world_raw)
    hud = HUD(WINDOW_WIDTH, WINDOW_HEIGHT)
    print(f"[train] palya: {TOWN}")

    # A legutobbi kep. A kamera a SAJAT szalan hivja a callbacket, ezert itt
    # csak lerakjuk, es a fo ciklus veszi at - igy a normalizalas es a tanitas
    # nem a szenzor szalat fogja.
    pending = {"image": None}

    def on_image(image):
        pending["image"] = image_to_tensor_array(image)

    ego = None
    traffic = []
    try:
        # Az EGO megy le eloszor, csak utana a forgalom. Ket okbol:
        #  - a hybrid physics a hero kore rajzolja a teljes-fizika kort, tehat
        #    a hero-nak mar lennie kell, amikor a TM bekapcsolja
        #  - ha a forgalom eloszor foglalna el minden spawn pontot, az egonak
        #    nem maradna hely
        ego = Ego(world, on_image)
        hud.set_vehicle(ego.player)

        target = "minden szabad spawn pont" if NUM_TRAFFIC is None else f"{NUM_TRAFFIC} auto"
        print(f"[train] forgalom lerakasa ({target})...")
        traffic = spawn_traffic(client, world_raw, NUM_TRAFFIC)
        print(f"[train] {len(traffic)} auto elindult autopilotban "
              f"(hybrid physics: {HYBRID_PHYSICS_RADIUS:.0f} m sugarban teljes fizika)")

        tm = client.get_trafficmanager()
        autopilot = START_WITH_AUTOPILOT
        set_autopilot(ego, tm, autopilot)

        # Kezi vezetes. Kontroller ha van, kulonben billentyuzet - a modul
        # magatol donti el, es a konzolra kiirja, melyiket talalta.
        controller = VehicleController()
        # A kezi vezerles csak autopilot NELKUL ervenyesul: ha mindketto
        # egyszerre kuldene parancsot, a TM minden tickben feluliirna a tiedet.
        throttle = brake = steer = 0.0

        clock = pygame.time.Clock()
        last_report = time.time()
        next_save = SAVE_EVERY_STEPS
        running = True

        # A szerver aszinkron modban fut (nem mi tickeljuk): sajat utemben
        # szimulal, mi pedig annyi kepet dolgozunk fel, amennyit birunk. Ez
        # itt helyes - a tanitas nem befolyasolhatja a szimulacio sebesseget,
        # kulonben az auto lassitva vezetne, mikozben a halo szamol.
        while running:
            clock.tick(30)

            # dt a fokozatos kormanyzashoz: a billentyu nem ad analog erteket,
            # ezert az ido alapjan epitjuk fel a kiteres merteket.
            dt = clock.get_time() / 1000.0

            for event in pygame.event.get():
                # A kezi vezetes kapcsoloi (hatramenet, kontroller be/kihuzas)
                controller.handle_event(event)

                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYUP:
                    if event.key == K_ESCAPE:
                        running = False
                    elif event.key == K_p:
                        autopilot = not autopilot
                        set_autopilot(ego, tm, autopilot)
                        hud.notification(f"Autopilot {'BE' if autopilot else 'KI'}")
                    elif event.key == K_t:
                        trainer.enabled = not trainer.enabled
                        hud.notification(f"Tanitas {'FUT' if trainer.enabled else 'SZUNET'}")
                    elif event.key == K_c:
                        reverse = pygame.key.get_mods() & KMOD_SHIFT
                        hud.notification(f"Idojaras: {ego.next_weather(reverse)}")
                    elif event.key == K_BACKSPACE:
                        ego.spawn()
                        hud.set_vehicle(ego.player)
                        set_autopilot(ego, tm, autopilot)
                        hud.notification("Uj auto, uj pozicio")
                    elif event.key == K_s:
                        trainer.save()
                        hud.notification("Checkpoint mentve")

            # Kezi vezetes. Autopilot alatt nem nyulunk az autohoz: a TM
            # kuldi a parancsokat, es a ketto harcolna egymassal.
            if not autopilot:
                throttle, brake, steer = controller.apply(ego.player, dt)

            # Uj kep? -> bufferbe. A tanitas es a rekonstrukcio a Trainer
            # sajat szalan fut (GPU-n), itt csak atadjuk az adatot.
            image = pending["image"]
            if image is not None:
                pending["image"] = None
                trainer.add(image)

            # --- rajzolas ---
            if ego.viewer_image is not None:
                display.blit(pygame.surfarray.make_surface(
                    ego.viewer_image.swapaxes(0, 1)), (0, 0))
            else:
                display.fill((0, 0, 0))

            font = hud.font_mono

            # --- a ket panel a jobb felso sarokban ---
            #
            #   1. Bemenet        - EZEN tanul a halo (160x80)
            #   2. AE rekonstrukcio - amit visszaad belole
            #
            # A ketto osszehasonlitasa mutatja a tanitas haladasat.
            pw = 360                               # panel szelesseg
            ph = int(pw * OBS_HEIGHT / OBS_WIDTH)  # 2:1 arany megtartva
            px_ = WINDOW_WIDTH - pw - 10           # bal szele
            y = 10

            def label(text, ypos):
                display.blit(font.render(text, True, (255, 255, 255)), (px_, ypos))
                return ypos + 17

            if trainer.last_input is not None:
                y = label("Bemenet - EZEN TANUL", y)
                display.blit(image_to_surface(trainer.last_input, pw, ph), (px_, y))
                pygame.draw.rect(display, (90, 90, 90), (px_, y, pw, ph), 1)
                y += ph + 10

                if trainer.last_recon is not None:
                    y = label("AE rekonstrukcio (decoded)", y)
                    display.blit(image_to_surface(trainer.last_recon, pw, ph), (px_, y))
                    pygame.draw.rect(display, (90, 90, 90), (px_, y, pw, ph), 1)

            z_min, z_max = trainer.latent_stats()

            # Autopilot alatt a kezi vezerles sorai csak zavarnanak - a TM
            # ertekei ugyis a HUD felso reszen latszanak.
            manual_lines = ([] if autopilot
                            else controller.hud_lines(throttle, brake, steer))

            hud.tick(world, clock)
            hud.render(display, extra_info=[
                "",
                "Kamera AE",
                "Tanitas:      % 11s" % ("FUT" if trainer.enabled else "SZUNET"),
                "Lepes:        % 11d" % trainer.steps,
                "Loss (MSE):   % 11.5f" % trainer.last_loss,
                # A latens tartomanya. A tanh miatt sosem lephet ki a
                # +-LATENT_SCALE-bol; ha viszont VEGIG a hataron all, a tanh
                # telitesben van - olyankor emelni kell a LATENT_SCALE-t.
                "Latens min:   % 11.3f" % z_min,
                "Latens max:   % 11.3f" % z_max,
                "Buffer:       % 8d/%d" % (len(trainer.buffer), BUFFER_SIZE),
                "Kepek:        % 11d" % trainer.frames_seen,
                # Lepes/kep arany: ha ez sokkal 1 alatt van, a halo lemarad az
                # adatgyujtestol, vagyis sok kep ugy esik ki a bufferbol, hogy
                # egyszer sem tanult belole.
                "Lepes/kep:    % 11.2f" % (trainer.steps / max(1, trainer.frames_seen)),
            ] + manual_lines)
            pygame.display.flip()

            # A >= osszehasonlitas es nem a modulo: a tanito szal sajat
            # utemben lep, tehat a szamlalo ket rajzolas kozott tobbet is
            # ugorhat - a modulo igy atlephetne a pontos tobbszorost.
            if trainer.steps >= next_save:
                trainer.save()
                next_save = trainer.steps + SAVE_EVERY_STEPS

            if time.time() - last_report > 2.0:
                last_report = time.time()
                print(f"\r[train] lepes {trainer.steps} | loss {trainer.last_loss:.5f} | "
                      f"latens {z_min:.2f}..{z_max:.2f} | "
                      f"buffer {len(trainer.buffer)}/{BUFFER_SIZE} | "
                      f"kep {trainer.frames_seen}", end="", flush=True)

    except KeyboardInterrupt:
        print("\n[train] leallitas (Ctrl+C)")

    finally:
        # Eloszor a tanito szalat allitjuk le, csak utana mentunk: kulonben a
        # mentes kozben meg irna a sulyokat, es fel-frissitett allapot kerulne
        # a checkpointba.
        trainer.stop()
        trainer.save()
        if ego is not None:
            ego.destroy()
        if traffic:
            client.apply_batch([carla.command.DestroyActor(v) for v in traffic])
            print(f"[train] {len(traffic)} forgalmi auto torolve")
        pygame.quit()


if __name__ == "__main__":
    main()
