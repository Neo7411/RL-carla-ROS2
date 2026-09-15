"""LiDAR autoencoder tanitasa ELO CARLA adaton.
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
from carla_env.wrappers import Lidar, lidar_bev_to_rgb, sensor_transforms  # noqa: E402

from lidar_ae import LidarAE, points_to_range_image  # noqa: E402
from feature_extractions.manual_control import VehicleController  # noqa: E402

HOST = "127.0.0.1"
PORT = 2000
TOWN = "Town04"

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# Hany auto keruljon a palyara. None = ahany spawn pont van (Town04-en ~370),
# de az meg hybrid physics mellett is megfogja a szimulaciot.
#
# Ha akadozik, ezt vedd lejjebb (30-40 mar keves gepen is elmegy); ha birja,
# emeld feljebb. A hybrid physics (lasd spawn_traffic) a szamitast kimeli, de
# a memoriat es a renderelest nem.
NUM_TRAFFIC = 50

# Hybrid physics: ezen a sugaron belul (meter) van teljes kerekfizika a hero
# korul, azon kivul a TM olcso "teleportalos" modban mozgatja az autokat.
# 70 m bosegesen tobb a lidar 50 m-es hatotavjanal, tehat minden auto, amit a
# szenzor lat, valodi fizikaval mozog.
HYBRID_PHYSICS_RADIUS = 70.0
EGO_SPEED_KMH = 60.0
START_WITH_AUTOPILOT = True

# ---------------------------------------------------------------------------
# Range image parameterei - EZEN tanul a halo
# ---------------------------------------------------------------------------
# A range image a lidar sajat nezopontjabol keszul: 64 sor = elevacios szogek
# (+10 fok .. -25 fok), 1024 oszlop = 360 fok azimut korbe az auto korul. A
# cella erteke a TAVOLSAG, nem egy raszterezett folt - vagyis ez a pontfelho
# tomor, szinte vesztesegmentes reprezentacioja.
#
# Ezert tanul ezen a halo, es nem a BEV-en: a BEV felulnezet, ahol az uttest
# nagy resze ures (~3% kitoltottseg, a graf Stemje utan 11% tartalmas pont).
# A range image-ben minden sugarnak van merese: ~96% kitoltottseg, 97%
# tartalmas pont. A graf-encoder (EdgeConv) ilyen suru bemenetre valo.
#
# Ezeknek egyeznie KELL a wrappers.py Lidar blueprintjevel (channels=64,
# upper_fov=10, lower_fov=-25, range=50), kulonben a vetites rossz cellakba
# szorja a pontokat.
RANGE_SIZE = (64, 1024)
FOV = (10.0, -25.0)
DEPTH_RANGE = (1.0, 50.0)
# depth_scale: a log2(d+1) skalazott melyseget viszi [0,1]-be.
# log2(50+1) = 5.67, ezert 6 a biztonsagos felso hatar.
DEPTH_SCALE = 6.0
# A log skalazas fixen bent van a lidar_ae.process_scan()-ben (nincs kapcsolo):
# a kozeli tartomany felbontasa vezetes szempontjabol mindig fontosabb.

# A BEV kep csak a HUD-nak kell, hogy lasd, mi tortenik az auto korul.
BEV_SIZE = 256

# ---------------------------------------------------------------------------
# Tanitas
# ---------------------------------------------------------------------------
# BATCH_SIZE=4: 276 ms/lepes es 3.6 GB VRAM egy RTX 4070 Laptopon (8 GB).
# Nyolccal mar 7 GB lenne, ami a CARLA szerver mellett nem fer be.
BATCH_SIZE = 4
LEARNING_RATE = 1e-4

# A HUD "kozeli" hibametrikajanak hatara a normalizalt skalan. A
# process_scan log2 skalazasa miatt 0.333 pont 15 meternek felel meg -
# vezetes szempontjabol nagyjabol eddig terjed az erdekes zona.
NEAR_THRESHOLD = 0.333
BUFFER_SIZE = 512       # hany range image-et tartunk a memoriaban
LEARNING_STARTS = 64    # ennyi kep alatt meg csak gyujtunk, nem tanulunk
SAVE_EVERY_STEPS = 1000
CKPT_PATH = os.path.join(os.path.dirname(__file__), "lidar_ae.ckpt")

# A LidarAE-nek atadott halo-config. A z_channels/ch az encodert es a decodert
# is erinti, a ch_mult/strides/num_res_blocks csak a decodert.
#
# A strides indexelese CSAPDA (decoder.py:143):
#
#     stride = strides[i_level - 1] if i_level > 0 else None
#
# Vagyis az i_level=0 szinten NINCS upsample, es a listat eggyel eltolva
# olvassa. Ezert len(ch_mult)-1 darab upsample fut, es a strides UTOLSO eleme
# soha nem kerul felhasznalasra (dummy).
#
# A Stem fuggolegesen /4, vizszintesen /8: (64, 1024) -> (16, 128). Ezt kell
# pontosan visszaadni, ehhez HAROM upsample kell, tehat NEGY szint. A
# felhasznalasi sorrend forditott (i_level=3 veszi a strides[2]-t):
#
#     (16, 128) --(1,2)--> (16, 256) --(2,2)--> (32, 512) --(2,2)--> (64, 1024)
DDCONFIG = dict(
    in_channels=1,      # csak a range csatorna (remission nelkul)
    out_ch=1,
    z_channels=16,
    ch=64,
    ch_mult=(1, 2, 4, 4),
    strides=((2, 2), (2, 2), (1, 2), (1, 1)),
    num_res_blocks=1,
    dropout=0.0,
    k=20,               # hany szomszed a graf-retegekben
    tanh_out=True,      # a bemenet [-1, 1], a kimenet is oda keruljon
)


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
    # A spawn pontokat EGYSZER keverjuk meg, es mindegyiket legfeljebb egyszer
    # probaljuk. A regi kod pontonkent ujrakeverte a teljes listat, ami
    # sok autonal negyzetes koltseg lett volna.
    points = random.sample(spawn_points, len(spawn_points))
    if count is None:
        count = len(points)

    def make_batch(pts):
        """Spawn+autopilot parancsok egy adag spawn pontra."""
        batch = []
        for point in pts:
            bp = random.choice(blueprints)
            if bp.has_attribute("color"):
                bp.set_attribute("color", random.choice(
                    bp.get_attribute("color").recommended_values))
            # A spawn pont SAJAT z-jet hasznaljuk, kis rahagyassal - Town04 nem
            # sik, a feluljarokon a pontok 10+ meteren vannak.
            tf = carla.Transform(point.location + carla.Location(z=0.3), point.rotation)
            batch.append(carla.command.SpawnActor(bp, tf).then(
                carla.command.SetAutopilot(carla.command.FutureActor, True, tm.get_port())))
        return batch

    # Egy batchben kuldjuk a spawnt: sok kulon RPC hivas percekig tartana.
    #
    # A foglalt pontok (az ego, es amit a szimulacio mar elfoglalt) hibat adnak
    # vissza, ezert NEM eleg pontosan `count` parancsot kuldeni - ugy kevesebb
    # auto lenne a kertnel. Amig van meg nem probalt pont es hianyzik auto,
    # kuldunk egy ujabb adagot a maradekbol.
    vehicles = []
    remaining = list(points)
    while remaining and len(vehicles) < count:
        take = min(count - len(vehicles), len(remaining))
        chunk, remaining = remaining[:take], remaining[take:]
        for response in client.apply_batch_sync(make_batch(chunk), True):
            if not response.error:
                vehicles.append(world.get_actor(response.actor_id))

    return vehicles


def range_to_surface(img, width, height):
    v = np.clip((img[0] + 1.0) * 0.5, 0.0, 1.0)     # vissza [0, 1]-be
    rgb = np.zeros((v.shape[0], v.shape[1], 3), dtype=np.uint8)
    rgb[:, :, 0] = ((1.0 - v) * 255).astype(np.uint8)   # kozel -> piros
    rgb[:, :, 2] = (v * 255).astype(np.uint8)           # tavol -> kek
    rgb[v < 0.02] = 0                                    # ures cella -> fekete
    surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
    return pygame.transform.scale(surface, (width, height))


def bev_to_surface(bev, width, height):
    rgb = lidar_bev_to_rgb(bev)
    surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
    return pygame.transform.scale(surface, (width, height))


class Ego(object):
    """Az ego auto, a lidarja es a spectator kameraja."""

    def __init__(self, world, on_points, on_bev):
        self.world = world
        self.on_points = on_points   # nyers XYZ -> ezen tanul a halo
        self.on_bev = on_bev         # felulnezeti kep -> csak a HUD-nak
        self.player = None
        self.lidar = None
        self.camera = None
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
        self.lidar = Lidar(self.world, width=BEV_SIZE, height=BEV_SIZE,
                           transform=sensor_transforms["lidar"],
                           attach_to=self,
                           on_recv_points=self.on_points,
                           on_recv_image=self.on_bev)

        camera_bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(WINDOW_WIDTH))
        camera_bp.set_attribute("image_size_y", str(WINDOW_HEIGHT))
        self.camera = self.world.spawn_actor(
            camera_bp, sensor_transforms["spectator"], attach_to=self.player)
        self.camera.listen(self._on_camera)

    def _on_camera(self, image):
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
        if self.camera is not None:
            self.camera.stop()
            self.camera.destroy()
            self.camera = None
        if self.lidar is not None:
            self.lidar.destroy()
            self.lidar = None
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
    def __init__(self, device):
        self.device = device
        self.model = LidarAE(ddconfig=DDCONFIG,
                             learning_rate=LEARNING_RATE).to(device)
        self.optimizer = self.model.configure_optimizers()

        if os.path.exists(CKPT_PATH):
            self.model.init_from_ckpt(CKPT_PATH)
            print(f"[train] folytatas innen: {CKPT_PATH}")
        self.buffer = deque(maxlen=BUFFER_SIZE)
        self.lock = threading.Lock()
        self.save_lock = threading.Lock()

        self.enabled = True
        self.steps = 0
        self.last_loss = float("nan")
        self.last_occ_loss = float("nan")
        self.frames_seen = 0
        self.last_input = None
        self.last_recon = None

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def add(self, bev):
        with self.lock:
            self.buffer.append(bev)
        self.frames_seen += 1
        self.last_input = bev

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

        # Sima L1 rekonstrukcio - ahogy az eredeti TopoAutoencoder is hasznalja.
        #
        # A range image ~97%-ban kitoltott (minden lezersugar beleutkozik
        # valamibe), ezert itt NINCS szukseg a ritka kepeknel elkerulhetetlen
        # sulyozasra. BEV-en kellett: ott a kep csak 3%-ban volt kitoltott, es
        # sima L1 mellett a halonak megerte mindent uresre allitani.
        #
        # L1 es nem MSE: az L1 elesebb kimenetet ad, az MSE elmosna az
        # objektumok konturjait - range image-nel pont az elek (egy auto szele,
        # egy fal vege) hordozzak az informaciot.
        loss = torch.nn.functional.l1_loss(x, x_rec)

        self.optimizer.zero_grad()
        loss.backward()
        with self.save_lock:
            self.optimizer.step()

        self.steps += 1
        self.last_loss = float(loss.detach())
        # A kozeli cellakon vett hiba kulon. Ez a beszedesebb szam: a tavoli
        # hatter (falak, epuletek) konnyen tanulhato es dominalja az atlagot,
        # mikozben vezetes szempontjabol a kozeli objektumok szamitanak.
        with torch.no_grad():
            near = x < NEAR_THRESHOLD
            if near.any():
                self.last_occ_loss = float((x[near] - x_rec[near]).abs().mean())

    @torch.no_grad()
    def _reconstruct(self, bev):
        """
        Egy kep atengedese a halon, megjelenitesre.

        eval() mod: a decoder dropoutja tanito modban zajossa tenne a
        megjelenitett kepet.
        """
        self.model.eval()
        x = torch.from_numpy(bev).unsqueeze(0).to(self.device)
        x_rec = self.model(x)
        return x_rec[0].cpu().numpy()

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
        torch.save({"state_dict": state,
                    "ddconfig": DDCONFIG,
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
    # BEV kepek szinte azonosak lennenek, az pedig hasznalhatatlan
    # tanitoadat.
    tm.ignore_lights_percentage(v, 100)
    tm.distance_to_leading_vehicle(v, 2.5)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] eszkoz: {device}")

    pygame.init()
    pygame.font.init()
    display = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.DOUBLEBUF)
    pygame.display.set_caption("LiDAR AE tanitas - T: tanitas, P: autopilot, S: mentes, ESC: kilepes")

    trainer = Trainer(device)

    client = carla.Client(HOST, PORT)
    client.set_timeout(20.0)
    world_raw = client.load_world(TOWN)
    world = WorldShim(world_raw)
    hud = HUD(WINDOW_WIDTH, WINDOW_HEIGHT)
    print(f"[train] palya: {TOWN}")

    # A legutobbi range image (a halonak) es BEV kep (a HUD-nak). A lidar a
    # SAJAT szalan hivja a callbackeket, ezert itt csak lerakjuk oket, es a fo
    # ciklus veszi at - igy a vetites es a tanitas nem a szenzor szalat fogja.
    pending = {"range": None, "bev": None}

    def on_points(xyz):
        # A parametereket EXPLICIT adjuk at, nem hagyatkozunk a lidar_ae.py
        # alapertelmezeseire: igy ha itt atallitod a felbontast vagy a FOV-ot,
        # tenylegesen ervenyre jut, nem csendben marad a regi ertek.
        pending["range"] = points_to_range_image(
            xyz, size=RANGE_SIZE, fov=FOV,
            depth_range=DEPTH_RANGE, depth_scale=DEPTH_SCALE)

    def on_bev(bev):
        pending["bev"] = bev

    ego = None
    traffic = []
    try:
        # Az EGO megy le eloszor, csak utana a forgalom. Ket okbol:
        #  - a hybrid physics a hero kore rajzolja a teljes-fizika kort, tehat
        #    a hero-nak mar lennie kell, amikor a TM bekapcsolja
        #  - ha a forgalom eloszor foglalna el mind a ~370 spawn pontot, az
        #    egonak nem maradna hely
        ego = Ego(world, on_points, on_bev)
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

            # Uj lidar kep? -> bufferbe. A tanitas es a rekonstrukcio a
            # Trainer sajat szalan fut (GPU-n), itt csak atadjuk az adatot.
            range_img = pending["range"]
            if range_img is not None:
                pending["range"] = None
                trainer.add(range_img)

            # --- rajzolas ---
            if ego.viewer_image is not None:
                display.blit(pygame.surfarray.make_surface(
                    ego.viewer_image.swapaxes(0, 1)), (0, 0))
            else:
                display.fill((0, 0, 0))

            font = hud.font_mono

            # --- MINDEN a jobb felso sarokban, egy oszlopban ---
            #
            #   1. Felulnezet (BEV)      - ezen NEM tanul a halo, csak neked
            #                              mutatja, mi van az auto korul
            #   2. Range image (bemenet) - EZEN tanul
            #   3. AE rekonstrukcio      - amit a halo visszaad belole
            #
            # A 2. es 3. panel osszehasonlitasa mutatja a tanitas haladasat:
            # ahogy tanul, a kettonek egyre jobban hasonlitania kell.
            #
            # A range image 64x1024, vagyis 16:1 arányú csik - ezert fektetve,
            # kisebb magassaggal fer el a panel szelessegeben.
            pw = 360                               # panel szelesseg
            px_ = WINDOW_WIDTH - pw - 10           # bal szele
            # A range image valodi aranya 16:1, ami 360 px szelesnel 22 px
            # magas csik lenne - azon semmit nem lehet kivenni. Ezert
            # fuggolegesen nyujtva rajzoljuk (90 px): a sorok igy vastagabbak,
            # az alakzatok lathatoak. A torzitas itt nem baj, ez csak
            # megjelenites - a halo a valodi aranyu adatot kapja.
            rh = 90
            y = 10

            def label(text, ypos):
                display.blit(font.render(text, True, (255, 255, 255)), (px_, ypos))
                return ypos + 17

            if pending["bev"] is not None:
                y = label("Felulnezet (csak nezni)", y)
                display.blit(bev_to_surface(pending["bev"], pw, pw), (px_, y))
                pygame.draw.rect(display, (90, 90, 90), (px_, y, pw, pw), 1)
                y += pw + 10

            if trainer.last_input is not None:
                y = label("Range image (bemenet) - EZEN TANUL", y)
                display.blit(range_to_surface(trainer.last_input, pw, rh), (px_, y))
                pygame.draw.rect(display, (90, 90, 90), (px_, y, pw, rh), 1)
                y += rh + 10

                if trainer.last_recon is not None:
                    y = label("AE rekonstrukcio (decoded)", y)
                    display.blit(range_to_surface(trainer.last_recon, pw, rh), (px_, y))
                    pygame.draw.rect(display, (90, 90, 90), (px_, y, pw, rh), 1)

            # Autopilot alatt a kezi vezerles sorai csak zavarnanak - a TM
            # ertekei ugyis a HUD felso reszen latszanak.
            manual_lines = ([] if autopilot
                            else controller.hud_lines(throttle, brake, steer))

            hud.tick(world, clock)
            hud.render(display, extra_info=[
                "",
                "LiDAR AE",
                "Tanitas:      % 11s" % ("FUT" if trainer.enabled else "SZUNET"),
                "Lepes:        % 11d" % trainer.steps,
                "Loss (L1):    % 11.5f" % trainer.last_loss,
                # EZT erdemes nezni: a foglalt cellakon vett hiba. A teljes
                # loss akkor is csokken, ha a halo csak az ures hatteret
                # tanulta meg - ez a szam viszont csak akkor, ha tenyleg
                # eltalalja, hol vannak az objektumok.
                "Loss (kozeli):% 11.5f" % trainer.last_occ_loss,
                "Buffer:       % 8d/%d" % (len(trainer.buffer), BUFFER_SIZE),
                "Kepek:        % 11d" % trainer.frames_seen,
                # Lepes/kep arany: ha ez sokkal 1 alatt van, a halo jobban
                # lemarad az adatgyujtestol, vagyis sok kep ugy esik ki a
                # bufferbol, hogy egyszer sem tanult belole.
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
                      f"kozeli {trainer.last_occ_loss:.5f} | "
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
