import carla
import numpy as np
import weakref


def get_actor_display_name(actor, truncate=250):
    name = " ".join(actor.type_id.replace("_", ".").title().split(".")[1:])
    return (name[:truncate - 1] + u"\u2026") if len(name) > truncate else name


def get_displacement_vector(car_pos, waypoint_pos, theta):
    """
    Calculates the displacement vector from the car to a waypoint, taking into account the orientation of the car.

    Parameters:
        car_pos (numpy.ndarray): 1D numpy array of shape (3,) representing the x, y, z coordinates of the car.
        waypoint_pos (numpy.ndarray): 1D numpy array of shape (3,) representing the x, y, z coordinates of the waypoint.
        theta (float): Angle in radians representing the orientation of the car.

    Returns:
        numpy.ndarray: 1D numpy array of shape (3,) representing the displacement vector from the car to the waypoint,
        with the car as the origin and the y-axis pointing in the direction of the car's orientation.
    """
    # Calculate the relative position of the waypoint with respect to the car
    relative_pos = waypoint_pos - car_pos

    theta = theta
    # Construct the rotation transformation matrix
    R = np.array([[np.cos(theta), np.sin(theta), 0],
                  [-np.sin(theta), np.cos(theta), 0],
                  [0, 0, 1]])
    T = np.array([[0, 1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])
    # Apply the rotation matrix to the relative position vector
    waypoint_car = R @ relative_pos
    waypoint_car = T @ waypoint_car
    # Set values very close to zero to exactly zero
    waypoint_car[np.abs(waypoint_car) < 10e-10] = 0

    return waypoint_car


def angle_diff(v0, v1):
    """
    Calculates the signed angle difference between 2D vectors v0 and v1.
    It returns the angle difference in radians between v0 and v1.
    The v0 is the reference for the sign of the angle
    """
    v0_xy = v0[:2]
    v1_xy = v1[:2]
    v0_xy_norm = np.linalg.norm(v0_xy)
    v1_xy_norm = np.linalg.norm(v1_xy)
    if v0_xy_norm == 0 or v1_xy_norm == 0:
        return 0

    v0_xy_u = v0_xy / v0_xy_norm
    v1_xy_u = v1_xy / v1_xy_norm
    # A clip NEM kozmetikai: ket egysegvektor skalarszorzata elvileg [-1, 1],
    # de a normalizalas lebegopontos kerekitese miatt lehet 1.0000000000000002,
    # es az arccos ott NaN-t ad. Ez majdnem parhuzamos vektoroknal fordul elo,
    # vagyis EGYENESEN HALADVA - a tipikus esetben. Az igy kapott NaN vegigment
    # a rewardon a replay bufferbe, es a SAC ott szallt el
    # ("Expected parameter loc ... to satisfy the constraint Real()").
    dot_product = np.clip(np.dot(v0_xy_u, v1_xy_u), -1.0, 1.0)
    angle = np.arccos(dot_product)

    # Calculate the sign of the angle using the cross product
    cross_product = np.cross(v0_xy_u, v1_xy_u)
    if cross_product < 0:
        angle = -angle
    if abs(angle) >= 2.3:
        return 0
    return round(angle, 2)


def distance_to_line(A, B, p):
    p[2] = 0
    num = np.linalg.norm(np.cross(B - A, A - p))
    denom = np.linalg.norm(B - A)
    if np.isclose(denom, 0):
        return np.linalg.norm(p - A)
    return num / denom


def vector(v):
    """ Turn carla Location/Vector3D/Rotation to np.array """
    if isinstance(v, carla.Location) or isinstance(v, carla.Vector3D):
        return np.array([v.x, v.y, v.z])
    elif isinstance(v, carla.Rotation):
        return np.array([v.pitch, v.yaw, v.roll])


def smooth_action(old_value, new_value, smooth_factor):
    return old_value * smooth_factor + new_value * (1.0 - smooth_factor)


def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K


def get_image_point(loc, K, w2c):
    # Calculate 2D projection of 3D coordinate

    # Format the input coordinate (loc is a carla.Position object)
    point = np.array([loc.x, loc.y, loc.z, 1])
    # transform to camera coordinates
    point_camera = np.dot(w2c, point)

    # New we must change from UE4's coordinate system to an "standard"
    # (x, y ,z) -> (y, -z, x)
    # and we remove the fourth componebonent also
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

    # now project 3D->2D using the camera matrix
    point_img = np.dot(K, point_camera)
    # normalize
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]

    return point_img[0:2].astype(int)


sensor_transforms = {
    "spectator": carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
    "dashboard": carla.Transform(carla.Location(x=1.6, z=1.7)),
    "lidar": carla.Transform(carla.Location(x=0.0, z=2.4)),
    "birdview": carla.Transform(carla.Location(x=90, y=210, z=175), carla.Rotation(pitch=-90))
}


# ===============================================================================
# CarlaActorBase
# ===============================================================================

class CarlaActorBase(object):
    def __init__(self, world, actor):
        self.world = world
        self.actor = actor
        self.world.actor_list.append(self)
        self.destroyed = False

    def destroy(self):
        if self.destroyed:
            raise Exception("Actor already destroyed.")
        else:
            print("Destroying ", self, "...")
            self.actor.destroy()
            self.world.actor_list.remove(self)
            self.destroyed = True

    def get_carla_actor(self):
        return self.actor

    def tick(self):
        pass

    def __getattr__(self, name):
        """Relay missing methods to underlying carla actor"""
        return getattr(self.actor, name)


# ===============================================================================
# Lidar
# ===============================================================================

def lidar_bev_to_rgb(bev):
    """
    Magassag-kodolt BEV (H, W) float [0,1] -> (H, W, 3) uint8, megjelenitesre.

    Szinskala szurkearnyalat helyett: a magassag igy szabad szemmel is
    leolvashato. Az ures cellak feketek maradnak, kulonben a "nincs itt semmi"
    ugyanugy nezne ki, mint a talajszintu pont.
    """
    rgb = np.zeros((bev.shape[0], bev.shape[1], 3), dtype=np.uint8)
    occupied = bev > 0.0
    # alacsony -> zold, kozepes -> sarga, magas -> piros
    rgb[:, :, 0] = np.clip(bev * 2.0, 0, 1) * 255        # R no a magassaggal
    rgb[:, :, 1] = np.clip(2.0 - bev * 2.0, 0, 1) * 255  # G csokken
    rgb[~occupied] = 0
    return rgb


class Lidar(CarlaActorBase):
    """
    Ray-cast lidar. Ket kimenete van, KULON celra:

      on_recv_image  -> (H, W) float [0,1] felulnezeti (BEV) kep. Ez csak
                        MEGJELENITES: az ember ezt tudja ertelmezni ranezesre.
                        A pixel erteke az oda eso legmagasabb pont z-je.

      on_recv_points -> (N, 3) nyers XYZ pontfelho a szenzor sajat
                        koordinatarendszereben. EZ megy a halonak: ebbol keszul
                        a range image (feature_extractions/lidar).

    Miert nem a BEV megy a halonak: a BEV felulnezet, ahol az uttest nagy resze
    ures - meressel a kep csak ~3%-ban kitoltott, es a graf-encoder Stemje utan
    a pontok mindossze 11%-a hordoz informaciot. A range image ezzel szemben a
    szenzor SAJAT nezopontjat orzi meg (64 elevacio x 1024 azimut), ahol
    gyakorlatilag minden sugarnak van merese: ~96% kitoltottseg, a Stem utan
    97% tartalmas pont. A graf-encodernek pont ilyen suru bemenet kell.
    """

    def __init__(self, world, width=256, height=256, transform=carla.Transform(), on_recv_image=None,
                 attach_to=None, on_recv_points=None):
        self._width = width
        self._height = height
        self.on_recv_image = on_recv_image
        self.on_recv_points = on_recv_points
        self.range = 50
        # A magassag-kodolas hatarai meterben, a szenzorhoz kepest. A szenzor
        # z=2.4-en van, tehat a talaj kb. -2.4. A felso hatar 4 m: e folott mar
        # csak epuletek es fak vannak, azokat egy szintre vonjuk ossze.
        self._z_min = -3.0
        self._z_max = 4.0

        # Setup lidar blueprint
        #
        # 360 fokos, mert a range image azimut tengelye korbeer, es a halo
        # CircularConv2d-je erre epul: az utolso oszlopot a legelsohoz padeli.
        # Kisebb FOV-nal a halo a jelenet ket, egymassal NEM szomszedos szelet
        # ragasztana ossze.
        #
        # points_per_second es rotation_frequency EGYUTT jar!
        #
        # A points_per_second a teljes pontkibocsatas, ami elosztodik a
        # fordulatok kozott. Egy teljes 64x1024-es range image-hez
        # fordulatonkent 65536 pont kell, tehat:
        #
        #     points_per_second = 64 * 1024 * rotation_frequency
        #     40 Hz -> 64 * 1024 * 40 = 2621440
        #
        # Ha csak a frekvenciat emeled a pontszam nelkul, a range image
        # aranyosan kilyukad (40 Hz-en 1310720 ponttal a fele uresen maradna),
        # es a halo a lyukakat tanulna meg.
        lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('points_per_second', '2621440')
        lidar_bp.set_attribute('channels', '64')
        lidar_bp.set_attribute('range', str(self.range))
        lidar_bp.set_attribute('upper_fov', '10')
        lidar_bp.set_attribute('horizontal_fov', '360')
        lidar_bp.set_attribute('lower_fov', '-25')
        lidar_bp.set_attribute('rotation_frequency', '40')

        # Create and setup camera actor
        weak_self = weakref.ref(self)
        actor = world.spawn_actor(lidar_bp, transform, attach_to=attach_to.get_carla_actor())
        actor.listen(lambda lidar_data: Lidar.process_lidar_input(weak_self, lidar_data))
        print("Spawned actor \"{}\"".format(actor.type_id))

        super().__init__(world, actor)

    @staticmethod
    def process_lidar_input(weak_self, raw):
        self = weak_self()
        if not self:
            return
        # A CARLA bufferje (x, y, z, intensity) negyesek sorozata.
        points = np.frombuffer(raw.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))

        if callable(self.on_recv_points):
            # A copy() kell: a szerver ujrahasznositja a buffert, tehat
            # masolas nelkul a tartalom barmikor felulirodhat alattunk.
            #
            # A CARLA y tengelye balra pozitiv (bal kezes rendszer), a
            # pcd2range gombi vetitese viszont jobb kezes rendszert var. Az y
            # elojelvaltasa nelkul a range image vizszintesen tukrozve lenne.
            xyz = points[:, :3].copy()
            xyz[:, 1] = -xyz[:, 1]
            self.on_recv_points(xyz)

        if not callable(self.on_recv_image):
            return

        # Meterbol pixelbe. A kep kozepe (0, 0) = az ego auto.
        #
        # NINCS np.fabs(): a regi kod abszolutertekkel szamolt, ami a kep egy
        # negyedebe gyurte az egesz jelenetet - az autotol balra es jobbra, il-
        # letve elore es hatra eso pontok egymasra tukrozodtek. 110 fokos,
        # elore nezo lidarnal ez meg nem tunt fel, 360 foknal viszont az auto
        # mogotti forgalom rahajtogatodna az elotte levore.
        scale = min(self._width, self._height) / (2.0 * float(self.range))
        px = points[:, 0] * scale + 0.5 * self._width
        py = points[:, 1] * scale + 0.5 * self._height

        # A hatotavon kivuli pontokat ELDOBJUK, nem levagjuk: a clip a kep
        # szelere kenne oket egy hamis "fal" csikot rajzolva.
        inside = (px >= 0) & (px < self._width) & (py >= 0) & (py < self._height)
        px = px[inside].astype(np.int32)
        py = py[inside].astype(np.int32)
        pz = points[inside, 2]

        # Magassag -> [0, 1]. A z_min alatti (talaj) es z_max feletti
        # (epuletteto) ertekek a ket vegponton torlodnak ossze.
        z_norm = np.clip((pz - self._z_min) / (self._z_max - self._z_min), 0.0, 1.0)

        # Egy cellaba tobb pont is eshet - a LEGMAGASABBAT tartjuk meg. A
        # maximum.at azert kell, mert a sima indexeles (img[py, px] = z) nem
        # determinisztikus utkozesnel: az utolso ertek nyerne, nem a legnagyobb.
        lidar_img = np.zeros((self._height, self._width), dtype=np.float32)
        np.maximum.at(lidar_img, (py, px), z_norm)

        self.on_recv_image(lidar_img)


# ===============================================================================
# CollisionSensor
# ===============================================================================

class CollisionSensor(CarlaActorBase):
    def __init__(self, world, vehicle, on_collision_fn):
        self.on_collision_fn = on_collision_fn

        # Collision history
        self.history = []

        # Setup sensor blueprint
        bp = world.get_blueprint_library().find("sensor.other.collision")

        # Create and setup sensor
        weak_self = weakref.ref(self)
        actor = world.spawn_actor(bp, carla.Transform(), attach_to=vehicle.get_carla_actor())
        actor.listen(lambda event: CollisionSensor.on_collision(weak_self, event))

        super().__init__(world, actor)

    @staticmethod
    def on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return

        # Call on_collision_fn
        if callable(self.on_collision_fn):
            self.on_collision_fn(event)


# ===============================================================================
# LaneInvasionSensor
# ===============================================================================

class LaneInvasionSensor(CarlaActorBase):
    def __init__(self, world, vehicle, on_invasion_fn):
        self.on_invasion_fn = on_invasion_fn

        # Setup sensor blueprint
        bp = world.get_blueprint_library().find("sensor.other.lane_invasion")

        # Create sensor
        weak_self = weakref.ref(self)
        actor = world.spawn_actor(bp, carla.Transform(), attach_to=vehicle.get_carla_actor())
        actor.listen(lambda event: LaneInvasionSensor.on_invasion(weak_self, event))

        super().__init__(world, actor)

    @staticmethod
    def on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return

        # Call on_invasion_fn
        if callable(self.on_invasion_fn):
            self.on_invasion_fn(event)


# ===============================================================================
# Camera
# ===============================================================================

class Camera(CarlaActorBase):
    def __init__(self, world, width, height, transform=carla.Transform(),
                 attach_to=None, on_recv_image=None,
                 camera_type="sensor.camera.rgb", color_converter=carla.ColorConverter.Raw, custom_palette=False):
        self.on_recv_image = on_recv_image
        self.color_converter = color_converter

        self.custom_palette = custom_palette
        # Setup camera blueprint
        camera_bp = world.get_blueprint_library().find(camera_type)
        camera_bp.set_attribute("image_size_x", str(width))
        camera_bp.set_attribute("image_size_y", str(height))
        camera_bp.set_attribute("fov", f"110")
        # camera_bp.set_attribute("sensor_tick", str(sensor_tick))

        # Motion blur KI. A CARLA RGB kameraja alapertelmezetten elmossa a
        # kepet (motion_blur_intensity=0.45), es 70 km/h-nal ez lathatoan
        # elkeni a sav szelet es a tavoli autokat.
        #
        # Ket okbol rossz ez nekunk:
        #   - a halonak pont ezek a reszletek kellenek a donteshez
        #   - az elmosas MERTEKE a sebessegtol fugg, tehat ugyanaz a hely
        #     mas kepet ad allva es haladva - a halonak ez csak zaj
        #
        # Ezt a beallitast a dataset gyujtoben (autoencoders/camera/
        # collect_dataset_v2.py) is ugyanigy kell tartani, kulonben az AE
        # mas kepen tanul, mint amit eles futasban lat.
        if camera_bp.has_attribute("motion_blur_intensity"):
            camera_bp.set_attribute("motion_blur_intensity", "0.0")
        if camera_bp.has_attribute("motion_blur_max_distortion"):
            camera_bp.set_attribute("motion_blur_max_distortion", "0.0")
        if camera_bp.has_attribute("motion_blur_min_object_screen_size"):
            camera_bp.set_attribute("motion_blur_min_object_screen_size", "0.0")

        # Create and setup camera actor
        weak_self = weakref.ref(self)
        actor = world.spawn_actor(camera_bp, transform, attach_to=attach_to.get_carla_actor())
        actor.listen(lambda image: Camera.process_camera_input(weak_self, image))
        print("Spawned actor \"{}\"".format(actor.type_id))

        super().__init__(world, actor)

    @staticmethod
    def process_camera_input(weak_self, image):
        self = weak_self()
        if not self:
            return
        if callable(self.on_recv_image):
            image.convert(self.color_converter)
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.on_recv_image(array)

    def destroy(self):
        super().destroy()


# ===============================================================================
# Vehicle
# ===============================================================================

class Vehicle(CarlaActorBase):
    def __init__(self, world, transform=carla.Transform(),
                 on_collision_fn=None, on_invasion_fn=None,
                 vehicle_type="vehicle.tesla.model3"):
        # Setup vehicle blueprint
        vehicle_bp = world.get_blueprint_library().find(vehicle_type)
        color = vehicle_bp.get_attribute("color").recommended_values[0]
        vehicle_bp.set_attribute("color", color)

        # Create vehicle actor
        actor = world.spawn_actor(vehicle_bp, transform)
        print("Spawned actor \"{}\"".format(actor.type_id))

        super().__init__(world, actor)

        # Maintain vehicle control
        self.control = carla.VehicleControl()

        if callable(on_collision_fn):
            self.collision_sensor = CollisionSensor(world, self, on_collision_fn=on_collision_fn)
        if callable(on_invasion_fn):
            self.lane_sensor = LaneInvasionSensor(world, self, on_invasion_fn=on_invasion_fn)

    def tick(self):
        self.actor.apply_control(self.control)

    def get_speed(self):
        """
        Return current vehicle speed in km/h
        """
        velocity = self.get_velocity()
        return 3.6 * np.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

    def get_angle(self, waypoint):
        fwd = vector(self.get_velocity())
        wp_fwd = vector(waypoint.transform.rotation.get_forward_vector())
        return angle_diff(wp_fwd, fwd)

    def get_closest_waypoint(self):
        return self.world.map.get_waypoint(self.get_transform().location, project_to_road=True)


# ===============================================================================
# World
# ===============================================================================

class World():
    def __init__(self, client, town):
        self.world = client.load_world(town)
        self.map = self.get_map()
        self.actor_list = []

    def tick(self):
        for actor in list(self.actor_list):
            actor.tick()

        self.world.tick()

    def destroy(self):
        print("Destroying all spawned actors")
        for actor in list(self.actor_list):
            actor.destroy()

    def get_carla_world(self):
        return self.world

    def __getattr__(self, name):
        """Relay missing methods to underlying carla object"""
        return getattr(self.world, name)
