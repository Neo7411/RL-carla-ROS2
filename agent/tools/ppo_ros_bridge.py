"""ROS-only bridge to CARLA — async mode with frame-synced stepping.

The bridge runs free, sensors stream at their native rates. The env
paces itself by *waiting for a fresh lidar frame* on each env.step()
instead of sleeping a fixed wall-clock interval. This couples each
policy step to one lidar revolution (~20 Hz) and removes the
fixed-sleep-vs-sensor-rate jitter that would otherwise alias frames.

Subscribed topics:
  /carla/ego_vehicle/semantic_lidar          PointCloud2  (BEV input)
  /carla/ego_vehicle/semantic_segmentation_front/image
                                              Image bgra8 (camera input)
  /carla/ego_vehicle/gnss                    NavSatFix    (lat/lon)
  /carla/ego_vehicle/imu                     Imu          (orientation → yaw)
  /carla/ego_vehicle/speedometer             Float32      (m/s)
  /carla/ego_vehicle/collision               CarlaCollisionEvent

Coordinate frame:
  GNSS lat/lon are projected through lanelet2's UtmProjector(Origin(0,0)).
  CARLA writes lat/lon such that this projection is sign-flipped in y
  relative to the lanelet2 .osm (and to set_transform). We flip y on
  ingest so pose_xy lives in the same frame as the .osm map.
"""

import math
import threading

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import PointCloud2, Image, NavSatFix, Imu
import sensor_msgs_py.point_cloud2 as pc2

from std_msgs.msg import Float32
from geometry_msgs.msg import Pose, Twist, Vector3

from lanelet2.core import GPSPoint
from lanelet2.io import Origin
from lanelet2.projection import UtmProjector

from carla_msgs.msg import (
    CarlaEgoVehicleControl,
    CarlaCollisionEvent,
)


# The bridge passes the semantic camera through at native resolution.
# Resizing/cropping happens in the env if needed. The bridge knows the
# camera is 400x70 from objects.json — we don't hard-code it here, the
# callback uses msg.height/msg.width.


def _yaw_to_quat(yaw):
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def _quat_to_yaw(qx, qy, qz, qw):
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


class PPOROSBridgeNode(Node):
    def __init__(self):
        super().__init__('ppo_data_reader')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # GNSS lat/lon -> world meters. Same projector as the lanelet2 map,
        # so projected points share the .osm coordinate frame after the
        # CARLA y-flip is applied (see _gnss_cb).
        self._proj = UtmProjector(Origin(0.0, 0.0))

        # ---- Shared state read by the Gym env ----
        self.latest_lidar = None
        self.latest_sem_cam = None  # (H, W, 3) uint8 RGB
        self._sem_lock = threading.Lock()

        # Lidar frame counter + condition: env.step() waits for this to
        # advance, giving us real sensor-paced sync without sync-mode
        # bridge ticking. Free-running but lock-stepped.
        self._lidar_frame = 0
        self._lidar_cv = threading.Condition()

        self.pose_xy = np.zeros(2, dtype=np.float32)
        self.yaw = 0.0
        self.speed_mps = 0.0
        self.collision_intensity = 0.0
        self.collided = False
        self._has_gnss = False
        self._has_imu = False

        # ---- Subscribers ----
        self.create_subscription(
            PointCloud2, '/carla/ego_vehicle/semantic_lidar',
            self._lidar_cb, qos)
        self.create_subscription(
            Image, '/carla/ego_vehicle/semantic_segmentation_front/image',
            self._sem_cam_cb, qos)
        self.create_subscription(
            NavSatFix, '/carla/ego_vehicle/gnss',
            self._gnss_cb, qos)
        self.create_subscription(
            Imu, '/carla/ego_vehicle/imu',
            self._imu_cb, qos)
        self.create_subscription(
            Float32, '/carla/ego_vehicle/speedometer',
            self._speed_cb, qos)
        self.create_subscription(
            CarlaCollisionEvent, '/carla/ego_vehicle/collision',
            self._collision_cb, qos)

        # ---- Publishers ----
        self.control_pub = self.create_publisher(
            CarlaEgoVehicleControl,
            '/carla/ego_vehicle/vehicle_control_cmd', 10)
        self.set_transform_pub = self.create_publisher(
            Pose, '/carla/ego_vehicle/control/set_transform', 10)
        self.set_velocity_pub = self.create_publisher(
            Twist, '/carla/ego_vehicle/control/set_target_velocity', 10)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _lidar_cb(self, msg):
        pts = pc2.read_points(
            msg, field_names=("x", "y", "z", "ObjTag"), skip_nans=True,
        )
        arr = np.asarray(list(pts))
        if arr.size > 0:
            self.latest_lidar = arr
            with self._lidar_cv:
                self._lidar_frame += 1
                self._lidar_cv.notify_all()

    def _sem_cam_cb(self, msg):
        # carla_ros_bridge publishes the semantic camera as a colorized
        # bgra8 image (palette already applied). Drop alpha, BGR→RGB,
        # pass through at native resolution.
        h, w = msg.height, msg.width
        if msg.encoding != "bgra8":
            return
        buf = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 4)
        rgb = cv2.cvtColor(buf, cv2.COLOR_BGRA2RGB)
        with self._sem_lock:
            self.latest_sem_cam = rgb

    def _gnss_cb(self, msg):
        # CARLA GNSS lat/lon, projected via UtmProjector(Origin(0,0)),
        # gives x ≈ CARLA-world x and y ≈ −CARLA-world y. Flip y so
        # pose_xy lives in the same frame as the lanelet2 .osm and
        # set_transform.
        gp = GPSPoint(float(msg.latitude), float(msg.longitude),
                      float(msg.altitude))
        p = self._proj.forward(gp)
        self.pose_xy[0] = p.x
        self.pose_xy[1] = -p.y
        self._has_gnss = True

    def _imu_cb(self, msg):
        # CARLA IMU orientation is in the same y-flipped GNSS frame, so
        # the yaw extracted directly is the heading we want? No — set_transform
        # publishes a yaw in the lanelet2/.osm frame and the CARLA actor uses
        # that. The IMU reports orientation in the same world the actor lives
        # in, so its yaw is consistent with our flipped pose. Empirically
        # confirmed by checking heading at spawn (yaw=0 in spawn config →
        # IMU also reports yaw≈0 in our convention).
        q = msg.orientation
        self.yaw = _quat_to_yaw(q.x, q.y, q.z, q.w)
        self._has_imu = True

    def has_pose(self):
        return self._has_gnss and self._has_imu

    def _speed_cb(self, msg):
        self.speed_mps = float(msg.data)

    def _collision_cb(self, msg):
        n = msg.normal_impulse
        intensity = math.sqrt(n.x * n.x + n.y * n.y + n.z * n.z)
        if intensity > self.collision_intensity:
            self.collision_intensity = intensity
        self.collided = True

    # ------------------------------------------------------------------
    # Gym-facing helpers
    # ------------------------------------------------------------------
    def get_sem_cam(self):
        with self._sem_lock:
            return None if self.latest_sem_cam is None else self.latest_sem_cam.copy()

    def wait_for_next_lidar(self, timeout=1.0):
        """Block until at least one new lidar frame has arrived since this
        call started. Returns True on success, False on timeout. Used by
        env.step() to pace itself to the sensor instead of wall-clock."""
        with self._lidar_cv:
            start = self._lidar_frame
            return self._lidar_cv.wait_for(
                lambda: self._lidar_frame > start,
                timeout=timeout,
            )

    def send_control(self, action):
        msg = CarlaEgoVehicleControl()
        msg.steer = float(np.clip(action[0], -1.0, 1.0))
        msg.throttle = float(np.clip(action[1], 0.0, 1.0))
        msg.brake = float(np.clip(action[2], 0.0, 1.0))
        msg.hand_brake = False
        msg.reverse = False
        msg.gear = 1
        msg.manual_gear_shift = False
        self.control_pub.publish(msg)

    def send_brake(self):
        msg = CarlaEgoVehicleControl()
        msg.steer = 0.0
        msg.throttle = 0.0
        msg.brake = 1.0
        msg.hand_brake = True
        msg.gear = 1
        self.control_pub.publish(msg)

    def teleport(self, x, y, z, yaw=0.0):
        pose = Pose()
        pose.position.x = float(x)
        pose.position.y = float(y)
        pose.position.z = float(z)
        qx, qy, qz, qw = _yaw_to_quat(yaw)
        pose.orientation.x = qx
        pose.orientation.y = qy
        pose.orientation.z = qz
        pose.orientation.w = qw
        self.set_transform_pub.publish(pose)

        zero = Twist()
        zero.linear = Vector3(x=0.0, y=0.0, z=0.0)
        zero.angular = Vector3(x=0.0, y=0.0, z=0.0)
        self.set_velocity_pub.publish(zero)

    def clear_event_flags(self):
        self.collided = False
        self.collision_intensity = 0.0
