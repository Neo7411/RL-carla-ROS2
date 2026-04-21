"""
Goal-conditioned CARLA RL environment (ROS2-only).

Observation (Dict):
    bev:    (H, W, 6) float32    LiDAR BEV + route mask channel
    vector: (6,) float32         normalized [speed, heading_err, cross_track,
                                 dist_to_goal, goal_dx_ego, goal_dy_ego]

Action: Box(2,) in [-1, 1]       [steer, throttle_or_brake]

Reset picks a random lanelet pose, plans a random reachable route via
MapRouter, and respawns the ego via the carla_ros_bridge topic
`/carla/ego_vehicle/control/set_transform`.
"""
from __future__ import annotations

import math
import os
import random
import threading
import time

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Quaternion

from tools.bev_processor import BEVProcessor
from tools.map_router import MapRouter

try:
    from carla_msgs.msg import CarlaEgoVehicleControl, CarlaCollisionEvent
    CARLA_MSGS_AVAILABLE = True
except ImportError:
    CARLA_MSGS_AVAILABLE = False


# Normalization constants for the vector observation
SPEED_MAX = 17.0       # m/s
DIST_MAX = 300.0       # m — normalize distance-to-goal
EGO_REL_MAX = 50.0     # m — clip goal-relative-xy for stability


class CarlaRLEnvironment(gym.Env):
    def __init__(self, ros_node: Node, bev_processor: BEVProcessor | None = None,
                 map_path: str | None = None):
        super().__init__()
        self._node = ros_node
        self._bev = bev_processor or BEVProcessor()

        if map_path is None:
            here = os.path.dirname(os.path.abspath(__file__))
            map_path = os.path.join(here, '..', 'maps', 'Town04.osm')
        self._router = MapRouter(map_path)
        self._rng = random.Random()

        self.observation_space = spaces.Dict({
            'bev': spaces.Box(low=0.0, high=10.0,
                              shape=self._bev.observation_shape,
                              dtype=np.float32),
            'vector': spaces.Box(low=-1.0, high=1.0, shape=(6,),
                                 dtype=np.float32),
        })
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,),
                                       dtype=np.float32)

        # Shared state (guarded by _lock)
        self._latest_points: np.ndarray | None = None
        self._latest_velocity = 0.0
        self._ego_xy = np.zeros(2, dtype=np.float32)
        self._ego_yaw = 0.0
        self._collision = False
        self._lock = threading.Lock()
        self._new_frame = threading.Event()

        self._step_count = 0
        self._max_steps = 1500
        self._prev_steer = 0.0
        self._prev_fraction = 0.0

        self._node.create_subscription(
            PointCloud2, '/carla/ego_vehicle/lidar', self._on_lidar, 10)
        self._node.create_subscription(
            Odometry, '/carla/ego_vehicle/odometry', self._on_odom, 10)

        if CARLA_MSGS_AVAILABLE:
            self._node.create_subscription(
                CarlaCollisionEvent, '/carla/ego_vehicle/collision',
                self._on_collision, 10)
            self._control_pub = self._node.create_publisher(
                CarlaEgoVehicleControl,
                '/carla/ego_vehicle/vehicle_control_cmd', 10)
            self._respawn_pub = self._node.create_publisher(
                Pose, '/carla/ego_vehicle/control/set_transform', 10)

    # ── ROS callbacks ───────────────────────────────────────────────────

    def _on_lidar(self, msg: PointCloud2):
        raw = np.frombuffer(msg.data, dtype=np.float32)
        num_fields = int(msg.point_step / 4)
        points = raw.reshape(-1, num_fields)[:, :4].copy()
        with self._lock:
            self._latest_points = points
        self._new_frame.set()

    def _on_odom(self, msg: Odometry):
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        speed = math.sqrt(vx * vx + vy * vy)
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = _yaw_from_quat(q.x, q.y, q.z, q.w)
        with self._lock:
            self._latest_velocity = speed
            self._ego_xy = np.array([p.x, p.y], dtype=np.float32)
            self._ego_yaw = yaw

    def _on_collision(self, msg):
        with self._lock:
            self._collision = True

    # ── Gym API ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng.seed(seed)

        self._step_count = 0
        self._collision = False
        self._prev_steer = 0.0
        self._prev_fraction = 0.0

        # Fixed spawn (matches the original working spawn point).
        # Random goal is still planned from here.
        start_xy = np.array([290.2, 168.9], dtype=np.float32)
        # Hardcoded yaw: the road at this spawn runs east-west in CARLA world
        # frame (yaw=0 points the car parallel to the road edges). The
        # Lanelet2 `heading_at` value for this location is off because the
        # nearest lanelet segment belongs to a bend that doesn't match the
        # actual road heading at the spawn point. Roll/pitch stay 0.
        start_yaw = 0.0
        self._router.plan_random_route(
            start_xy, min_len=20.0, max_len=1000.0,
            max_tries=60, rng=self._rng)

        # Announce the goal for this episode
        if self._router.waypoints is not None:
            goal = self._router.waypoints[-1]
            print(f"[env] new episode — goal=({goal[0]:+.1f}, {goal[1]:+.1f})  "
                  f"route_len={self._router.total_length:.1f} m  "
                  f"waypoints={len(self._router.waypoints)}")

        # Respawn ego at the route start
        if CARLA_MSGS_AVAILABLE:
            pose = Pose()
            pose.position.x = float(start_xy[0])
            pose.position.y = float(start_xy[1])
            pose.position.z = 2.0
            # Roll = pitch = 0 (level spawn); yaw aligned with lanelet heading
            # so the car starts parallel to the road edges.
            qz = math.sin(start_yaw * 0.5)
            qw = math.cos(start_yaw * 0.5)
            pose.orientation = Quaternion(x=0.0, y=0.0, z=qz, w=qw)
            self._respawn_pub.publish(pose)
            # Brief full brake to settle physics, then explicitly release.
            stop = CarlaEgoVehicleControl()
            stop.brake = 1.0
            stop.hand_brake = False
            self._control_pub.publish(stop)
            time.sleep(1)
            release = CarlaEgoVehicleControl()
            release.throttle = 0.0
            release.brake = 0.0
            release.hand_brake = False
            release.manual_gear_shift = False
            self._control_pub.publish(release)

        time.sleep(1)
        self._new_frame.clear()
        self._new_frame.wait(timeout=5.0)

        # Clear any spurious collision events triggered by the teleport itself.
        # The bridge can fire CarlaCollisionEvent when the ego is snapped to
        # a new transform (ground contact, brief overlap with nearby geometry).
        # Clearing here — AFTER the respawn settles — prevents an immediate
        # termination on step 1 which would cause an empty episode + double reset.
        with self._lock:
            self._collision = False

        # Initialize route-progress baseline AFTER the router has the ego's
        # actual pose — prevents a fake "progress" spike on the first step.
        with self._lock:
            ego_xy = self._ego_xy.copy()
            ego_yaw = self._ego_yaw
        if self._router.waypoints is not None:
            self._prev_fraction = self._router.progress(ego_xy, ego_yaw).fraction_done

        return self._build_observation(), {}

    def step(self, action):
        self._step_count += 1
        self._publish_control(action)

        self._new_frame.clear()
        self._new_frame.wait(timeout=3.0)

        obs = self._build_observation()
        with self._lock:
            v = self._latest_velocity
            c = self._collision
            ego_xy = self._ego_xy.copy()
            ego_yaw = self._ego_yaw

        prog = self._router.progress(ego_xy, ego_yaw)

        # Collision grace period: ignore collisions in the first few steps
        # after reset. The bridge can fire spurious events right after the
        # teleport (ground contact, nearby geometry) — treating them as real
        # would produce empty episodes + back-to-back resets.
        COLLISION_GRACE_STEPS = 5
        if c and self._step_count <= COLLISION_GRACE_STEPS:
            with self._lock:
                self._collision = False
            c = False

        reward = self._compute_reward(v, c, action, prog)

        # Terminate only on collision or goal-reached. Off-route is a soft
        # signal handled in the reward; letting the car cross lanes freely
        # is intentional — future traffic-avoidance behavior may require it.
        terminated = bool(c or prog.goal_reached)
        truncated = self._step_count >= self._max_steps

        info = {
            'success': bool(prog.goal_reached),
            'collision': bool(c),
            'route_fraction': prog.fraction_done,
            'dist_to_goal': prog.dist_to_goal,
            'speed': float(v),
        }
        self._prev_steer = float(action[0])
        self._prev_fraction = prog.fraction_done
        return obs, reward, terminated, truncated, info

    # ── Observation & control helpers ──────────────────────────────────

    def _build_observation(self) -> dict:
        with self._lock:
            points = self._latest_points
            ego_xy = self._ego_xy.copy()
            ego_yaw = self._ego_yaw
            speed = self._latest_velocity

        prog = self._router.progress(ego_xy, ego_yaw) if \
            self._router.waypoints is not None else None

        # Project route into LiDAR-local frame (ego-forward = +x)
        route_local = None
        if prog is not None:
            wps = self._router.waypoints
            route_local = _world_to_local(wps, ego_xy, ego_yaw)

        if points is not None:
            bev = self._bev.process(points, route_local=route_local)
        else:
            bev = np.zeros(self._bev.observation_shape, dtype=np.float32)
            if route_local is not None:
                self._bev._rasterize_route(bev, route_local)

        vec = self._vector_obs(speed, ego_xy, ego_yaw, prog)
        return {'bev': bev.astype(np.float32), 'vector': vec}

    def _vector_obs(self, speed: float, ego_xy: np.ndarray, ego_yaw: float,
                    prog) -> np.ndarray:
        if prog is None:
            return np.zeros(6, dtype=np.float32)
        goal = self._router.waypoints[-1]
        rel = _world_to_local(goal[None, :], ego_xy, ego_yaw)[0]
        vec = np.array([
            np.clip(speed / SPEED_MAX, -1.0, 1.0),
            np.clip(prog.heading_err / math.pi, -1.0, 1.0),
            np.clip(prog.cross_track / 5.0, -1.0, 1.0),
            np.clip(prog.dist_to_goal / DIST_MAX, 0.0, 1.0),
            np.clip(rel[0] / EGO_REL_MAX, -1.0, 1.0),
            np.clip(rel[1] / EGO_REL_MAX, -1.0, 1.0),
        ], dtype=np.float32)
        return vec

    def _publish_control(self, action):
        if not CARLA_MSGS_AVAILABLE:
            return
        ctrl = CarlaEgoVehicleControl()
        ctrl.steer = float(np.clip(action[0], -1.0, 1.0))

        # Throttle bias: action[1] = 0 maps to ~40% throttle so the car rolls
        # out of the box. Policy has to actively pull action[1] toward −1 to
        # brake. Brake-only activates in [-1.0, -0.7]; the rest is throttle.
        # Throttle is capped at 0.6 so top speed stays moderate.
        a = float(np.clip(action[1], -1.0, 1.0))
        if a >= -0.7:
            if a >= 0.0:
                ctrl.throttle = float(0.4 + 0.2 * a)           # [0.4, 0.6]
            else:
                ctrl.throttle = float(max(0.0, 0.4 + (0.4 / 0.7) * a))  # [0.0, 0.4]
            ctrl.brake = 0.0
        else:
            # [-1.0, -0.7] → [1.0, 0.0] brake (sharp brake zone)
            ctrl.throttle = 0.0
            ctrl.brake = float(np.clip((-0.7 - a) / 0.3, 0.0, 1.0))

        ctrl.hand_brake = False
        ctrl.reverse = False
        ctrl.manual_gear_shift = False
        ctrl.gear = 0
        self._control_pub.publish(ctrl)

    # ── Reward ──────────────────────────────────────────────────────────

    def _compute_reward(self, velocity, collision, action, prog) -> float:
        # Terminal: collision — softened (−30 baseline) while the policy is
        # still learning to drive. Fear of crashes was suppressing exploration.
        # Ramp this back up (−100 or lower) once the agent reliably drives.
        if collision:
            return -30.0 - 2.0 * velocity
        if prog.goal_reached:
            # Large arrival bonus + strong time/speed bonus so policy hurries.
            return 400.0 + 20.0 * velocity

        # Dense goal-seeking: meters gained toward the goal along the route.
        d_frac = prog.fraction_done - self._prev_fraction
        progress_m = d_frac * self._router.total_length
        r_progress = 10.0 * progress_m

        # Speed — mild quadratic: policy prefers moving but isn't pushed toward
        # reckless top speed. No idle penalty: paying negative reward while
        # stopped paradoxically pushes the policy to end episodes fast.
        v = max(velocity, 0.0)
        r_speed = 0.1 * v + 0.02 * v * v   # at 15 m/s → 1.5 + 4.5 = 6.0

        # Weak heading-toward-route shaping. No lane-keeping, no cross-track.
        r_heading = -0.1 * abs(prog.heading_err)

        return float(r_progress + r_speed + r_heading)


# ── helpers ─────────────────────────────────────────────────────────────

def _yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    # ZYX yaw (heading in the world XY plane)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _world_to_local(pts_xy: np.ndarray, ego_xy: np.ndarray,
                    ego_yaw: float) -> np.ndarray:
    """Rotate/translate world XY into ego-forward (+x) frame."""
    c = math.cos(-ego_yaw)
    s = math.sin(-ego_yaw)
    d = pts_xy - ego_xy
    out = np.empty_like(d)
    out[:, 0] = c * d[:, 0] - s * d[:, 1]
    out[:, 1] = s * d[:, 0] + c * d[:, 1]
    return out
