"""
Goal-reaching CARLA RL environment (ROS2-only).

The agent sees LiDAR BEV + a small goal-relative vector and must drive from
the fixed spawn to the fixed goal XY without crashing and without leaving
its lane.

Observation (Dict):
    bev:    (H, W, 5) float32      LiDAR BEV
    vector: (4,) float32            normalized [speed, dist_to_goal,
                                               goal_dx_ego, goal_dy_ego]

Action: Box(2,) in [-1, 1]        [steer, throttle_or_brake]
    action[1]=0 → ~50% throttle (bias so the untrained policy rolls).

Reward (per step):
    r_progress  = k_prog * (prev_dist - curr_dist)   # positive while closing
    r_distance  = -0.01 * curr_dist                   # mild "far is bad" prior
    r_speed     = 3.0 * (v / SPEED_TARGET)^2          # push toward 120 km/h
    r_time      = -1.0                                # anti-dawdle
    r_idle      = -2.0 if v < 0.5 else 0              # anti-stop
    r_lane      = -5.0 per lane-invasion event
    r_goal      = +300.0 on arrival (one-shot, terminal)
    r_collision = -150 - 1.5*v (terminal)
"""
from __future__ import annotations

import math
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

try:
    from carla_msgs.msg import (CarlaEgoVehicleControl, CarlaCollisionEvent,
                                CarlaLaneInvasionEvent)
    CARLA_MSGS_AVAILABLE = True
except ImportError:
    CARLA_MSGS_AVAILABLE = False


# Target top speed: 120 km/h ≈ 33.33 m/s.
SPEED_TARGET = 33.33

# Fixed spawn and goal.
SPAWN_XY = (290.2, 168.9)
SPAWN_YAW = 0.0
GOAL_XY = (386.6, 224.8)
GOAL_RADIUS = 5.0          # metres, arrival threshold

# Observation normalization.
DIST_NORM = 300.0          # m — max expected distance to goal
REL_NORM = 300.0           # m — clip goal-relative xy into [-1, 1]

# Reward tuning.
PROGRESS_GAIN = 2.0        # reward per metre closed toward the goal
DIST_GAIN = 0.01           # small constant "far is bad" prior
LANE_INVASION_PENALTY = 5.0
GOAL_BONUS = 300.0


class CarlaRLEnvironment(gym.Env):
    def __init__(self, ros_node: Node, bev_processor: BEVProcessor | None = None):
        super().__init__()
        self._node = ros_node
        self._bev = bev_processor or BEVProcessor()

        self.observation_space = spaces.Dict({
            'bev': spaces.Box(low=0.0, high=10.0,
                              shape=self._bev.observation_shape,
                              dtype=np.float32),
            'vector': spaces.Box(low=-1.0, high=1.0, shape=(4,),
                                 dtype=np.float32),
        })
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,),
                                       dtype=np.float32)

        # Shared state (guarded by _lock)
        self._latest_points: np.ndarray | None = None
        self._latest_velocity = 0.0
        self._ego_xy = np.array(SPAWN_XY, dtype=np.float32)
        self._ego_yaw = 0.0
        self._collision = False
        self._lane_invasions = 0        # counter, consumed each step
        self._lock = threading.Lock()
        self._new_frame = threading.Event()

        self._step_count = 0
        self._max_steps = 1500
        self._prev_dist = float(np.linalg.norm(
            np.array(GOAL_XY) - np.array(SPAWN_XY)))

        self._node.create_subscription(
            PointCloud2, '/carla/ego_vehicle/lidar', self._on_lidar, 10)
        self._node.create_subscription(
            Odometry, '/carla/ego_vehicle/odometry', self._on_odom, 10)

        if CARLA_MSGS_AVAILABLE:
            self._node.create_subscription(
                CarlaCollisionEvent, '/carla/ego_vehicle/collision',
                self._on_collision, 10)
            self._node.create_subscription(
                CarlaLaneInvasionEvent, '/carla/ego_vehicle/lane_invasion',
                self._on_lane_invasion, 10)
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

    def _on_lane_invasion(self, msg):
        # Each message = one lane-marking crossing. Tally and consume in step.
        with self._lock:
            self._lane_invasions += 1

    # ── Gym API ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self._collision = False
        self._lane_invasions = 0

        if CARLA_MSGS_AVAILABLE:
            pose = Pose()
            pose.position.x = float(SPAWN_XY[0])
            pose.position.y = float(SPAWN_XY[1])
            pose.position.z = 2.0
            qz = math.sin(SPAWN_YAW * 0.5)
            qw = math.cos(SPAWN_YAW * 0.5)
            pose.orientation = Quaternion(x=0.0, y=0.0, z=qz, w=qw)
            self._respawn_pub.publish(pose)

            stop = CarlaEgoVehicleControl()
            stop.brake = 1.0
            self._control_pub.publish(stop)
            time.sleep(1)
            release = CarlaEgoVehicleControl()
            self._control_pub.publish(release)

        time.sleep(1)
        self._new_frame.clear()
        self._new_frame.wait(timeout=5.0)

        # Clear collision/lane-invasion events fired by the teleport itself.
        with self._lock:
            self._collision = False
            self._lane_invasions = 0
            ego_xy = self._ego_xy.copy()

        self._prev_dist = float(np.linalg.norm(np.array(GOAL_XY) - ego_xy))

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
            lane_events = self._lane_invasions
            self._lane_invasions = 0       # consume
            ego_xy = self._ego_xy.copy()

        dist = float(np.linalg.norm(np.array(GOAL_XY) - ego_xy))
        goal_reached = dist < GOAL_RADIUS

        # Collision grace period — ignore spurious events right after teleport.
        COLLISION_GRACE_STEPS = 5
        if c and self._step_count <= COLLISION_GRACE_STEPS:
            with self._lock:
                self._collision = False
            c = False

        reward = self._compute_reward(v, c, dist, lane_events, goal_reached)
        self._prev_dist = dist

        terminated = bool(c or goal_reached)
        truncated = self._step_count >= self._max_steps

        info = {
            'collision': bool(c),
            'goal_reached': bool(goal_reached),
            'dist_to_goal': dist,
            'speed': float(v),
            'speed_kmh': float(v * 3.6),
            'lane_invasions': int(lane_events),
        }
        return obs, reward, terminated, truncated, info

    # ── Observation & control ──────────────────────────────────────────

    def _build_observation(self) -> dict:
        with self._lock:
            points = self._latest_points
            ego_xy = self._ego_xy.copy()
            ego_yaw = self._ego_yaw
            speed = self._latest_velocity

        if points is not None:
            bev = self._bev.process(points).astype(np.float32)
        else:
            bev = np.zeros(self._bev.observation_shape, dtype=np.float32)

        # Goal relative to the ego, projected into ego-forward frame so the
        # policy sees "goal is ahead-left / behind-right" in its own coords.
        goal = np.array(GOAL_XY, dtype=np.float32)
        dxy_world = goal - ego_xy
        c, s = math.cos(-ego_yaw), math.sin(-ego_yaw)
        gx_ego = c * dxy_world[0] - s * dxy_world[1]
        gy_ego = s * dxy_world[0] + c * dxy_world[1]
        dist = float(np.linalg.norm(dxy_world))

        vec = np.array([
            np.clip(speed / SPEED_TARGET, 0.0, 1.5),
            np.clip(dist / DIST_NORM, 0.0, 1.5),
            np.clip(gx_ego / REL_NORM, -1.0, 1.0),
            np.clip(gy_ego / REL_NORM, -1.0, 1.0),
        ], dtype=np.float32)

        return {'bev': bev, 'vector': vec}

    def _publish_control(self, action):
        if not CARLA_MSGS_AVAILABLE:
            return
        ctrl = CarlaEgoVehicleControl()
        ctrl.steer = float(np.clip(action[0], -1.0, 1.0))

        # Throttle bias: action[1]=0 → 50% throttle so the untrained policy
        # rolls. a ≥ 0: throttle 0.5..1.0. a in [-0.5, 0]: throttle 0.5..0.
        # a in [-1, -0.5]: brake 0..1 (sharp brake zone).
        a = float(np.clip(action[1], -1.0, 1.0))
        if a >= 0.0:
            ctrl.throttle = float(0.5 + 0.5 * a)
            ctrl.brake = 0.0
        elif a >= -0.5:
            ctrl.throttle = float(0.5 + a)
            ctrl.brake = 0.0
        else:
            ctrl.throttle = 0.0
            ctrl.brake = float(np.clip((-0.5 - a) / 0.5, 0.0, 1.0))

        ctrl.hand_brake = False
        ctrl.reverse = False
        ctrl.manual_gear_shift = False
        ctrl.gear = 0
        self._control_pub.publish(ctrl)

    # ── Reward ──────────────────────────────────────────────────────────

    def _compute_reward(self, velocity: float, collision: bool,
                        dist: float, lane_events: int,
                        goal_reached: bool) -> float:
        if collision:
            return -150.0 - 1.5 * max(velocity, 0.0)
        if goal_reached:
            return GOAL_BONUS

        v = max(velocity, 0.0)

        # Progress: positive while closing, negative while receding.
        r_progress = PROGRESS_GAIN * (self._prev_dist - dist)

        # Constant far-is-bad prior — keeps gradient alive when progress
        # is momentarily flat (e.g. while turning).
        r_distance = -DIST_GAIN * dist

        # Speed — quadratic, capped just above the target.
        r_speed = min(3.5, 3.0 * (v / SPEED_TARGET) ** 2)

        r_time = -1.0
        r_idle = -2.0 if v < 0.5 else 0.0

        # Lane invasion: flat penalty per crossing this step.
        r_lane = -LANE_INVASION_PENALTY * lane_events

        return float(r_progress + r_distance + r_speed
                     + r_time + r_idle + r_lane)


# ── helpers ─────────────────────────────────────────────────────────────

def _yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)
