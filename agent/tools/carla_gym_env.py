"""Gym environment wrapping the CARLA ROS bridge for goal-directed PPO.

Observation (Dict — needs MultiInputPolicy):
  image: uint8 (H, W, 6)
    [..., 0:3] = front semantic-segmentation camera (resized RGB)
    [..., 3:6] = top-down semantic-lidar BEV (resized to match)
  state: float32 (6,)
    [dist_to_goal/100, sin(bearing_err), cos(bearing_err),
     speed/30, lateral_offset/4, route_progress]

Action: [steer, throttle, brake] in [-1,1]/[0,1]/[0,1].

Coordinate frames:
  - Odometry, lanelet2, goal manager all live in the right-handed,
    Y-up world frame.
  - The CARLA-editor coordinates you paste are Y-down; GoalManager
    flips Y for you. See goal_manager.py.
"""

import math
import os
import time

import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces

try:
    from .lanelet2_router import Lanelet2Router
    from .goal_manager import GoalManager
except ImportError:
    from lanelet2_router import Lanelet2Router
    from goal_manager import GoalManager


# ---------------------------------------------------------------------------
# Semantic-lidar tag groups (CARLA semantic tags).
# ---------------------------------------------------------------------------
TAG_ROAD = 1
TAG_SIDEWALK = 2
TAG_BUILDING = 3
TAG_PEDESTRIAN = 4
TAG_POLE = 5
TAG_ROADLINE = 6
TAG_VEHICLE = 10
TAG_WALL = 11
TAG_TRAFFIC_SIGN = 12

DRIVABLE_TAGS = (TAG_ROAD, TAG_ROADLINE)
OBSTACLE_TAGS = (TAG_VEHICLE, TAG_PEDESTRIAN)
STATIC_TAGS = (TAG_SIDEWALK, TAG_BUILDING, TAG_WALL, TAG_POLE, TAG_TRAFFIC_SIGN)


# ---------------------------------------------------------------------------
# Episode / reward configuration.
# ---------------------------------------------------------------------------
SPAWN_POINT = {
    "x": 360.2, "y": 168.9, "z": 0.3,
    "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
}

# Goal as pasted from the CARLA editor (left-handed, Y-down). GoalManager
# flips Y so we can paste editor coords verbatim.
GOAL_EDITOR_X = 386.10
GOAL_EDITOR_Y = -228.16
GOAL_RADIUS_M = 8.0   # bumped: closest routable lanelet ends ~5.6m off goal

# Map for routing.
HERE = os.path.dirname(os.path.abspath(__file__))
LANELET2_OSM = os.path.normpath(os.path.join(HERE, "..", "..", "maps", "Town04.osm"))

# Per-stream policy input sizes — lossless. Camera goes in at the
# bridge's native 70x400; BEV is rasterized at 256x256 (covers the
# 128m x 128m world around the ego at 0.5 m/px). Each stream gets its
# own NatureCNN trunk via MultiInputPolicy.
CAM_H, CAM_W = 70, 400
BEV_H, BEV_W = 256, 256

TARGET_SPEED_KMH = 60.0
TARGET_SPEED_MPS = TARGET_SPEED_KMH / 3.6

# Action shaping: in the early phase the policy can't escape an idle
# attractor (penalties for state dominate any action). Floor the
# throttle so the car always moves; the policy learns to steer first.
THROTTLE_FLOOR = 0.15

# Per-step reward weights — tuned for "any forward motion beats idling".
W_SPEED = 1.5             # was 0.5 — stronger pull toward forward motion
W_OVERSPEED = 0.2
W_STEER = 0.05
W_OBSTACLE = 0.5
W_TIME = 0.005            # was 0.01
W_IDLE = 0.05             # was 0.3 — don't crush an exploring policy
W_PROGRESS = 1.0          # +meters closer to goal this step
W_LATERAL = 0.05          # was 0.2
W_HEADING = 0.03          # was 0.1

# Sparse events.
R_GOAL = 100.0
R_COLLISION = -90.0
R_STUCK = -25.0

IDLE_SPEED_MPS = 1.0
COLLISION_INTENSITY_THRESHOLD = 100.0
STUCK_SPEED_MPS = 0.5
STEP_DT = 0.05            # only used as the reset-settle pacer now
STUCK_STEPS = int(8.0 / STEP_DT)
EPISODE_STEP_LIMIT = int(120.0 / STEP_DT)
RESET_SETTLE_S = 0.4
OBSTACLE_PROXIMITY_M = 6.0


class CarlaGymEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(self, ros_node, render_mode=None):
        super().__init__()
        self.node = ros_node
        self.render_mode = render_mode

        # BEV grid for the lidar input *before* it's resized to IMG_HxIMG_W.
        self.grid_size = 256
        self.resolution = 0.5
        self.center = self.grid_size // 2

        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Two separate image streams: SB3's MultiInputPolicy builds a
        # NatureCNN trunk per Box(uint8) entry, so the camera and BEV are
        # processed by independent CNNs and concatenated with the state
        # MLP at the latent layer. Better than channel-stacking — neither
        # modality has to share filters with the other.
        self.observation_space = spaces.Dict({
            "camera": spaces.Box(low=0, high=255,
                                 shape=(CAM_H, CAM_W, 3), dtype=np.uint8),
            "bev":    spaces.Box(low=0, high=255,
                                 shape=(BEV_H, BEV_W, 3), dtype=np.uint8),
            "state":  spaces.Box(low=-1.0, high=1.0,
                                 shape=(6,), dtype=np.float32),
        })

        self.router = Lanelet2Router(LANELET2_OSM)
        self.goal = GoalManager.from_carla_editor(
            GOAL_EDITOR_X, GOAL_EDITOR_Y, radius_m=GOAL_RADIUS_M)

        self._episode_steps = 0
        self._stuck_steps = 0
        self._last_action = np.zeros(3, dtype=np.float32)

        self.window = None
        self.window_size = 512

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.node.teleport(
            SPAWN_POINT["x"], SPAWN_POINT["y"], SPAWN_POINT["z"],
            yaw=math.radians(SPAWN_POINT["yaw"]),
        )

        # Async settle: hold the brake while the bridge applies the teleport
        # and the suspension settles.
        deadline = time.time() + RESET_SETTLE_S
        while time.time() < deadline:
            self.node.send_brake()
            time.sleep(STEP_DT)

        self.node.clear_event_flags()

        # Plan the route once per episode from the current odometry pose.
        ego_x, ego_y = float(self.node.pose_xy[0]), float(self.node.pose_xy[1])
        if not self.router.plan((ego_x, ego_y), (self.goal.x, self.goal.y)):
            print("⚠ Lanelet2 routing failed — falling back to straight-line bearing.")

        self.goal.reset(ego_x, ego_y)

        self._episode_steps = 0
        self._stuck_steps = 0
        self._last_action = np.zeros(3, dtype=np.float32)

        obs = self._build_obs()
        info = {
            "speed_kmh": self.node.speed_mps * 3.6,
            "dist_to_goal_m": self.goal._dist(ego_x, ego_y),
        }
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        # Throttle floor: while the policy is still random, force the
        # car to move forward so it can experience progress reward at all.
        # Steering and brake remain fully under the policy.
        if action[2] < 0.5:  # not actively braking
            action[1] = max(float(action[1]), THROTTLE_FLOOR)
        self.node.send_control(action)

        # Sensor-paced sync: wait for one fresh lidar revolution. Falls
        # back to a fixed sleep if the bridge stops publishing.
        if not self.node.wait_for_next_lidar(timeout=1.0):
            time.sleep(STEP_DT)
        self._episode_steps += 1

        obs = self._build_obs()
        reward, terminated, truncated, info = self._compute_reward_and_done(action)
        self._last_action = action
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Observation building
    # ------------------------------------------------------------------
    def _build_obs(self):
        # Lossless pass-through: BEV is rasterized at 256x256 directly.
        bev = self._lidar_to_bev(self.node.latest_lidar)              # (BEV_H, BEV_W, 3)

        sem = self.node.get_sem_cam()
        if sem is None:
            cam = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
        elif sem.shape[:2] != (CAM_H, CAM_W):
            # objects.json changed — fall back to a resize so training
            # doesn't crash on a shape mismatch.
            cam = cv2.resize(sem, (CAM_W, CAM_H), interpolation=cv2.INTER_AREA)
        else:
            cam = sem

        # State features. Read-only — goal.features() (which mutates the
        # progress tracker) is called once per step in _compute_reward_and_done.
        ego_x, ego_y = float(self.node.pose_xy[0]), float(self.node.pose_xy[1])
        ego_yaw = float(self.node.yaw)
        dist = self.goal._dist(ego_x, ego_y)
        bearing_to_goal = math.atan2(self.goal.y - ego_y, self.goal.x - ego_x)
        bearing_err = _wrap_pi(bearing_to_goal - ego_yaw)

        lateral, _route_heading, progress = self.router.query(ego_x, ego_y)

        speed = float(self.node.speed_mps)
        state = np.array([
            np.clip(dist / 100.0, 0.0, 10.0),
            math.sin(bearing_err),
            math.cos(bearing_err),
            np.clip(speed / 30.0, -1.0, 2.0),
            np.clip(lateral / 4.0, -1.0, 1.0),
            float(progress),
        ], dtype=np.float32)

        return {"camera": cam, "bev": bev, "state": state}

    # ------------------------------------------------------------------
    # Reward + termination
    # ------------------------------------------------------------------
    def _compute_reward_and_done(self, action):
        ego_x, ego_y = float(self.node.pose_xy[0]), float(self.node.pose_xy[1])
        ego_yaw = float(self.node.yaw)
        speed = float(self.node.speed_mps)

        dist, bearing_err, progress_delta = self.goal.features(ego_x, ego_y, ego_yaw)
        lateral, route_heading, route_progress = self.router.query(ego_x, ego_y)
        heading_err = abs(_wrap_pi(route_heading - ego_yaw)) if self.router.has_route() else 0.0

        # --- Speed term ---
        if speed <= TARGET_SPEED_MPS:
            speed_reward = W_SPEED * (speed / TARGET_SPEED_MPS)
        else:
            speed_reward = W_SPEED - W_OVERSPEED * (speed - TARGET_SPEED_MPS)

        steer_penalty = W_STEER * abs(float(action[0]) - float(self._last_action[0]))
        proximity_penalty = self._proximity_penalty()
        idle_penalty = W_IDLE if speed < IDLE_SPEED_MPS else 0.0

        # --- Goal-following terms ---
        progress_reward = W_PROGRESS * progress_delta  # +ve if got closer
        lateral_penalty = W_LATERAL * min(abs(lateral), 4.0)
        heading_penalty = W_HEADING * heading_err

        reward = (
            speed_reward
            + progress_reward
            - steer_penalty
            - proximity_penalty
            - idle_penalty
            - lateral_penalty
            - heading_penalty
            - W_TIME
        )

        terminated = False
        truncated = False
        info = {
            "speed_kmh": speed * 3.6,
            "dist_to_goal_m": dist,
            "lateral_m": lateral,
            "route_progress": route_progress,
        }

        # Goal reached → big positive, terminate.
        if self.goal.reached(ego_x, ego_y):
            reward += R_GOAL
            terminated = True
            info["termination"] = "goal"

        # Collision.
        elif self.node.collided and self.node.collision_intensity > COLLISION_INTENSITY_THRESHOLD:
            reward += R_COLLISION
            terminated = True
            info["termination"] = "collision"
            info["collision_intensity"] = self.node.collision_intensity

        # Stuck.
        if not terminated and speed < STUCK_SPEED_MPS:
            self._stuck_steps += 1
            if self._stuck_steps > STUCK_STEPS:
                reward += R_STUCK
                truncated = True
                info["termination"] = "stuck"
        else:
            self._stuck_steps = 0

        # Time limit.
        if not terminated and self._episode_steps > EPISODE_STEP_LIMIT:
            truncated = True
            info.setdefault("termination", "timeout")

        info["reward_components"] = {
            "speed": speed_reward,
            "progress": progress_reward,
            "steer": -steer_penalty,
            "proximity": -proximity_penalty,
            "idle": -idle_penalty,
            "lateral": -lateral_penalty,
            "heading": -heading_penalty,
            "time": -W_TIME,
        }
        return float(reward), terminated, truncated, info

    def _proximity_penalty(self):
        pts = self.node.latest_lidar
        if pts is None or len(pts) == 0 or pts.dtype.names is None:
            return 0.0
        x = pts['x']
        y = pts['y']
        tags = pts['ObjTag']
        mask = np.isin(tags, OBSTACLE_TAGS) & (x > 0.0)
        if not np.any(mask):
            return 0.0
        d_min = float(np.sqrt(x[mask] ** 2 + y[mask] ** 2).min())
        if d_min >= OBSTACLE_PROXIMITY_M:
            return 0.0
        return W_OBSTACLE * (1.0 - d_min / OBSTACLE_PROXIMITY_M)

    # ------------------------------------------------------------------
    # BEV rasterization
    # ------------------------------------------------------------------
    def _lidar_to_bev(self, points):
        bev = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        if points is None or len(points) == 0 or points.dtype.names is None:
            return bev

        x = points['x']
        y = points['y']
        tags = points['ObjTag']

        px = (-x / self.resolution + self.center).astype(np.int32)
        py = (y / self.resolution + self.center).astype(np.int32)

        valid = (px >= 0) & (px < self.grid_size) & (py >= 0) & (py < self.grid_size)
        px, py, tags = px[valid], py[valid], tags[valid]

        drivable = np.isin(tags, DRIVABLE_TAGS)
        obstacles = np.isin(tags, OBSTACLE_TAGS)
        statics = np.isin(tags, STATIC_TAGS)

        bev[px[drivable], py[drivable], 0] = 255
        bev[px[obstacles], py[obstacles], 1] = 255
        bev[px[statics], py[statics], 2] = 255
        return bev

    def close(self):
        if self.window is not None:
            import pygame
            pygame.quit()
            self.window = None


def _wrap_pi(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a
