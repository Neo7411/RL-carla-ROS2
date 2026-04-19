import numpy as np
import threading
import time
import gymnasium as gym
from gymnasium import spaces

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose

from tools.bev_processor import BEVProcessor

try:
    from carla_msgs.msg import CarlaEgoVehicleControl, CarlaCollisionEvent
    CARLA_MSGS_AVAILABLE = True
except ImportError:
    CARLA_MSGS_AVAILABLE = False

class CarlaRLEnvironment(gym.Env):
    def __init__(self, ros_node: Node, bev_processor: BEVProcessor = None):
        super().__init__()
        self._node = ros_node
        self._bev = bev_processor or BEVProcessor()

        self.observation_space = spaces.Box(
            low=0.0, high=10.0, shape=self._bev.observation_shape, dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        self._latest_points = None
        self._latest_velocity = 0.0
        self._collision = False
        self._lock = threading.Lock()
        self._new_frame = threading.Event()
        
        self._step_count = 0
        self._max_steps = 1000

        self._node.create_subscription(PointCloud2, '/carla/ego_vehicle/lidar', self._on_lidar, 10)
        self._node.create_subscription(Odometry, '/carla/ego_vehicle/odometry', self._on_odom, 10)
        
        if CARLA_MSGS_AVAILABLE:
            self._node.create_subscription(CarlaCollisionEvent, '/carla/ego_vehicle/collision', self._on_collision, 10)
            self._control_pub = self._node.create_publisher(CarlaEgoVehicleControl, '/carla/ego_vehicle/vehicle_control_cmd', 10)
            self._respawn_pub = self._node.create_publisher(Pose, '/carla/ego_vehicle/control/set_transform', 10)

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
        speed = np.sqrt(vx**2 + vy**2)
        with self._lock:
            self._latest_velocity = speed

    def _on_collision(self, msg):
        with self._lock:
            self._collision = True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self._collision = False

        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = 303.2, 172.4, 1.0
        pose.orientation.w = 1.0
        if CARLA_MSGS_AVAILABLE:
            self._respawn_pub.publish(pose)
            stop = CarlaEgoVehicleControl(brake=1.0, hand_brake=True)
            self._control_pub.publish(stop)

        time.sleep(0.5)
        self._new_frame.clear()
        self._new_frame.wait(timeout=5.0)
        return self._build_observation(), {}

    def step(self, action):
        self._step_count += 1
        self._publish_control(action)

        self._new_frame.clear()
        self._new_frame.wait(timeout=3.0)
        
        obs = self._build_observation()
        with self._lock:
            v, c = self._latest_velocity, self._collision
        
        reward = self._compute_reward(v, c, action)
        terminated = c
        truncated = self._step_count >= self._max_steps
        
        return obs, reward, terminated, truncated, {}

    def _build_observation(self):
        with self._lock:
            if self._latest_points is not None:
                return self._bev.process(self._latest_points)
            return np.zeros(self._bev.observation_shape, dtype=np.float32)

    def _publish_control(self, action):
        if not CARLA_MSGS_AVAILABLE: return
        ctrl = CarlaEgoVehicleControl()
        # Steering
        ctrl.steer = float(np.clip(action[0], -1.0, 1.0))
        # Throttle/Brake: action[1] > 0 is throttle, < 0 is brake
        if action[1] > 0:
            ctrl.throttle = float(np.clip(action[1], 0.0, 1.0))
            ctrl.brake = 0.0
        else:
            ctrl.throttle = 0.0
            ctrl.brake = float(np.clip(-action[1], 0.0, 1.0))
        self._control_pub.publish(ctrl)

    def _compute_reward(self, velocity, collision, action):
        if collision:
            return -100.0
        
        # INCREASED speed reward: velocity is in m/s
        reward = velocity * 2.0 
        
        # Penalize standing still or very slow movement
        if velocity < 1.0:
            reward -= 5.0
            
        # Small penalty for extreme steering to keep the car straight
        reward -= abs(action[0]) * 0.1
        
        return float(reward)