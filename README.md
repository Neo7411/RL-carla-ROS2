# RL-carla-ROS2 — deep-dive guide

A goal-conditioned reinforcement-learning driving agent for the **CARLA** simulator, connected over **ROS 2 Humble** and trained with **Stable-Baselines3 PPO**. The agent learns to drive an ego vehicle from a fixed spawn point to random reachable goals in Town04, using a bird's-eye-view (BEV) LiDAR representation plus a compact ego/goal state vector. Route planning uses the **Lanelet2** map shipped in `agent/maps/Town04.osm`.

This README is the long-form project documentation. Every file has its own section that explains not just *what* the code does but *why* every design decision was made — including the mistakes we ran into and how we worked around them.

---

## Table of contents

1. [System architecture](#1-system-architecture)
2. [Repository layout](#2-repository-layout)
3. [Data flow — one environment step end-to-end](#3-data-flow--one-environment-step-end-to-end)
4. [`tools/bev_processor.py` — LiDAR to BEV tensor](#4-toolsbev_processorpy--lidar-to-bev-tensor)
5. [`tools/map_router.py` — Lanelet2 routing](#5-toolsmap_routerpy--lanelet2-routing)
6. [`tools/carla_env.py` — Gymnasium environment](#6-toolscarla_envpy--gymnasium-environment)
7. [`tools/custom_cnn.py` — feature extractor](#7-toolscustom_cnnpy--feature-extractor)
8. [`train.py` — PPO training loop](#8-trainpy--ppo-training-loop)
9. [Observation / action / reward specification](#9-observation--action--reward-specification)
10. [Goal detection: how the agent knows it has arrived](#10-goal-detection-how-the-agent-knows-it-has-arrived)
11. [How the agent perceives the map and decides where to go](#11-how-the-agent-perceives-the-map-and-decides-where-to-go)
12. [Known quirks, bugs we fixed, and design decisions](#12-known-quirks-bugs-we-fixed-and-design-decisions)
13. [Running, monitoring, and debugging](#13-running-monitoring-and-debugging)
14. [Roadmap — what comes next](#14-roadmap--what-comes-next)

---

## 1. System architecture

```
                +-----------------------+
                |  CARLA simulator      |
                |   (Town04 + ego)      |
                +----------+------------+
                           | carla-ros-bridge
                           v
   ROS 2 topics (sensors in, control out):
     /carla/ego_vehicle/lidar                  (PointCloud2)
     /carla/ego_vehicle/odometry               (Odometry)
     /carla/ego_vehicle/collision              (CarlaCollisionEvent)
     /carla/ego_vehicle/vehicle_control_cmd    (CarlaEgoVehicleControl)
     /carla/ego_vehicle/control/set_transform  (Pose — teleport / respawn)
                           |
                           v
   +----------------------------------------------+
   | CarlaRLEnvironment (Gymnasium)               |
   |                                              |
   |  obs = {                                     |
   |    bev:    [256, 256, 6]  float32            |
   |    vector: [6]            float32            |
   |  }                                           |
   |  action = [steer, throttle/brake]  ∈ [-1,1]  |
   |                                              |
   |  reward: progress + speed² − collision       |
   +----------+-----------------------------------+
              |
              v
   +----------------------------------------------+
   |  SB3 PPO  (MultiInputPolicy)                 |
   |    BEVGoalExtractor:                         |
   |      CNN branch (preserves spatial layout)   |
   |      MLP branch (ego / goal vector)          |
   |    pi/vf heads: [256, 256]                   |
   +----------------------------------------------+

   Side channel:
   Lanelet2 map (Town04.osm)
     │
     └─> MapRouter ── planned route (waypoints) ──> rendered into BEV channel 5
                                                 └─> progress / dist-to-goal features
                                                 └─> goal-reached detection
```

**Key separation of concerns**

- CARLA does physics and sensor simulation.
- The ROS 2 bridge exposes topics; our code **only** talks to ROS 2, never to CARLA directly.
- `BEVProcessor` turns a raw LiDAR point cloud into a spatial tensor.
- `MapRouter` plans routes, and answers "how far to the goal / am I on the road / is the goal reached".
- `CarlaRLEnvironment` glues these together into a Gymnasium `Env`.
- `BEVGoalExtractor` is the neural feature extractor consumed by SB3 PPO.

---

## 2. Repository layout

```
RL-carla-ROS2/
├── README.md                      <— this file
└── agent/
    ├── train.py                   PPO training entry point
    ├── requirements.txt           pinned Python deps (torch, SB3, gymnasium)
    ├── maps/
    │   └── Town04.osm             Lanelet2 map (OSM XML) — 462 lanelets
    ├── tools/
    │   ├── carla_env.py           Gymnasium env (ROS 2 I/O, reward, resets)
    │   ├── bev_processor.py       LiDAR → BEV tensor + route rasterizer
    │   ├── custom_cnn.py          BEVGoalExtractor (CNN + MLP feature extractor)
    │   └── map_router.py          Lanelet2 loader, routing graph, progress metrics
    ├── help/
    │   └── points.py              CARLA spawn-point visualizer (diagnostic utility)
    ├── checkpoints/               periodic PPO checkpoints (every 50 000 steps)
    └── ppo_carla_tensorboard/     TensorBoard logs
```

---

## 3. Data flow — one environment step end-to-end

Understanding a single `env.step(action)` call end-to-end is the single best way to grok the system.

1. **PPO samples an action** `[steer, throttle_or_brake]`, each in `[-1, 1]`.
2. `CarlaRLEnvironment.step()` calls `_publish_control(action)`, which:
   - Remaps `action[1]` through a throttle-bias curve (so `0 → 60 % throttle`).
   - Constructs a `CarlaEgoVehicleControl` with explicit `hand_brake=False`, `reverse=False`, `manual_gear_shift=False`, `gear=0` — *every time*, because the bridge latches old values otherwise.
   - Publishes to `/carla/ego_vehicle/vehicle_control_cmd`.
3. The env **waits** for a fresh LiDAR frame (`self._new_frame.wait(timeout=3.0)`), which is set by the LiDAR callback whenever a new point cloud arrives. This is how step pacing is synchronized with the simulator.
4. The LiDAR callback writes the raw point cloud into `self._latest_points` under a lock; the odometry callback writes ego `(x, y)` position, yaw (from quaternion), and speed; the collision callback flips a boolean.
5. `_build_observation()` is called:
   - Reads the latest point cloud + ego pose under the lock.
   - Calls `MapRouter.progress(ego_xy, ego_yaw)` to compute route progress metrics (cross-track error, heading error, distance to goal, completion fraction, goal-reached flag).
   - Projects the route polyline from world frame into LiDAR-local frame (ego-forward = +x).
   - Calls `BEVProcessor.process(points, route_local=route_local)` → returns a `(256, 256, 6)` float32 tensor.
   - Builds the 6-d `vector` observation by normalising `[speed, heading_err, cross_track, dist_to_goal, goal_dx_ego, goal_dy_ego]`.
   - Returns `{"bev": ..., "vector": ...}`.
6. `_compute_reward()` evaluates terminal events (collision / goal-reached) and dense shaping (progress, speed, heading).
7. Collision grace period: if a collision fires in the first 5 steps of the episode, it is ignored (the teleport itself fires spurious collisions).
8. `step()` returns `(obs, reward, terminated, truncated, info)`. `terminated` is only true on collision or goal-reached; 1500 steps triggers `truncated`.
9. SB3 appends the transition to the rollout buffer. After `n_steps=512` transitions, PPO computes advantages and does 10 epochs of gradient updates.

Most per-step latency is in **step 3** (waiting for a LiDAR frame over ROS 2). That is why GPU utilisation is currently ~15 %: the policy forward pass is <2 ms but the simulator runs at ~15 Hz.

---

## 4. `tools/bev_processor.py` — LiDAR to BEV tensor

Converts a raw `[N, 4]` LiDAR point cloud (`x, y, z, intensity`) into a 6-channel BEV tensor.

### Parameters
```python
BEVProcessor(
    x_range=(-128.0, 128.0),   # 256 m × 256 m footprint
    y_range=(-128.0, 128.0),
    z_range=(-2.0, 4.0),
    bev_height=256,
    bev_width=256,
)   # → 1 m per cell, 6 channels
```

### The 6 channels

| Index | Channel           | Computation                                                 | Why it matters                                |
| ----- | ----------------- | ----------------------------------------------------------- | --------------------------------------------- |
| 0     | Max height        | `np.maximum.at(bev[:,:,0], (yi, xi), z)`                    | Tall objects (trees, buildings, other cars)   |
| 1     | Height range      | max − min per cell                                          | Roofs vs. ground; helps distinguish cars      |
| 2     | Point density     | `np.add.at(bev[:,:,2], (yi, xi), 1); /= max`                | Dense returns = solid obstacle; sparse = noise |
| 3     | Mean intensity    | Σ intensity / count per cell                                | Retroreflective surfaces (road markings)      |
| 4     | Binary occupancy  | 1 where count > 0                                           | Simple "is there anything here"               |
| 5     | **Route mask**    | Rasterized planned route in ego-local frame                 | Tells the CNN where the goal direction is     |

### How projection works (vectorised, no Python loops)

1. `_crop(points)` keeps only points inside the 3D ROI (filters far returns and flying points).
2. `_compute_indices(points)` maps `(x, y)` → `(col, row)` grid coordinates.
3. `_project(points)` uses NumPy scatter ops (`np.maximum.at`, `np.add.at`) to fill channels 0–4 in one pass.

### Route channel (new for goal-conditioning)

`_rasterize_route(bev, route_local)` walks successive waypoints in LiDAR-local frame and paints a 1 into channel 5 using line interpolation (`np.linspace` from `(x0, y0)` to `(x1, y1)`). This gives the CNN a pixel-perfect map of "the route I should follow" overlaid on the same grid as the obstacles. Critically, because the route is in **ego-local** frame, the CNN sees it translate and rotate naturally as the car moves.

### Why 256 × 256 at ±128 m?

- Original was 200 × 200 at ±50 m (0.5 m/cell). That was only 100 m of look-ahead — too short once routes run 200+ m.
- We picked 256² at ±128 m (1 m/cell) as a middle ground: doubles the spatial horizon, keeps resolution roughly comparable, only 1.64× more pixels than before.
- Caveat: if the carla-ros-bridge LiDAR `range` is capped below 128 m, the outer half of the BEV will always be empty.

---

## 5. `tools/map_router.py` — Lanelet2 routing

### Purpose

- Load a Lanelet2/OSM map.
- Build a `RoutingGraph` for vehicle traffic rules.
- Sample random reachable goals and densify the shortest path into a waypoint polyline.
- Provide per-step progress metrics, including the **goal-reached** test.

### Loading the map

```python
self._projector = UtmProjector(Origin(0.0, 0.0))
self._map       = load(osm_path, self._projector)
rules           = create_rules(Locations.Germany, Participants.Vehicle)
self._graph     = lanelet2.routing.RoutingGraph(self._map, rules)
```

The `.osm` file stores lat/lon around `(0, 0)`, which is treated as a tangent plane so that world XY is already in meters. Traffic rules are configured for vehicles in the "Germany" location profile — the Lanelet2 default — which is enough for shortest-path routing.

### Coordinate convention: `FLIP_Y = True`

CARLA uses a **left-handed** world frame (x forward, y right, yaw clockwise). Lanelet2/OSM stores ENU-like right-handed coords. `MapRouter._xy(point)` applies `y_carla = -y_lanelet` so **every XY leaving the module is already in CARLA frame**. Yaw is *not* flipped inside MapRouter; callers have to be aware (see §11).

### Picking a random goal

`plan_random_route(start_xy, min_len, max_len, max_tries, rng)` does, in order:
1. Find the lanelet whose centerline is closest to `start_xy`.
2. Sample a random target lanelet up to `max_tries` times.
3. Ask the routing graph for a shortest path; if one exists and is 2+ lanelets long, densify it with `_densify(path)`.
4. If total length is in `[min_len, max_len]`, accept it.
5. If nothing fit the length filter, accept *any* valid route as a fallback.
6. If even that failed, synthesize a 60 m straight line in the current heading — guarantees `progress()` never crashes.

### Densification

`_densify(path)` concatenates lanelet centerline points into a polyline, then resamples at `waypoint_spacing=2 m` by linear interpolation. Every waypoint is stored as a 2-D numpy row `[x, y]`.

### Route progress

`progress(ego_xy, ego_yaw) -> Progress`:
1. Advances an internal cursor `_wp_idx` to the waypoint nearest the car (within a 50-waypoint look-ahead window).
2. Computes **cross-track error** by projecting the ego onto the current waypoint segment (signed via 2-D cross product; left is positive).
3. Computes **heading error** = `seg_yaw − ego_yaw`, wrapped to `[-π, π]`.
4. Computes **remaining arc length** = total − cumulative-at-wp-idx + distance-from-wp-to-ego.
5. Sets `goal_reached = True` if `ego` is within `goal_radius = 5 m` of the last waypoint.
6. Sets `off_route = True` if `|cross_track| > 25 m` (raised because the spawn is ~14 m off the Lanelet2 centerline in Town04).

### `heading_at(xy)`

Returns the lane-tangent heading at `xy` by finding the nearest centerline segment. **Caveat**: at some points the nearest segment is on an adjacent bend, giving a heading that looks wrong visually. For our fixed spawn we therefore hardcode `start_yaw = 0` in `carla_env.py` instead of calling `heading_at`.

---

## 6. `tools/carla_env.py` — Gymnasium environment

This is the glue layer. It:
- Subscribes to LiDAR / odometry / collision ROS 2 topics.
- Publishes control + teleport commands.
- Builds the Dict observation.
- Computes the reward.
- Manages episode lifecycle (reset / step / termination).

### Observation space

```python
Dict({
  "bev":    Box(low=0, high=10, shape=(256, 256, 6), dtype=float32),
  "vector": Box(low=-1, high=1, shape=(6,),           dtype=float32),
})
```

### Vector observation (6 floats)

| Index | Feature           | Normalisation                        |
| ----- | ----------------- | ------------------------------------ |
| 0     | Speed             | `v / SPEED_MAX`  (15 m/s)            |
| 1     | Heading error     | `err / π`  rad                       |
| 2     | Cross-track err.  | `err / 5`  m                         |
| 3     | Distance to goal  | `dist / 300`  m (arc length)         |
| 4     | Goal Δx (ego-rel) | `dx / 50`  m (clipped)               |
| 5     | Goal Δy (ego-rel) | `dy / 50`  m (clipped)               |

### Action space

```python
Box(low=-1, high=1, shape=(2,), dtype=float32)
```

| Index | Meaning                           |
| ----- | --------------------------------- |
| 0     | Steer (−1 left, +1 right)         |
| 1     | Signed throttle/brake → remapped  |

#### Throttle-bias curve (critical)

A PPO policy at initialisation samples `action[1] ≈ 0`. If `0` mapped to "idle", the car would never move and the policy would learn nothing. We therefore remap:

| `action[1]`          | Control                        |
| -------------------- | ------------------------------ |
| `+1.0`               | Full throttle (1.0)            |
|  `0.0`               | **0.60 throttle** (car rolls)  |
| `-0.7`               | Zero throttle, zero brake      |
| `-1.0`               | Full brake (1.0)               |

So the policy has to *actively* output `action[1] < -0.7` to brake, otherwise it drives. This was the single biggest change that got the agent to start moving.

### Episode lifecycle — `reset()`

1. Reset internal counters (`_step_count = 0`, `_prev_fraction = 0`, `_collision = False`).
2. Hardcode `start_xy = (290.2, 168.9)`, `start_yaw = 0.0`. Lanelet-heading-based yaw was off visually, so we use yaw=0 (the road here runs east-west).
3. Plan a random route from `start_xy`. Print `[env] new episode — goal=(..., ...) route_len=... waypoints=...`.
4. Publish respawn Pose (with unit quaternion `(0, 0, sin(yaw/2), cos(yaw/2))`).
5. Publish a brief full-brake, sleep 1 s, then publish a zero-brake release so the bridge's last-latched command is cleared.
6. Wait ~5 s for the first LiDAR frame after respawn.
7. **Clear `_collision = False` again** — the teleport itself can fire a spurious collision event during the 1-s sleeps; clearing after the waits prevents a same-frame termination.
8. Initialise `_prev_fraction` from the first `progress()` call so step 1's dense progress reward is zero (prevents a fake spike).
9. Return `(obs, info)`.

### Episode lifecycle — `step(action)`

1. Publish the action (with throttle bias).
2. Wait for a fresh LiDAR frame.
3. Build observation, read ego state under the lock.
4. Compute `progress()`.
5. **Collision grace period**: for the first 5 steps of the episode, a collision is swallowed (reset the flag and treat `c = False`). Prevents empty double-reset episodes.
6. Compute reward.
7. `terminated = collision or goal_reached`; `truncated = step_count >= 1500`.
8. Return standard gym 5-tuple.

### Design decisions that matter

- **Shared state guarded by a threading lock.** ROS 2 callbacks run on the executor thread; `reset` / `step` run on the main thread. Every write to `_latest_points`, `_latest_velocity`, `_ego_xy`, `_ego_yaw`, `_collision` is inside `with self._lock:`.
- **No autopilot, no traffic manager.** The bridge has `/carla/ego_vehicle/enable_autopilot` — we never touch it. Control is 100 % the agent's.
- **Hand-brake workaround.** Every control message explicitly sets `hand_brake=False`. Without this, the bridge kept the last value and the car stayed locked after a reset-time full brake.

---

## 7. `tools/custom_cnn.py` — feature extractor

SB3's `MultiInputPolicy` expects a `BaseFeaturesExtractor` that consumes the Dict observation and returns a single feature vector.

### `BEVGoalExtractor`

```
 bev [B, 256, 256, 6]              vector [B, 6]
      │ permute NHWC → NCHW             │
      ▼                                  ▼
 Conv2d 6→32      stride 2 + ReLU   Linear  6 → 64  + ReLU
 Conv2d 32→64     stride 2 + ReLU   Linear 64 → 64  + ReLU
 Conv2d 64→128    stride 2 + ReLU        │
 Conv2d 128→256   stride 2 + ReLU        │
 Conv2d 256→256   stride 2 + ReLU        │
 Flatten → Linear 16384 → 512 + ReLU     │
 Linear 512 → 512 + ReLU                 │
           └──────── concat ─────────────┘
                       │
                       ▼  features_dim = 576
         PPO policy & value heads [256, 256]
```

**~9.6 M parameters.**

### Why no `AdaptiveAvgPool2d`?

An earlier version ended the CNN with `AdaptiveAvgPool2d((1, 1))` (global average pool → 256-dim vector). That collapsed all spatial information: the policy could only tell "there is an obstacle somewhere" but not **where**. After removing it and flattening the 8 × 8 feature map, the policy can reason about obstacle layout — crucial for dynamic avoidance.

### Why not just pool less aggressively?

You could use `AdaptiveAvgPool2d((4, 4))` for a compromise. We went all the way (flatten the whole grid) because the GPU was at 15 % utilisation; paying a larger FC head is free compute.

### `cnn_dim` and `mlp_dim`

- `cnn_dim = 512` — output dim of the CNN branch.
- `mlp_dim = 64` — output dim of the vector branch.
- `features_dim = cnn_dim + mlp_dim = 576` — the concatenated feature vector handed to SB3's policy heads (two-layer MLP of 256-256).

---

## 8. `train.py` — PPO training loop

### What it does

1. Initialise ROS 2 + spin the node in a background thread.
2. Instantiate `CarlaRLEnvironment`.
3. Either load `carla_ppo_goal.zip` (resume) or create a fresh PPO model.
4. Attach callbacks (checkpoint every 50k steps, entropy-coefficient scheduler).
5. Call `model.learn(total_timesteps=10_000_000)`.

### Hyperparameters

| Hyperparameter | Value                                   | Reason                                          |
| -------------- | --------------------------------------- | ----------------------------------------------- |
| policy         | `MultiInputPolicy` with `BEVGoalExtractor` | Dict observation                              |
| learning_rate  | **linear schedule 3e-4 → 1e-5**         | High LR early for exploration, low late for fine-tuning |
| n_steps        | **512** (was 2048)                      | 4× more gradient updates per wall-clock hour   |
| batch_size     | 64                                      | Default; works fine                             |
| n_epochs       | 10                                      | Default                                         |
| gamma          | 0.99                                    | Standard continuous-control                     |
| gae_lambda     | 0.95                                    | Default                                         |
| ent_coef       | **scheduled 0.01 → 0.001 via callback** | Explore early, commit late                      |
| clip_range     | 0.2                                     | Default                                         |
| log_std_init   | **−0.5** (default 0.0)                  | Less jittery action sampling; policy keeps speed |
| total_timesteps| 10 000 000                              | Plenty of budget                                |

### `EntropyCoefScheduler`

SB3's `ent_coef` is a scalar attribute on the model. The custom callback sets it each step based on `num_timesteps / total_steps` — linear interpolation from 0.01 to 0.001. This mimics a standard RL exploration schedule without forking SB3.

### `CheckpointCallback`

Saves `checkpoints/carla_ppo_goal_<step>_steps.zip` every 50 000 steps. The top-level `carla_ppo_goal.zip` is only created on `KeyboardInterrupt` — that is the file the auto-resume logic looks for.

### Resume logic

```python
if os.path.exists(MODEL_PATH):
    model = PPO.load(MODEL_PATH, env=env, tensorboard_log=LOG_DIR)
```

**Resume is safe only when the observation space, action space, and extractor architecture haven't changed.** If any of those change, delete `carla_ppo_goal.zip` (intermediate `checkpoints/*.zip` don't auto-resume, so they can stay as history). Reward function, hyperparams, and termination rules can all change freely without breaking an old checkpoint.

---

## 9. Observation / action / reward specification

### Reward function (current)

**Terminal events**

| Event         | Reward                    | Ends episode? |
| ------------- | ------------------------- | ------------- |
| Collision     | −30 − 2 × velocity        | yes (except in first 5 steps — grace) |
| Goal reached  | +400 + 20 × velocity      | yes           |
| Step cap      | no extra reward           | truncated at 1500 steps |

**Dense per-step**

| Term           | Formula                                               | Role                                       |
| -------------- | ----------------------------------------------------- | ------------------------------------------ |
| Route progress | `10 × Δ(fraction_done) × total_route_length`          | **Dominant goal-seeking signal**           |
| Speed          | `0.2 × v + 0.05 × v²`                                 | Quadratic — strongly prefers high speed    |
| Heading err.   | `−0.1 × |heading_err|`                                | Weak nudge to face route direction         |

**Design notes**

- Collision penalty was softened from −200 → −30 while the policy is learning to drive. Fear of crashes was preventing it from exploring throttle. Plan to ramp back up once it reliably drives.
- **No idle / cross-track / lane-keeping penalty.** Idle penalty paradoxically made the policy end episodes fast (brake → crash → end). Lane-keeping would conflict with the planned future of dynamic obstacle avoidance, which may require lane-crossing evasive maneuvers.
- **Off-route no longer terminates.** The car can freely leave the planned route; it loses the dense progress signal but can still earn the goal-reach bonus.

---

## 10. Goal detection: how the agent knows it has arrived

This is one of the most common "does it actually work?" questions, so it gets its own section.

### The check is purely geometric

Inside `MapRouter.progress()`:

```python
goal_xy = self._waypoints[-1]                     # last densified waypoint
goal_dist = np.linalg.norm(ego_xy - goal_xy)
goal_reached = goal_dist < self.goal_radius       # default 5.0 m
```

So:

1. After `plan_random_route()`, the route is a polyline of waypoints spaced ~2 m apart. The **last waypoint** is the goal.
2. On every `step()`, the env reads the ego's XY from odometry and calls `progress()`.
3. `progress()` computes Euclidean distance from the ego to the last waypoint.
4. If that distance is under 5 m, `goal_reached = True` is returned.
5. Back in `carla_env.py`, this triggers:
   - The goal-reach terminal reward `+400 + 20 × velocity`.
   - `terminated = True`.
   - `info["success"] = True` (aggregated by SB3's `ep_info_buffer`; shows up in TensorBoard if wrapped with `EvalCallback`).

### There is no heading / speed / stopping requirement

Touching the 5 m radius with *any* pose or velocity counts as success. If you want to tighten the criterion later (e.g. "must stop within 3 m facing lane direction"), extend `MapRouter.progress()` and guard the bonus in `_compute_reward()`.

### Per-episode logging

Every `reset()` prints:
```
[env] new episode — goal=(+271.3, -185.4)  route_len=187.5 m  waypoints=94
```

Use this to correlate TensorBoard spikes with specific episodes.

---

## 11. How the agent perceives the map and decides where to go

This section answers the question: *"Is the map loaded into the agent? How does the policy know where the goal is and which way to drive?"*

The short answer is that the **map is not given to the policy**. It stays in the environment, gets processed by a classical planner, and the output of that planner is distilled into two observation channels the neural network can actually consume. This is the standard autonomous-driving decomposition: **high-level map planner + low-level RL controller**.

### 11.1 Three layers of knowledge

| Layer | Who owns it | What it contains | Who sees it |
|---|---|---|---|
| **Map** | `MapRouter` (env) | Full Lanelet2 road network, all lanelets, connectivity | Only the planner |
| **Route** | `MapRouter.waypoints` | A densified 2D polyline from the spawn to the sampled goal | Env — used to build obs |
| **Observation** | `carla_env._build_observation()` | BEV tensor + 6-D vector, in ego-local frame | **The policy** |

The policy never reads the OSM file, never sees the lanelet graph, never calls the routing algorithm. It only sees the **result**: "given where you are right now, here is a line drawn on the ground you should follow, and here are six scalars describing your relationship to the goal."

### 11.2 Where the map is loaded

[agent/tools/carla_env.py:55-58](agent/tools/carla_env.py#L55-L58):
```python
if map_path is None:
    here = os.path.dirname(os.path.abspath(__file__))
    map_path = os.path.join(here, '..', 'maps', 'Town04.osm')
self._router = MapRouter(map_path)
```

This runs **once**, in the environment's constructor. Inside [map_router.py](agent/tools/map_router.py):

1. `lanelet2.io.load(path, UtmProjector(Origin(0, 0)))` parses the OSM file into a `LaneletMap`.
2. `traffic_rules.create(Locations.Germany, Participants.Vehicle)` creates a drivable-lane filter.
3. `routing.RoutingGraph(lanelet_map, rules)` builds a directed graph where nodes are lanelets and edges are legal transitions.

From this point onward, the map lives in memory as a `RoutingGraph` object. It never moves into the policy.

### 11.3 How a route is produced each episode

On every `reset()` — [carla_env.py:147-149](agent/tools/carla_env.py#L147-L149):
```python
self._router.plan_random_route(
    start_xy, min_len=20.0, max_len=1000.0,
    max_tries=60, rng=self._rng)
```

Inside `plan_random_route`:
1. Snap `start_xy` to the nearest drivable lanelet.
2. Pick a random other lanelet within a length bound.
3. Call `routing_graph.getRoute(start, goal)` → returns a sequence of connected lanelets.
4. Extract each lanelet's centerline and concatenate → a polyline of 2D points.
5. Densify to one waypoint every 2 m → `self._router.waypoints`, shape `(N, 2)`.

If routing fails (common on edge cases), fall back to a synthetic 60 m straight line forward. This guarantees every episode has a valid route.

`self._router.waypoints` is the **route** — a list of `(x, y)` world-frame points from spawn to goal. This is the bridge between map knowledge and observation.

### 11.4 How the route reaches the policy — channel 1 (spatial)

Every step, [carla_env.py:262-269](agent/tools/carla_env.py#L262-L269):
```python
wps = self._router.waypoints                          # world-frame (N, 2)
route_local = _world_to_local(wps, ego_xy, ego_yaw)   # ego-frame  (N, 2)
bev = self._bev.process(points, route_local=route_local)
```

`_world_to_local` subtracts the ego position and rotates by `-ego_yaw` so the route is expressed in the ego's frame (x-forward, y-left). Then [bev_processor.py](agent/tools/bev_processor.py):

1. Converts each local `(x, y)` waypoint to a BEV pixel via the same projection used for LiDAR points.
2. Draws line segments between consecutive pixels into **channel 5** of the `(256, 256, 6)` BEV tensor using a Bresenham-style interpolator.
3. Returns a 6-channel image where channels 0–4 are LiDAR-derived (occupancy, density, height, etc.) and channel 5 is the route mask.

Because the route is in ego-local frame and uses the same pixel grid as the LiDAR, **the CNN sees the obstacles and the desired path on the same canvas**. A convolutional filter can directly reason about "is there something between me and the route" or "does the route bend right here." This is the key perceptual trick.

### 11.5 How the route reaches the policy — channel 2 (scalar)

Spatial information is powerful but high-dimensional. Some signals are better delivered as compact scalars. [carla_env.py:278-292](agent/tools/carla_env.py#L278-L292):

```python
goal = self._router.waypoints[-1]
rel = _world_to_local(goal[None, :], ego_xy, ego_yaw)[0]
vec = np.array([
    np.clip(speed / SPEED_MAX, -1.0, 1.0),                 # how fast am I
    np.clip(prog.heading_err / math.pi, -1.0, 1.0),        # angle to route tangent
    np.clip(prog.cross_track / 5.0, -1.0, 1.0),            # lateral offset from route
    np.clip(prog.dist_to_goal / DIST_MAX, 0.0, 1.0),       # straight-line metres to goal
    np.clip(rel[0] / EGO_REL_MAX, -1.0, 1.0),              # goal x in ego frame
    np.clip(rel[1] / EGO_REL_MAX, -1.0, 1.0),              # goal y in ego frame
], dtype=np.float32)
```

These six numbers give the policy **direct numerical access** to the goal's position and its own error relative to the plan. The MLP branch of `BEVGoalExtractor` consumes them. Unlike the spatial channel, this representation is robust even when the route falls outside the BEV viewport (e.g., goal is 200 m away, while the BEV only shows ±128 m).

### 11.6 What `MapRouter.progress()` computes

Called every step on the current `(ego_xy, ego_yaw)`. Produces a `Progress` dataclass with:

- `nearest_idx` — index of the waypoint closest to the ego.
- `cross_track` — signed lateral distance from the route polyline (metres).
- `heading_err` — angle between the ego yaw and the route tangent at `nearest_idx` (radians, wrapped to `[-π, π]`).
- `dist_to_goal` — straight-line Euclidean distance to `waypoints[-1]`.
- `fraction_done` — fraction of arc-length traversed along the route.
- `goal_reached` — `True` if `dist_to_goal < 5.0 m`.

The environment uses `fraction_done` to compute a **dense per-step progress reward** (`10 × Δm` along the route) and `goal_reached` as a **terminal success signal**.

### 11.7 Putting it together — what the policy sees vs. what it controls

```
   MAP (Town04.osm)
          │
          │  Lanelet2 load + routing graph
          ▼
   ROUTE (N, 2) world-frame waypoints ────────────────────┐
          │                                               │
          │  world→ego transform                          │
          ▼                                               │
   ROUTE_LOCAL (N, 2) ego-frame ─────────┐                │
          │                              │                │
          │  rasterize into channel 5    │  progress()    │
          ▼                              ▼                ▼
   BEV tensor (256, 256, 6)        (heading_err,       dist_to_goal,
     ch 0-4: LiDAR                  cross_track)       goal_dx_ego,
     ch 5  : route mask                                goal_dy_ego
          │                                      │
          ▼                                      ▼
        CNN ────────► 512-D feat   MLP ────► 64-D feat
                  concat → 576-D
                        │
                        ▼
                   pi / vf heads
                        │
                        ▼
               action = [steer, throttle]
```

The policy never knows the word "lanelet", never queries the map, never sees world coordinates. Its entire mental model of "where am I supposed to go" is:

- a thin line drawn on a 256×256 image, redrawn every step from the ego's current position, and
- six numbers summarising how off-course and how far from the goal it is.

### 11.8 Why this design generalises

- **New town, same policy.** Swap `Town04.osm` for another Lanelet2 map and the planner produces routes with identical structure. The policy keeps working — it only learned to follow a line and minimise a goal vector, neither of which is Town04-specific.
- **Rotation invariance.** Because everything is in ego-local frame, the agent does not have to learn "north is up" or "roads on this map go east-west". Any world-frame rotation of the scene produces the same observation.
- **Obstacle avoidance composes naturally.** When NPC traffic is added (on the roadmap), moving vehicles show up in LiDAR channels 0–4. The agent already reasons about "is there something between me and the route," so the same CNN features can handle dynamic obstacles — the route is a soft target, not a hard constraint.

### 11.9 What the agent cannot see

Being explicit about limitations:
- **No lane information.** The policy does not know which lane it is in or where lane boundaries are. This is deliberate — the [training objective](#9-observation--action--reward-specification) forbids lane-keeping priors, because the long-term goal is traffic-aware avoidance that sometimes requires crossing lanes.
- **No traffic lights / signs.** Nothing in the observation encodes signals. Adding this would require new obs channels.
- **No global localisation.** The policy only has ego-local information. It has no idea what world coordinates it is at. The goal direction tells it where to go relative to itself, which is sufficient.
- **No lookahead beyond BEV.** If the route exits the ±128 m BEV viewport, the policy falls back to the scalar goal vector. Very long routes therefore feel like "drive forward until the goal is visible."

---

## 12. Known quirks, bugs we fixed, and design decisions

These are the "why does this line exist" moments. Each one is a real bug we ran into.

### Coordinate frames

- **`FLIP_Y = True`** in `MapRouter`. Lanelet2 is right-handed ENU; CARLA is left-handed. We negate `y` inside the router so everything outside sees CARLA frame.
- **Yaw isn't flipped inside `MapRouter`.** We initially called `heading_at(spawn)` and used it directly — the car spawned ~23° off-axis. Negating it fixed the sign for some spawns but not others. For the current spawn `(290.2, 168.9)` the nearest lanelet segment heading is on a bend that doesn't match the actual road heading, so we now **hardcode `start_yaw = 0`**.

### Quaternion gotcha

For a while the code had `Quaternion(x=0, y=0, z=qz, w=180.0)` — a non-unit quaternion. Most consumers either normalise silently (collapsing `qz` to ~0, yielding identity rotation) or reject it. This "coincidentally" worked because identity happened to be the correct orientation. The proper form is `Quaternion(x=0, y=0, z=sin(yaw/2), w=cos(yaw/2))`.

### Hand-brake latch

`CarlaEgoVehicleControl` fields that you don't set are zero-initialised, which is *mostly* fine — but the bridge internally keeps the last value of `hand_brake` and `manual_gear_shift`. We saw the car stay locked after a reset-time full-brake because `hand_brake=True` from a stop command persisted. The fix is to set `hand_brake=False`, `reverse=False`, `manual_gear_shift=False`, `gear=0` on *every* control message.

### Spurious collisions on teleport

When the bridge snaps the ego to a new transform, the physics engine can register a brief collision (ground contact, near-actor overlap). The `CarlaCollisionEvent` fires during the reset's sleep windows → `_collision=True` → step 1 terminates → SB3 calls `reset()` again → infinite double-reset.

Two fixes together:
- **Clear `_collision = False` *after* the respawn sleeps** — not just before.
- **5-step collision grace period** — any collision in the first 5 steps of a new episode is swallowed (flag reset, `c=False`).

### Fake progress spike at step 1

`progress_m = Δfraction_done × total_length`. At `reset()`, `_prev_fraction = 0`. On step 1, `progress()` snaps `_wp_idx` to the waypoint nearest the ego — which can be well into the route. That produced a huge one-off "progress" reward on every episode.

Fix: call `progress()` once at the end of `reset()` purely to **initialise `_prev_fraction`**, so step 1 sees `Δ = 0`.

### 14-metre spawn-to-centerline offset

At `(290.2, 168.9)` the nearest Lanelet2 centerline point is ~14 m away. That is a real mismatch between the CARLA asset and the Lanelet2 map — not a bug in our code. We accommodate it by:
- Raising `off_route_thresh` to 25 m.
- Snapping the route cursor to the waypoint nearest the spawn in `_set_route(start_xy=...)`, so cross-track is measured against the correct part of the route.
- Removing the lane-keeping penalty entirely.

### Network design: no global pool

See §7. Global average pool collapsed spatial layout; removing it was a big quality win.

### Throttle bias

See §6. Without the `action[1]=0 → 60 % throttle` remap, PPO's initial near-zero sampling keeps the car stationary and training stalls.

### ROS 2 is the FPS ceiling

We currently get ~15 FPS end-to-end. The GPU is at ~15 % utilisation. Bigger networks won't help — the bottleneck is the LiDAR publish rate. The only way to increase throughput is **horizontal scaling** (multiple CARLA instances in parallel).

---

## 13. Running, monitoring, and debugging

### Prerequisites

- ROS 2 Humble (provides `lanelet2`, `rclpy`, sensor / nav / geometry msgs).
- CARLA simulator + carla-ros-bridge (provides `carla_msgs`).
- Python deps in `agent/requirements.txt` (PyTorch cu118, SB3 extras, gymnasium).

### Launch

```bash
# 1. Start CARLA + the ROS 2 bridge (Town04 + ego_vehicle spawned)

# 2. Activate the Python env that has torch / SB3 / gymnasium
conda activate carla_rl

# 3. From the agent/ directory:
cd agent
python train.py
```

### Monitor

```bash
tensorboard --logdir agent/ppo_carla_tensorboard/
```

**Scalars worth watching**

| Scalar                         | Target                    | Interpretation                                     |
| ------------------------------ | ------------------------- | -------------------------------------------------- |
| `rollout/ep_rew_mean`          | trending up               | Dominant health signal                             |
| `rollout/ep_len_mean`          | trending up, toward 1500  | Short episodes = policy is dying                   |
| `train/explained_variance`     | > 0.5 ideally             | Value function quality                             |
| `train/value_loss`             | stable, not exploding     | If it blows up, reward is too noisy                |
| `train/std`                    | dropping over time        | Policy is becoming decisive                        |
| `train/approx_kl`              | 0.005 – 0.02              | Much higher = policy is moving too fast            |
| `train/clip_fraction`          | < 0.1 late-stage          | Much higher = LR too high or reward discontinuous  |

### Things to try when the agent is stuck

| Symptom                           | Likely cause                     | Fix                                      |
| --------------------------------- | -------------------------------- | ---------------------------------------- |
| Car doesn't move at all           | Throttle bias wrong              | See `_publish_control` curve             |
| Episodes all ~5–10 steps          | Spurious collisions / respawn bug | Check the grace period                   |
| Reward rising but ep_len falling  | Reward exploit on reset          | Check `_prev_fraction` initialisation    |
| `value_loss` exploding            | Non-stationary reward            | Smooth reward signal; lower its range    |
| GPU stuck at ~15 %                | ROS 2 bottleneck                 | Scale horizontally; not a model issue    |

---

## 14. Roadmap — what comes next

The baseline is in place. Worthwhile next steps, roughly in priority order:

1. **Dynamic NPC traffic.** The env's observation and reward are already ready — BEV channels 0-4 pick up moving cars for free. Spawn NPCs via CARLA's traffic manager and test.
2. **Frame-stacking (3 BEV frames).** A single BEV doesn't encode motion; stacked BEVs let the CNN infer velocity of other actors — essential for dynamic avoidance.
3. **Vectorised envs (`SubprocVecEnv`).** Multiple CARLA instances → linear throughput win. Biggest compute unblocker.
4. **Evaluation callback.** Deterministic rollout every 50k steps, logging success rate / average goal distance / collision rate to TensorBoard scalars.
5. **Curriculum.** Start with 20-50 m routes, raise to 200+ m after success-rate > 50 %.
6. **LiDAR range check.** Confirm the bridge is configured for ≥ 128 m range, otherwise the BEV periphery is always empty.
7. **Stricter goal criterion.** Optional tighter test (must stop within 3 m, heading within ±15° of the route tangent).
8. **Record videos during eval.** A ROS 2 bag + rviz2 overlay of the route + BEV is invaluable for post-hoc debugging.

---

If you made it this far, you now know the system top to bottom. Every file has a single, clean job. The next big lever is **throughput** (horizontal scaling) and **signal** (frame stacks, dynamic traffic) — not model size.
