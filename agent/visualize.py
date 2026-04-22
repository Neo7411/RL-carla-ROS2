"""
User-friendly live visualization of the CARLA BEV-driving agent.

Layout:
  ┌─────────────────────────────┬──────────────────────────────┐
  │                             │   SPEED  (big digits)        │
  │   BEV obstacle map          │   [================    ] km/h│
  │   (ego at centre, facing    │                              │
  │    up; obstacles in red,    │   THROTTLE  [========  ]     │
  │    free space in grey)      │   BRAKE     [          ]     │
  │                             │   STEER     [    |====>]     │
  │                             │                              │
  │                             │   STATUS: driving / crashed  │
  └─────────────────────────────┴──────────────────────────────┘

Drives the car with either:
  - a trained PPO policy (if --policy path is passed)
  - a constant "roll forward" action (default — pipeline sanity check)

Usage:
    python visualize.py                         # constant-throttle baseline
    python visualize.py --policy carla_ppo.zip  # use trained policy
    python visualize.py --episodes 5 --no-window --save-mp4
"""
from __future__ import annotations

import argparse
import os
import threading

import numpy as np
import rclpy

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow
from matplotlib.gridspec import GridSpec

from tools.bev_processor import BEVProcessor
from tools.carla_env import CarlaRLEnvironment, SPEED_TARGET, GOAL_XY


FRAME_DIR = "./viz_frames"


def make_figure():
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.3, 1.0], wspace=0.1)

    ax_bev = fig.add_subplot(gs[0, 0])
    ax_bev.set_xticks([]); ax_bev.set_yticks([])
    ax_bev.set_title("BEV — obstacles (red) + ego (green ▲)",
                     fontsize=13, fontweight='bold')

    ax_hud = fig.add_subplot(gs[0, 1])
    ax_hud.axis('off')
    return fig, ax_bev, ax_hud


def draw_bev(ax, obs: dict):
    """Composite BEV: grey = any occupancy, red = tall objects (likely
    vehicles / walls). Rotated so ego faces up."""
    bev = obs['bev']
    occ = bev[:, :, 4]       # binary occupancy
    hgt = bev[:, :, 0]       # max height

    occ_d = np.rot90(occ, k=1)
    hgt_d = np.rot90(hgt, k=1)

    h, w = occ_d.shape
    rgb = np.ones((h, w, 3), dtype=np.float32)     # white background
    grey_mask = occ_d > 0.5
    rgb[grey_mask] = [0.7, 0.7, 0.7]
    tall_mask = (occ_d > 0.5) & (hgt_d > 0.5)
    rgb[tall_mask] = [0.85, 0.15, 0.15]

    ax.clear()
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("BEV — obstacles (red) + ego (green ▲)",
                 fontsize=13, fontweight='bold')
    ax.imshow(rgb, origin='upper')

    cx, cy = w / 2.0, h / 2.0
    ax.add_patch(FancyArrow(cx, cy + 6, 0, -12,
                            width=4, color='limegreen', zorder=5,
                            length_includes_head=True, head_width=8,
                            head_length=6))


def draw_hud(ax, speed_kmh: float, action: np.ndarray,
             collision: bool, episode: int, step: int, reward: float,
             dist_to_goal: float, goal_reached: bool,
             lane_invasions: int):
    ax.clear()
    ax.axis('off')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    ax.text(0.5, 0.97, f"Episode {episode}   |   Step {step}",
            ha='center', va='top', fontsize=12, color='#444')

    # Distance to goal — informative line right under the header.
    ax.text(0.5, 0.92, f"goal @ ({GOAL_XY[0]:.1f}, {GOAL_XY[1]:.1f})   "
                       f"dist = {dist_to_goal:6.1f} m",
            ha='center', va='top', fontsize=11, color='#2a9df4',
            family='monospace')

    # SPEED — big digits
    speed_color = 'green' if speed_kmh < 130 else 'red'
    ax.text(0.5, 0.88, f"{speed_kmh:5.1f}", ha='center', va='top',
            fontsize=48, fontweight='bold', color=speed_color,
            family='monospace')
    ax.text(0.5, 0.73, "km/h", ha='center', va='top',
            fontsize=14, color='#666')

    # Speed bar (0 … 120 km/h)
    bar_y = 0.66
    ax.add_patch(Rectangle((0.1, bar_y), 0.8, 0.04,
                           facecolor='#eee', edgecolor='#999'))
    target_kmh = SPEED_TARGET * 3.6          # ≈ 120
    frac = min(1.0, max(0.0, speed_kmh / target_kmh))
    ax.add_patch(Rectangle((0.1, bar_y), 0.8 * frac, 0.04,
                           facecolor='#2a9df4', edgecolor='none'))
    ax.text(0.1, bar_y - 0.025, "0", fontsize=9, color='#666')
    ax.text(0.9, bar_y - 0.025, f"{int(target_kmh)}", fontsize=9,
            color='#666', ha='right')

    # Throttle / brake / steer
    a_steer = float(np.clip(action[0], -1.0, 1.0))
    a_thr = float(np.clip(action[1], -1.0, 1.0))
    throttle = max(0.0, a_thr)
    brake = max(0.0, -a_thr)

    _bar(ax, 0.50, "THROTTLE", throttle, color='#2ecc71')
    _bar(ax, 0.42, "BRAKE",    brake,    color='#e74c3c')
    _steer_bar(ax, 0.32, a_steer)

    ax.text(0.1, 0.22, f"reward = {reward:+7.3f}", fontsize=12,
            family='monospace', color='#333')
    ax.text(0.1, 0.18, f"lane invasions this step: {lane_invasions}",
            fontsize=10, family='monospace',
            color='#c0392b' if lane_invasions > 0 else '#666')

    if goal_reached:
        ax.add_patch(Rectangle((0.05, 0.05), 0.9, 0.1,
                               facecolor='#2ecc71', edgecolor='none'))
        ax.text(0.5, 0.10, "GOAL REACHED", ha='center', va='center',
                fontsize=20, fontweight='bold', color='white')
    elif collision:
        ax.add_patch(Rectangle((0.05, 0.05), 0.9, 0.1,
                               facecolor='#c0392b', edgecolor='none'))
        ax.text(0.5, 0.10, "COLLISION", ha='center', va='center',
                fontsize=20, fontweight='bold', color='white')
    else:
        ax.add_patch(Rectangle((0.05, 0.05), 0.9, 0.1,
                               facecolor='#27ae60', edgecolor='none'))
        ax.text(0.5, 0.10, "DRIVING", ha='center', va='center',
                fontsize=20, fontweight='bold', color='white')


def _bar(ax, y: float, label: str, value: float, color: str):
    ax.text(0.1, y + 0.015, label, fontsize=10, color='#333',
            family='monospace')
    ax.add_patch(Rectangle((0.35, y), 0.55, 0.03,
                           facecolor='#eee', edgecolor='#999'))
    ax.add_patch(Rectangle((0.35, y), 0.55 * value, 0.03,
                           facecolor=color, edgecolor='none'))
    ax.text(0.92, y + 0.015, f"{value:.2f}", fontsize=9,
            family='monospace', color='#333', va='center')


def _steer_bar(ax, y: float, value: float):
    ax.text(0.1, y + 0.015, "STEER", fontsize=10, color='#333',
            family='monospace')
    ax.add_patch(Rectangle((0.35, y), 0.55, 0.03,
                           facecolor='#eee', edgecolor='#999'))
    centre = 0.35 + 0.55 * 0.5
    ax.plot([centre, centre], [y, y + 0.03], color='#333', lw=1)
    if value >= 0:
        ax.add_patch(Rectangle((centre, y), 0.275 * value, 0.03,
                               facecolor='#f39c12', edgecolor='none'))
    else:
        ax.add_patch(Rectangle((centre + 0.275 * value, y),
                               -0.275 * value, 0.03,
                               facecolor='#f39c12', edgecolor='none'))
    ax.text(0.92, y + 0.015, f"{value:+.2f}", fontsize=9,
            family='monospace', color='#333', va='center')


def run(episodes: int, max_steps_per_ep: int, show_window: bool,
        save_mp4: bool, policy_path: str | None):
    os.makedirs(FRAME_DIR, exist_ok=True)
    if not show_window:
        matplotlib.use('Agg')

    rclpy.init()
    node = rclpy.create_node('viz_node')
    bev = BEVProcessor()
    env = CarlaRLEnvironment(node, bev)

    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    model = None
    if policy_path and os.path.exists(policy_path):
        from stable_baselines3 import PPO
        model = PPO.load(policy_path, env=env)
        print(f"[viz] using trained policy: {policy_path}")
    else:
        print("[viz] no policy — constant action (throttle=0.4, steer=0)")

    fig, ax_bev, ax_hud = make_figure()
    if show_window:
        plt.ion()
        plt.show(block=False)

    global_frame = 0
    try:
        for ep in range(episodes):
            obs, _ = env.reset()
            for step in range(max_steps_per_ep):
                if model is not None:
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    action = np.array([0.0, 0.4], dtype=np.float32)

                obs, reward, terminated, truncated, info = env.step(action)

                draw_bev(ax_bev, obs)
                draw_hud(ax_hud,
                         speed_kmh=info.get('speed_kmh', 0.0),
                         action=action,
                         collision=info.get('collision', False),
                         episode=ep, step=step, reward=reward,
                         dist_to_goal=info.get('dist_to_goal', 0.0),
                         goal_reached=info.get('goal_reached', False),
                         lane_invasions=info.get('lane_invasions', 0))

                fname = os.path.join(FRAME_DIR, f"frame_{global_frame:06d}.png")
                fig.savefig(fname, dpi=90, bbox_inches='tight')
                global_frame += 1

                if show_window:
                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()
                    plt.pause(0.001)

                if terminated or truncated:
                    if info.get('goal_reached'):
                        status = "GOAL"
                    elif info.get('collision'):
                        status = "COLLISION"
                    else:
                        status = "TIMEOUT"
                    print(f"[viz] ep {ep} ended after {step+1} steps — {status}")
                    break
    finally:
        node.destroy_node()
        rclpy.shutdown()
        plt.ioff()
        plt.close(fig)

    print(f"[viz] wrote {global_frame} frames → {FRAME_DIR}/")
    if save_mp4:
        out = "viz.mp4"
        cmd = (f"ffmpeg -y -framerate 10 -i {FRAME_DIR}/frame_%06d.png "
               f"-c:v libx264 -pix_fmt yuv420p {out}")
        print(f"[viz] stitching video: {cmd}")
        os.system(cmd)
        print(f"[viz] video → {out}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', type=int, default=3)
    p.add_argument('--max-steps', type=int, default=400)
    p.add_argument('--no-window', action='store_true')
    p.add_argument('--save-mp4', action='store_true')
    p.add_argument('--policy', type=str, default=None,
                   help="Path to trained PPO .zip; omit for constant-throttle baseline")
    args = p.parse_args()

    run(episodes=args.episodes,
        max_steps_per_ep=args.max_steps,
        show_window=not args.no_window,
        save_mp4=args.save_mp4,
        policy_path=args.policy)
