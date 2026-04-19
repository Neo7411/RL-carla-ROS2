#!/usr/bin/env python3
"""
CARLA Spawn Point Visualizer
Draws all spawn points in the CARLA server view with index, X and Y labels.

Usage:
    python3 visualize_spawn_points.py [--host HOST] [--port PORT] [--life-time SECONDS]

Requirements:
    pip install carla
"""

import argparse
import carla
import time
import sys


# ── Appearance settings ──────────────────────────────────────────────────────
POINT_COLOR   = carla.Color(r=255, g=50,  b=50,  a=255)   # red dots
TEXT_COLOR    = carla.Color(r=255, g=255, b=50,  a=255)    # yellow labels
ARROW_COLOR   = carla.Color(r=50,  g=200, b=255, a=255)    # cyan arrows (yaw direction)
POINT_SIZE    = 0.15   # metres (radius of the debug sphere)
TEXT_OFFSET_Z = 0.8    # metres above spawn point for the label
ARROW_LENGTH  = 1.5    # metres (forward arrow showing yaw)
LIFE_TIME     = 60.0   # seconds to keep markers alive (0 = forever until server restart)
# ─────────────────────────────────────────────────────────────────────────────


def draw_spawn_points(world: carla.World, life_time: float) -> int:
    """Draw all spawn points and return how many were drawn."""
    spawn_points = world.get_map().get_spawn_points()
    debug = world.debug

    print(f"\n  Found {len(spawn_points)} spawn points on map '{world.get_map().name}'")
    print("  Drawing … (this may take a moment for large maps)\n")

    for idx, transform in enumerate(spawn_points):
        loc = transform.location
        rot = transform.rotation

        # ── Sphere at the spawn location ──────────────────────────────────
        debug.draw_point(
            loc + carla.Location(z=0.1),
            size=POINT_SIZE,
            color=POINT_COLOR,
            life_time=life_time,
        )

        # ── Forward arrow (shows yaw direction) ───────────────────────────
        debug.draw_arrow(
            begin=loc + carla.Location(z=0.3),
            end=loc + carla.Location(z=0.3)
                + carla.Location(
                    x=ARROW_LENGTH * __import__("math").cos(__import__("math").radians(rot.yaw)),
                    y=ARROW_LENGTH * __import__("math").sin(__import__("math").radians(rot.yaw)),
                ),
            thickness=0.05,
            arrow_size=0.15,
            color=ARROW_COLOR,
            life_time=life_time,
        )

        # ── Text label: index + X / Y ─────────────────────────────────────
        label = f"#{idx}  x={loc.x:.1f}  y={loc.y:.1f}"
        debug.draw_string(
            loc + carla.Location(z=TEXT_OFFSET_Z),
            label,
            draw_shadow=True,
            color=TEXT_COLOR,
            life_time=life_time,
        )

        print(f"  [{idx:>4}]  x={loc.x:>9.2f}  y={loc.y:>9.2f}  z={loc.z:>7.2f}  yaw={rot.yaw:>7.2f}°")

    return len(spawn_points)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize CARLA spawn points in the server window."
    )
    parser.add_argument("--host",      default="127.0.0.1",  help="CARLA server host (default: 127.0.0.1)")
    parser.add_argument("--port",      default=2000, type=int, help="CARLA server port (default: 2000)")
    parser.add_argument("--life-time", default=LIFE_TIME, type=float,
                        help=f"How many seconds to keep markers visible (default: {LIFE_TIME}, 0=forever)")
    parser.add_argument("--timeout",   default=10.0, type=float, help="Connection timeout in seconds")
    args = parser.parse_args()

    # ── Connect ───────────────────────────────────────────────────────────────
    print(f"\n  Connecting to CARLA at {args.host}:{args.port} …")
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(args.timeout)
        world  = client.load_world("Town04")
        server_ver = client.get_server_version()
        client_ver = client.get_client_version()
        print(f"  Connected  ✓  (server {server_ver} / client {client_ver})")
    except RuntimeError as exc:
        print(f"\n  ERROR: Could not connect — {exc}")
        print("  Make sure CarlaUE4 is running and the port is correct.\n")
        sys.exit(1)

    # ── Draw ──────────────────────────────────────────────────────────────────
    count = draw_spawn_points(world, life_time=args.life_time)

    life_msg = (
        "markers will persist until the server is restarted"
        if args.life_time == 0
        else f"markers will disappear after {args.life_time:.0f} s"
    )
    print(f"\n  Done — {count} spawn points drawn in the server view.")
    print(f"  ({life_msg})\n")

    # Keep process alive so markers stay visible if life_time is short
    if 0 < args.life_time < 300:
        print("  Keeping process alive … press Ctrl+C to exit early.\n")
        try:
            time.sleep(args.life_time)
        except KeyboardInterrupt:
            print("  Interrupted by user.")


if __name__ == "__main__":
    main()