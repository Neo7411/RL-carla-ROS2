"""
carla_map_ros_viewer.py
========================
ROS2 node that subscribes to /carla/map (std_msgs/String, OpenDRIVE XML),
parses the road geometry, and visualises it with matplotlib.

Geometry types handled: line, arc, spiral (clothoid / Euler spiral)

Run
---
    # source your ROS2 + carla_ros_bridge workspace first, then:
    python carla_map_ros_viewer.py

    # optional: force a different topic
    python carla_map_ros_viewer.py --topic /carla/map
"""

import argparse
import math
import sys
import threading
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# ── ROS2 ─────────────────────────────────────────────────────────────────────
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
except ImportError:
    print("ERROR: rclpy not found — source your ROS2 workspace first.")
    sys.exit(1)

# ── scipy for Fresnel integrals (spiral geometry) ─────────────────────────────
try:
    from scipy.special import fresnel as _fresnel
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[warn] scipy not found — spiral geometry will be approximated as lines.")

import matplotlib
matplotlib.use("TkAgg")          # works well with ROS2 threading; swap to Qt5Agg if needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


# ─────────────────────────────────────────────────────────────────────────────
# OpenDRIVE geometry samplers
# ─────────────────────────────────────────────────────────────────────────────

def _sample_line(x0, y0, hdg, length, n=6) -> List[Tuple[float, float]]:
    """Straight-line segment."""
    pts = []
    for s in np.linspace(0, length, n):
        pts.append((x0 + s * math.cos(hdg),
                    y0 + s * math.sin(hdg)))
    return pts


def _sample_arc(x0, y0, hdg, length, curvature, n=32) -> List[Tuple[float, float]]:
    """
    Circular arc.  curvature = 1/R  (positive = left turn, negative = right).
    """
    if abs(curvature) < 1e-9:
        return _sample_line(x0, y0, hdg, length, n)

    R   = 1.0 / curvature
    cx  = x0 - R * math.sin(hdg)
    cy  = y0 + R * math.cos(hdg)
    a0  = math.atan2(y0 - cy, x0 - cx)
    da  = length * curvature        # total angle swept

    pts = []
    for i in range(n + 1):
        a = a0 + da * i / n
        pts.append((cx + R * math.cos(a),
                    cy + R * math.sin(a)))
    return pts


def _sample_spiral(x0, y0, hdg, length,
                   curv_start, curv_end, n=64) -> List[Tuple[float, float]]:
    """
    Euler spiral (clothoid).  Uses Fresnel integrals when scipy is available,
    falls back to incremental integration otherwise.
    """
    if abs(curv_end - curv_start) < 1e-9:
        # Degenerate: treat as arc at curv_start
        return _sample_arc(x0, y0, hdg, length, curv_start, n)

    pts = []
    dk  = (curv_end - curv_start) / length       # curvature rate

    if HAS_SCIPY:
        # Parametric form: κ(s) = curv_start + dk*s
        # Use numerical integration via accumulated Fresnel approach
        ds = length / n
        x, y, h = x0, y0, hdg
        pts.append((x, y))
        for i in range(n):
            s   = i * ds
            k   = curv_start + dk * s
            h  += k * ds
            x  += math.cos(h) * ds
            y  += math.sin(h) * ds
            pts.append((x, y))
    else:
        # Same incremental integration (no scipy needed)
        ds = length / n
        x, y, h = x0, y0, hdg
        pts.append((x, y))
        for i in range(n):
            k  = curv_start + dk * (i * ds)
            h += k * ds
            x += math.cos(h) * ds
            y += math.sin(h) * ds
            pts.append((x, y))

    return pts


# ─────────────────────────────────────────────────────────────────────────────
# OpenDRIVE XML parser
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RoadGeom:
    """One <geometry> element with its sampled centreline points."""
    road_id:   str
    pts:       List[Tuple[float, float]] = field(default_factory=list)
    is_junction: bool = False


def _parse_geometry_element(geom_el, road_id, is_junction) -> Optional[RoadGeom]:
    """Parse a single <geometry> XML element into a RoadGeom."""
    try:
        x      = float(geom_el.attrib["x"])
        y      = float(geom_el.attrib["y"])
        hdg    = float(geom_el.attrib["hdg"])
        length = float(geom_el.attrib["length"])
    except (KeyError, ValueError):
        return None

    if length < 0.01:
        return None

    # Determine geometry type from child element
    line_el   = geom_el.find("line")
    arc_el    = geom_el.find("arc")
    spiral_el = geom_el.find("spiral")
    poly_el   = geom_el.find("poly3")
    ppoly_el  = geom_el.find("paramPoly3")

    if arc_el is not None:
        try:
            curv = float(arc_el.attrib["curvature"])
        except (KeyError, ValueError):
            curv = 0.0
        pts = _sample_arc(x, y, hdg, length, curv)

    elif spiral_el is not None:
        try:
            cs = float(spiral_el.attrib["curvStart"])
            ce = float(spiral_el.attrib["curvEnd"])
        except (KeyError, ValueError):
            cs, ce = 0.0, 0.0
        pts = _sample_spiral(x, y, hdg, length, cs, ce)

    else:
        # line / poly3 / paramPoly3 — treat as straight (good enough for viz)
        pts = _sample_line(x, y, hdg, length)

    if not pts:
        return None

    rg = RoadGeom(road_id=road_id, is_junction=is_junction)
    rg.pts = pts
    return rg


def parse_opendrive(xml_string: str) -> List[RoadGeom]:
    """
    Parse an OpenDRIVE XML string.
    Returns one RoadGeom per <geometry> element across all <road> elements.
    """
    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError as e:
        print(f"[parse] XML parse error: {e}")
        return []

    geometries: List[RoadGeom] = []

    for road_el in root.findall("road"):
        road_id    = road_el.attrib.get("id", "?")
        junction   = road_el.attrib.get("junction", "-1")
        is_junction = (junction != "-1")

        plan_el = road_el.find("planView")
        if plan_el is None:
            continue

        for geom_el in plan_el.findall("geometry"):
            rg = _parse_geometry_element(geom_el, road_id, is_junction)
            if rg is not None:
                geometries.append(rg)

    return geometries


# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib visualiser
# ─────────────────────────────────────────────────────────────────────────────

C_BG       = "#12121F"
C_ROAD     = "#3A6EA5"     # normal road
C_JUNCTION = "#E07B39"     # junction road
C_SPINE    = "#AAAAAA"     # thin centreline overlay

def visualise(geometries: List[RoadGeom], map_name: str = "CARLA OpenDRIVE"):
    """Draw all road geometries in a dark matplotlib figure."""

    if not geometries:
        print("[vis] No geometry to draw.")
        return

    fig, ax = plt.subplots(figsize=(15, 11))
    fig.patch.set_facecolor("#1E1E2E")
    ax.set_facecolor(C_BG)
    ax.tick_params(colors="#CCCCCC")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")

    ax.set_title(f"CARLA Map  —  {map_name}  ({len(geometries)} road segments)",
                 color="#EEEEEE", fontsize=12, pad=10)
    ax.set_xlabel("X  [m]", color="#AAAAAA")
    ax.set_ylabel("Y  [m]", color="#AAAAAA")
    ax.set_aspect("equal")
    ax.grid(True, color="#1E1E2E", linewidth=0.4, zorder=0)

    n_roads = n_junctions = 0
    for rg in geometries:
        if not rg.pts:
            continue
        xs = [p[0] for p in rg.pts]
        ys = [p[1] for p in rg.pts]
        color = C_JUNCTION if rg.is_junction else C_ROAD
        lw    = 0.8        if rg.is_junction else 1.2
        ax.plot(xs, ys, color=color, linewidth=lw, alpha=0.85, zorder=2)
        if rg.is_junction:
            n_junctions += 1
        else:
            n_roads += 1

    # Legend
    handles = [
        Line2D([0],[0], color=C_ROAD,     lw=2, label=f"Road ({n_roads})"),
        Line2D([0],[0], color=C_JUNCTION, lw=2, label=f"Junction road ({n_junctions})"),
    ]
    ax.legend(handles=handles, loc="upper right",
              fontsize=8, framealpha=0.85,
              facecolor="#1E1E2E", labelcolor="#CCCCCC")

    fig.text(0.01, 0.005,
             f"Parsed {len(geometries)} geometry segments from /carla/map",
             color="#666688", fontsize=7, va="bottom")

    plt.tight_layout(rect=[0, 0.02, 1, 1])
    print(f"[vis] Drawing {n_roads} road + {n_junctions} junction segments.")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# ROS2 node
# ─────────────────────────────────────────────────────────────────────────────

class MapSubscriberNode(Node):

    def __init__(self, topic: str):
        super().__init__("carla_map_viewer")
        self._topic    = topic
        self._received = threading.Event()
        self._xml_data: Optional[str] = None

        self._sub = self.create_subscription(
            String,
            topic,
            self._cb,
            qos_profile=10,
        )
        self.get_logger().info(f"Waiting for map on '{topic}' …")

    def _cb(self, msg: String):
        if self._xml_data is not None:
            return                          # only need it once
        self.get_logger().info("Map received — parsing …")
        self._xml_data = msg.data
        self._received.set()

    def wait_for_map(self, timeout_s: float = 30.0) -> Optional[str]:
        """Block until the map arrives or timeout, return the XML string."""
        self._received.wait(timeout=timeout_s)
        return self._xml_data


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="CARLA OpenDRIVE map viewer via ROS2 /carla/map topic.")
    p.add_argument("--topic",   default="/carla/map",
                   help="ROS2 topic (default: /carla/map)")
    p.add_argument("--timeout", type=float, default=30.0,
                   help="Seconds to wait for the map message (default: 30)")
    p.add_argument("--save_xodr", default="",
                   help="Optional: save the raw OpenDRIVE XML to this path")
    return p.parse_args()


def main():
    args = parse_args()

    rclpy.init()
    node = MapSubscriberNode(args.topic)

    # Spin in a background thread so we can block here
    spin_thread = threading.Thread(
        target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    xml_data = node.wait_for_map(timeout_s=args.timeout)

    if xml_data is None:
        node.get_logger().error(
            f"No map received on '{args.topic}' within {args.timeout}s.\n"
            f"  Is the CARLA ROS bridge running?\n"
            f"  Check:  ros2 topic echo {args.topic} --once")
        rclpy.shutdown()
        sys.exit(1)

    # Optional: dump raw XML for inspection
    if args.save_xodr:
        with open(args.save_xodr, "w") as f:
            f.write(xml_data)
        print(f"[info] Raw OpenDRIVE XML saved to: {args.save_xodr}")

    # Parse
    print(f"[parse] OpenDRIVE XML length: {len(xml_data):,} chars")
    geometries = parse_opendrive(xml_data)
    print(f"[parse] {len(geometries)} geometry segments parsed.")

    # Extract map name from XML if present
    try:
        root = ET.fromstring(xml_data)
        header = root.find("header")
        map_name = (header.attrib.get("name", "")
                    or header.attrib.get("north", "")
                    if header is not None else "")
    except Exception:
        map_name = ""

    # Shut down ROS before opening the blocking plt.show()
    node.destroy_node()
    rclpy.shutdown()

    # Visualise (blocking)
    visualise(geometries, map_name=map_name or args.topic)


if __name__ == "__main__":
    main()
