"""
MapRouter — Lanelet2-based routing + progress tracking for CARLA Town04.

Loads a .osm Lanelet2 map, builds a vehicle routing graph, samples random
reachable (start, goal) pairs, densifies routes into waypoint polylines, and
reports progress metrics (cross-track, heading error, distance-to-goal).

Coordinate convention
---------------------
Lanelet2/OSM stores ENU-like right-handed coords; CARLA uses left-handed
(y-flipped). We apply `y_carla = -y_lanelet` on ingest so every x/y outside
this file is already in CARLA frame. Flip FLIP_Y if an overlay test disagrees.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

import lanelet2
from lanelet2.io import Origin, load
from lanelet2.projection import UtmProjector
from lanelet2.traffic_rules import Locations, Participants, create as create_rules

FLIP_Y = True  # set False if overlay shows mirrored routes


@dataclass
class Progress:
    dist_to_goal: float       # meters, along remaining route
    cross_track: float        # signed meters (left +, right -)
    heading_err: float        # radians in [-pi, pi]
    next_wp_xy: np.ndarray    # (2,) next waypoint in world frame
    fraction_done: float      # 0..1
    goal_reached: bool
    off_route: bool


class MapRouter:
    def __init__(self, osm_path: str, waypoint_spacing: float = 2.0,
                 off_route_thresh: float = 25.0, goal_radius: float = 5.0):
        self._projector = UtmProjector(Origin(0.0, 0.0))
        self._map = load(osm_path, self._projector)
        rules = create_rules(Locations.Germany, Participants.Vehicle)
        self._graph = lanelet2.routing.RoutingGraph(self._map, rules)
        self._lanelets = list(self._map.laneletLayer)

        self.spacing = waypoint_spacing
        self.off_route_thresh = off_route_thresh
        self.goal_radius = goal_radius

        # Active route state
        self._waypoints: np.ndarray | None = None   # (N, 2)
        self._cum_dist: np.ndarray | None = None    # (N,) cumulative arc length
        self._total_len: float = 0.0
        self._wp_idx: int = 0

    # ── public API ──────────────────────────────────────────────────────

    @property
    def waypoints(self) -> np.ndarray | None:
        return self._waypoints

    @property
    def total_length(self) -> float:
        return self._total_len

    def random_spawn_pose(self, rng: random.Random | None = None
                          ) -> tuple[np.ndarray, float]:
        """Pick a random lanelet centerline point and its heading (radians).

        Returns (xy[2], yaw_rad) in CARLA world frame.
        """
        rng = rng or random
        ll = rng.choice(self._lanelets)
        cl = list(ll.centerline)
        if len(cl) < 2:
            return self.random_spawn_pose(rng)
        i = rng.randint(0, len(cl) - 2)
        p0 = self._xy(cl[i])
        p1 = self._xy(cl[i + 1])
        yaw = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        return p0, yaw

    def snap_to_lane(self, xy: np.ndarray) -> tuple[np.ndarray, float]:
        """Return (nearest_centerline_xy, heading_rad) for the given world xy."""
        ll = self._nearest_lanelet(xy)
        if ll is None:
            return xy.astype(np.float32), 0.0
        cl = list(ll.centerline)
        best_i, best_d = 0, float('inf')
        for i, p in enumerate(cl):
            q = self._xy(p)
            d = (q[0] - xy[0]) ** 2 + (q[1] - xy[1]) ** 2
            if d < best_d:
                best_d, best_i = d, i
        j = min(best_i + 1, len(cl) - 1)
        if j == best_i:
            j, best_i = best_i, max(best_i - 1, 0)
        p0 = self._xy(cl[best_i]); p1 = self._xy(cl[j])
        yaw = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        return p0, yaw

    def heading_at(self, xy: np.ndarray) -> float:
        """Heading (radians) of the nearest lanelet centerline segment."""
        ll = self._nearest_lanelet(xy)
        if ll is None:
            return 0.0
        cl = list(ll.centerline)
        if len(cl) < 2:
            return 0.0
        best_i, best_d = 0, float('inf')
        for i, p in enumerate(cl):
            q = self._xy(p)
            d = (q[0] - xy[0]) ** 2 + (q[1] - xy[1]) ** 2
            if d < best_d:
                best_d, best_i = d, i
        j = min(best_i + 1, len(cl) - 1)
        if j == best_i:
            j, best_i = best_i, max(best_i - 1, 0)
        p0 = self._xy(cl[best_i]); p1 = self._xy(cl[j])
        return math.atan2(p1[1] - p0[1], p1[0] - p0[0])

    def plan_random_route(self, start_xy: np.ndarray,
                          min_len: float = 50.0, max_len: float = 250.0,
                          max_tries: int = 30,
                          rng: random.Random | None = None) -> bool:
        """Pick a reachable random goal and build the waypoint list.

        Returns True on success, False if no route found after max_tries.
        """
        rng = rng or random
        start_ll = self._nearest_lanelet(start_xy)
        if start_ll is None:
            return False

        fallback: np.ndarray | None = None
        for _ in range(max_tries):
            goal_ll = rng.choice(self._lanelets)
            if goal_ll.id == start_ll.id:
                continue
            route = self._graph.getRoute(start_ll, goal_ll)
            if route is None:
                continue
            path = route.shortestPath()
            if path is None or len(path) < 2:
                continue
            wps = self._densify(path)
            total = self._arc_lengths(wps)[-1]
            if min_len <= total <= max_len:
                self._set_route(wps, start_xy)
                return True
            if fallback is None and len(wps) >= 2:
                fallback = wps

        if fallback is not None:
            self._set_route(fallback, start_xy)
            return True

        # Last resort: synthetic straight-line route so step() never crashes.
        yaw = self.heading_at(start_xy)
        span = np.linspace(0.0, 60.0, 31, dtype=np.float32)
        synth = np.stack([
            start_xy[0] + span * math.cos(yaw),
            start_xy[1] + span * math.sin(yaw),
        ], axis=1)
        self._set_route(synth, start_xy)
        return False

    def progress(self, ego_xy: np.ndarray, ego_yaw: float) -> Progress:
        """Query route progress. Advances the internal waypoint cursor."""
        assert self._waypoints is not None, "call plan_random_route() first"

        # Advance cursor past any waypoints behind us (within a window)
        search_to = min(self._wp_idx + 50, len(self._waypoints) - 1)
        best_i = self._wp_idx
        best_d = float('inf')
        for i in range(self._wp_idx, search_to + 1):
            d = np.linalg.norm(self._waypoints[i] - ego_xy)
            if d < best_d:
                best_d = d
                best_i = i
        self._wp_idx = best_i

        # Next waypoint (look-ahead 1 segment)
        nxt_i = min(self._wp_idx + 1, len(self._waypoints) - 1)
        nxt = self._waypoints[nxt_i]

        # Cross-track and heading error against the local segment
        seg_a = self._waypoints[self._wp_idx]
        seg_b = self._waypoints[nxt_i]
        seg_dir = seg_b - seg_a
        seg_len = np.linalg.norm(seg_dir) + 1e-6
        seg_hat = seg_dir / seg_len
        to_ego = ego_xy - seg_a
        cross = seg_hat[0] * to_ego[1] - seg_hat[1] * to_ego[0]  # z of cross
        seg_yaw = math.atan2(seg_dir[1], seg_dir[0])
        heading_err = _wrap(seg_yaw - ego_yaw)

        # Remaining arc length along the route
        remaining = (self._total_len - self._cum_dist[self._wp_idx]
                     + float(np.linalg.norm(ego_xy - seg_a)))
        remaining = max(0.0, remaining)
        frac = 1.0 - remaining / max(self._total_len, 1e-6)

        goal_xy = self._waypoints[-1]
        goal_dist = float(np.linalg.norm(ego_xy - goal_xy))
        goal_reached = goal_dist < self.goal_radius
        off_route = abs(cross) > self.off_route_thresh

        return Progress(
            dist_to_goal=remaining,
            cross_track=float(cross),
            heading_err=float(heading_err),
            next_wp_xy=nxt.copy(),
            fraction_done=float(np.clip(frac, 0.0, 1.0)),
            goal_reached=goal_reached,
            off_route=off_route,
        )

    # ── internals ───────────────────────────────────────────────────────

    def _xy(self, point) -> np.ndarray:
        y = -point.y if FLIP_Y else point.y
        return np.array([point.x, y], dtype=np.float32)

    def _nearest_lanelet(self, xy: np.ndarray):
        best = None
        best_d = float('inf')
        for ll in self._lanelets:
            for p in ll.centerline:
                q = self._xy(p)
                d = (q[0] - xy[0]) ** 2 + (q[1] - xy[1]) ** 2
                if d < best_d:
                    best_d = d
                    best = ll
        return best

    def _densify(self, path) -> np.ndarray:
        """Turn a lanelet path into a ~uniform polyline at `spacing` meters."""
        raw: list[np.ndarray] = []
        for ll in path:
            for p in ll.centerline:
                raw.append(self._xy(p))
        if not raw:
            return np.zeros((0, 2), dtype=np.float32)

        pts = np.array(raw, dtype=np.float32)
        out = [pts[0]]
        acc = 0.0
        for i in range(1, len(pts)):
            seg = pts[i] - pts[i - 1]
            d = float(np.linalg.norm(seg))
            if d < 1e-3:
                continue
            acc += d
            while acc >= self.spacing:
                acc -= self.spacing
                t = 1.0 - acc / d
                out.append(pts[i - 1] + t * seg)
        if not np.allclose(out[-1], pts[-1]):
            out.append(pts[-1])
        return np.array(out, dtype=np.float32)

    def _arc_lengths(self, pts: np.ndarray) -> np.ndarray:
        if len(pts) < 2:
            return np.zeros(len(pts), dtype=np.float32)
        d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        return np.concatenate([[0.0], np.cumsum(d)]).astype(np.float32)

    def _set_route(self, waypoints: np.ndarray,
                   start_xy: np.ndarray | None = None):
        self._waypoints = waypoints
        self._cum_dist = self._arc_lengths(waypoints)
        self._total_len = float(self._cum_dist[-1])
        if start_xy is not None and len(waypoints) > 0:
            d = np.linalg.norm(waypoints - start_xy, axis=1)
            self._wp_idx = int(np.argmin(d))
        else:
            self._wp_idx = 0


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


if __name__ == '__main__':
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    osm = os.path.join(here, '..', 'maps', 'Town04.osm')
    r = MapRouter(osm)
    print(f"loaded {len(r._lanelets)} lanelets")

    rng = random.Random(0)
    for trial in range(5):
        start, yaw = r.random_spawn_pose(rng)
        ok = r.plan_random_route(start, rng=rng)
        if ok:
            print(f"trial {trial}: start={start.round(1)} len={r.total_length:.1f}m "
                  f"waypoints={len(r.waypoints)}")
            prog = r.progress(start, yaw)
            print(f"   first progress: cross={prog.cross_track:.2f} "
                  f"heading_err={math.degrees(prog.heading_err):.1f}° "
                  f"to_goal={prog.dist_to_goal:.1f}m")
        else:
            print(f"trial {trial}: no route")
