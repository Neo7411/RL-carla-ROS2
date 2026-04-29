"""Lanelet2 wrapper for routing and lane-aware features.

The Town04.osm file ships with `local_x`/`local_y` tags that match CARLA
world coordinates exactly when loaded via UtmProjector(Origin(0,0)).
Verified empirically on 2026-04-28: GPS (lat=0.00282565, lon=0.00222746)
projects to (x=248.204, y=312.750), which matches the file's local_x/y tags.

Coordinate frame: this module operates in the world frame used by
/carla/ego_vehicle/odometry (right-handed, y-up). It is NOT the CARLA
editor frame (left-handed, y-down). The GoalManager handles that flip.
"""

import math
import numpy as np

import lanelet2
import lanelet2.geometry
from lanelet2.core import BasicPoint2d
from lanelet2.io import Origin
from lanelet2.projection import UtmProjector


_CANDIDATE_RADIUS_M = 15.0
_MAX_CANDIDATES = 12
_QUERY_WINDOW = 40  # ±vertices searched around the last nearest index


class Lanelet2Router:
    def __init__(self, osm_path):
        proj = UtmProjector(Origin(0.0, 0.0))
        self.map = lanelet2.io.load(osm_path, proj)
        rules = lanelet2.traffic_rules.create(
            lanelet2.traffic_rules.Locations.Germany,
            lanelet2.traffic_rules.Participants.Vehicle,
        )
        self.graph = lanelet2.routing.RoutingGraph(self.map, rules)

        # Cached centerline polyline for the current route — recomputed on
        # plan(). We flatten it once and operate on numpy arrays per step.
        self._route_xy = None      # (N, 2)
        self._route_seg_len = None  # (N-1,) cumulative lengths
        self._route_total_len = 0.0

        # Locality hint for query(): the nearest vertex moves at most a
        # few indices per step. Search ±_QUERY_WINDOW around the last
        # hit instead of scanning all N vertices. Reset on plan().
        self._last_idx = 0

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------
    def _candidates(self, x, y):
        """Lanelets within _CANDIDATE_RADIUS_M of (x,y), nearest first."""
        pt = BasicPoint2d(float(x), float(y))
        nearby = lanelet2.geometry.findWithin2d(
            self.map.laneletLayer, pt, _CANDIDATE_RADIUS_M)
        return [ll for _, ll in nearby[:_MAX_CANDIDATES]]

    def plan(self, start_xy, goal_xy):
        """Plan a route and cache its centerline for fast per-step queries.

        The closest lanelet to a query point is sometimes unroutable (e.g.
        an opposing-direction lanelet at an intersection). We try every
        (start_candidate, goal_candidate) pair within _CANDIDATE_RADIUS_M
        of each point, sorted by combined distance, and take the first
        pair that yields a route.

        Returns True if a route was found, False otherwise.
        """
        starts = self._candidates(*start_xy)
        goals = self._candidates(*goal_xy)
        if not starts or not goals:
            self._route_xy = None
            return False

        # Iterate nearest-first on both sides — closest start is the most
        # likely match for the actual ego lanelet.
        route = None
        for sll in starts:
            for gll in goals:
                r = self.graph.getRoute(sll, gll)
                if r is not None:
                    route = r
                    break
            if route is not None:
                break

        if route is None:
            self._route_xy = None
            return False

        path = route.shortestPath()
        pts = []
        for ll in path:
            for p in ll.centerline:
                if not pts or (pts[-1][0] != p.x or pts[-1][1] != p.y):
                    pts.append((p.x, p.y))
        if len(pts) < 2:
            self._route_xy = None
            return False

        self._route_xy = np.asarray(pts, dtype=np.float32)
        diffs = np.diff(self._route_xy, axis=0)
        seg = np.linalg.norm(diffs, axis=1)
        self._route_seg_len = np.concatenate(([0.0], np.cumsum(seg)))
        self._route_total_len = float(self._route_seg_len[-1])
        self._last_idx = 0
        return True

    def has_route(self):
        return self._route_xy is not None

    # ------------------------------------------------------------------
    # Per-step queries
    # ------------------------------------------------------------------
    def query(self, x, y):
        """Return lane-frame features at (x, y) relative to the cached route.

        Outputs:
          lateral_offset (m, signed; left of route = positive)
          route_heading  (rad, atan2 of nearest segment tangent)
          progress       (fraction in [0, 1] along the cached route)
        """
        if self._route_xy is None:
            return 0.0, 0.0, 0.0

        p = np.array([x, y], dtype=np.float32)
        N = len(self._route_xy)

        # Locality search: scan a small window around the previous hit.
        # The agent moves <1 m/step at 60 km/h × 0.05 s, so the nearest
        # vertex shifts by 1–2 indices most steps. If the window's local
        # minimum is at its boundary we fall back to a full scan (handles
        # teleports / large jumps).
        lo = max(0, self._last_idx - _QUERY_WINDOW)
        hi = min(N, self._last_idx + _QUERY_WINDOW + 1)
        d_local = np.linalg.norm(self._route_xy[lo:hi] - p, axis=1)
        i_local = int(np.argmin(d_local))
        i = lo + i_local
        if i_local == 0 and lo > 0 or i_local == hi - lo - 1 and hi < N:
            # Hit the window edge — could be a teleport or after reset.
            d = np.linalg.norm(self._route_xy - p, axis=1)
            i = int(np.argmin(d))
        else:
            # Reuse the local distances, padding the unsearched range
            # with infinity so segment-selection below still works.
            d = np.full(N, np.inf, dtype=np.float32)
            d[lo:hi] = d_local
        self._last_idx = i

        # Choose the segment touching the nearest vertex that's closest to p.
        if i == 0:
            j = 0
        elif i == len(self._route_xy) - 1:
            j = i - 1
        else:
            j = i if d[i + 1] < d[i - 1] else i - 1

        a = self._route_xy[j]
        b = self._route_xy[j + 1]
        ab = b - a
        ab_len = float(np.linalg.norm(ab)) + 1e-9
        t = float(np.clip(np.dot(p - a, ab) / (ab_len * ab_len), 0.0, 1.0))
        foot = a + t * ab

        # Signed lateral offset: cross product sign of (ab × ap).
        cross = ab[0] * (p[1] - a[1]) - ab[1] * (p[0] - a[0])
        lateral = float(np.linalg.norm(p - foot)) * (1.0 if cross >= 0 else -1.0)

        heading = math.atan2(float(ab[1]), float(ab[0]))

        progress_m = float(self._route_seg_len[j] + t * ab_len)
        progress = progress_m / max(self._route_total_len, 1e-3)
        return lateral, heading, float(np.clip(progress, 0.0, 1.0))
