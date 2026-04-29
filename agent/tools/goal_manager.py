"""Goal pose + per-step goal-relative features.

Coordinate convention:
    The CARLA editor reports spawn-point Y in a left-handed frame
    (Y-down). The bridge's /carla/ego_vehicle/odometry topic publishes
    in a right-handed frame (Y-up). Pasting an editor coordinate as-is
    would put the goal on the wrong side of the map.

    GoalManager.from_carla_editor() flips Y so you can paste editor
    coordinates verbatim. Use the explicit __init__ if you already have
    world-frame (odometry-frame) coordinates.
"""

import math


class GoalManager:
    def __init__(self, x, y, radius_m=5.0):
        self.x = float(x)
        self.y = float(y)
        self.radius = float(radius_m)
        self._prev_dist = None

    @classmethod
    def from_carla_editor(cls, x, y_editor, radius_m=5.0):
        """Build a goal from CARLA-editor (left-handed) coordinates."""
        return cls(x, -y_editor, radius_m=radius_m)

    def reset(self, ego_x, ego_y):
        """Call at the start of every episode to seed progress tracking."""
        self._prev_dist = self._dist(ego_x, ego_y)

    def _dist(self, x, y):
        return math.hypot(self.x - x, self.y - y)

    def features(self, ego_x, ego_y, ego_yaw):
        """Return (dist_m, bearing_error_rad, progress_delta_m).

        bearing_error is the heading the vehicle should turn through to
        face the goal directly: 0 means already aimed at it, +pi/2 means
        the goal is 90° to the left, etc. progress_delta is positive when
        we got closer this step.
        """
        dist = self._dist(ego_x, ego_y)
        bearing_to_goal = math.atan2(self.y - ego_y, self.x - ego_x)
        bearing_err = _wrap_pi(bearing_to_goal - ego_yaw)
        if self._prev_dist is None:
            delta = 0.0
        else:
            delta = self._prev_dist - dist
        self._prev_dist = dist
        return dist, bearing_err, delta

    def reached(self, ego_x, ego_y):
        return self._dist(ego_x, ego_y) <= self.radius


def _wrap_pi(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a
