import numpy as np


class BEVProcessor:

    def __init__(
        self,
        x_range: tuple = (-128.0, 128.0),
        y_range: tuple = (-128.0, 128.0),
        z_range: tuple = (-2.0, 4.0),
        bev_height: int = 256,
        bev_width: int = 256,
    ):
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.z_min, self.z_max = z_range
        self.h = bev_height
        self.w = bev_width
        self.num_channels = 6  # +1 for route mask

    # ── Public Interface ─────────────────────────────────────

    @property
    def observation_shape(self) -> tuple:
        """Shape that TF-Agents needs for observation_spec."""
        return (self.h, self.w, self.num_channels)

    def process(self, points: np.ndarray,
                route_local: np.ndarray | None = None) -> np.ndarray:
        """
        Full pipeline: raw points (+ optional route) → BEV tensor.

        Args:
            points: [N, 4] float32 array (x, y, z, intensity) in LiDAR frame.
            route_local: optional [M, 2] route polyline in LiDAR-local frame
                (ego-forward = +x). Rasterized into channel 5.

        Returns:
            [H, W, 6] float32 BEV tensor.
        """
        cropped = self._crop(points)

        if len(cropped) == 0:
            bev = np.zeros(self.observation_shape, dtype=np.float32)
        else:
            bev = self._project(cropped)

        if route_local is not None and len(route_local) >= 2:
            self._rasterize_route(bev, route_local)
        return bev

    def _rasterize_route(self, bev: np.ndarray, route_local: np.ndarray) -> None:
        """Draw the route polyline into channel 5 using Bresenham-style
        interpolation. Route is in local meters; we reuse _compute_indices."""
        # Keep points within BEV bounds; clip is fine here since we want the
        # path to reach the edge even when the goal is outside view.
        xs = route_local[:, 0]
        ys = route_local[:, 1]
        xi = ((xs - self.x_min) / (self.x_max - self.x_min) * self.w).astype(np.int32)
        yi = ((ys - self.y_min) / (self.y_max - self.y_min) * self.h).astype(np.int32)
        xi = np.clip(xi, 0, self.w - 1)
        yi = np.clip(yi, 0, self.h - 1)

        mask = bev[:, :, 5]
        for k in range(len(xi) - 1):
            x0, y0, x1, y1 = int(xi[k]), int(yi[k]), int(xi[k + 1]), int(yi[k + 1])
            n = max(abs(x1 - x0), abs(y1 - y0)) + 1
            if n == 1:
                mask[y0, x0] = 1.0
                continue
            xs_l = np.linspace(x0, x1, n).astype(np.int32)
            ys_l = np.linspace(y0, y1, n).astype(np.int32)
            mask[ys_l, xs_l] = 1.0

    # ── Private Methods ──────────────────────────────────────

    def _crop(self, points: np.ndarray) -> np.ndarray:
        """Keep only points inside the 3D region of interest."""
        mask = (
            (points[:, 0] >= self.x_min) & (points[:, 0] <= self.x_max) &
            (points[:, 1] >= self.y_min) & (points[:, 1] <= self.y_max) &
            (points[:, 2] >= self.z_min) & (points[:, 2] <= self.z_max)
        )
        return points[mask]

    def _compute_indices(self, points: np.ndarray):
        """Map continuous (x, y) → discrete grid (col, row)."""
        xi = ((points[:, 0] - self.x_min) / (self.x_max - self.x_min) * self.w).astype(np.int32)
        yi = ((points[:, 1] - self.y_min) / (self.y_max - self.y_min) * self.h).astype(np.int32)
        xi = np.clip(xi, 0, self.w - 1)
        yi = np.clip(yi, 0, self.h - 1)
        return xi, yi

    def _project(self, points: np.ndarray) -> np.ndarray:
        """
        Vectorized BEV projection using np scatter operations.
        All points are processed simultaneously — no Python loops.
        """
        xi, yi = self._compute_indices(points)
        z = points[:, 2]
        intensity = points[:, 3]

        bev = np.zeros(self.observation_shape, dtype=np.float32)

        # Channel 0: Max Height
        bev[:, :, 0] = -np.inf
        np.maximum.at(bev[:, :, 0], (yi, xi), z)
        bev[:, :, 0][bev[:, :, 0] == -np.inf] = 0.0

        # Helper: Min Height (needed for height range)
        min_z = np.full((self.h, self.w), np.inf, dtype=np.float32)
        np.minimum.at(min_z, (yi, xi), z)

        # Channel 1: Height Range
        occupied = min_z < np.inf
        bev[:, :, 1][occupied] = bev[:, :, 0][occupied] - min_z[occupied]

        # Channel 2: Point Density (normalized to [0, 1])
        np.add.at(bev[:, :, 2], (yi, xi), 1.0)
        max_density = bev[:, :, 2].max()
        if max_density > 0:
            bev[:, :, 2] /= max_density

        # Channel 3: Mean Intensity
        intensity_sum = np.zeros((self.h, self.w), dtype=np.float32)
        count = np.zeros((self.h, self.w), dtype=np.float32)
        np.add.at(intensity_sum, (yi, xi), intensity)
        np.add.at(count, (yi, xi), 1.0)
        nonzero = count > 0
        bev[:, :, 3][nonzero] = intensity_sum[nonzero] / count[nonzero]

        # Channel 4: Binary Occupancy
        bev[:, :, 4] = nonzero.astype(np.float32)

        return bev


# ── Self-Test ────────────────────────────────────────────────
if __name__ == '__main__':
    proc = BEVProcessor()

    # Synthetic scene: ground plane + a car
    np.random.seed(42)

    ground = np.column_stack([
        np.random.uniform(-30, 30, 10000),
        np.random.uniform(-30, 30, 10000),
        np.full(10000, -1.7) + np.random.normal(0, 0.05, 10000),
        np.random.uniform(0.1, 0.3, 10000),
    ]).astype(np.float32)

    car = np.column_stack([
        np.random.uniform(-1, 1, 2000),
        np.random.uniform(8, 12, 2000),
        np.random.uniform(-1.7, 0.0, 2000),
        np.random.uniform(0.5, 0.9, 2000),
    ]).astype(np.float32)

    points = np.vstack([ground, car])
    bev = proc.process(points)

    print(f"Input shape:       {points.shape}")
    print(f"Output shape:      {bev.shape}")
    print(f"Observation spec:  {proc.observation_shape}")
    print(f"Occupied cells:    {int(bev[:, :, 4].sum())} / {proc.h * proc.w}")
    print(f"Max height:        {bev[:, :, 0].max():.2f} m")
    print(f"Max height range:  {bev[:, :, 1].max():.2f} m")
    print("✓ BEVProcessor self-test PASSED")
