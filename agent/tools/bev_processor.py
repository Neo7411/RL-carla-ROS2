"""
LiDAR → Bird's-Eye-View tensor (5 channels, no route information).

Channels:
    0: max height
    1: height range (max − min)
    2: point density (normalized)
    3: mean intensity
    4: binary occupancy
"""
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
        self.num_channels = 5

    @property
    def observation_shape(self) -> tuple:
        return (self.h, self.w, self.num_channels)

    def process(self, points: np.ndarray) -> np.ndarray:
        cropped = self._crop(points)
        if len(cropped) == 0:
            return np.zeros(self.observation_shape, dtype=np.float32)
        return self._project(cropped)

    def _crop(self, points: np.ndarray) -> np.ndarray:
        mask = (
            (points[:, 0] >= self.x_min) & (points[:, 0] <= self.x_max) &
            (points[:, 1] >= self.y_min) & (points[:, 1] <= self.y_max) &
            (points[:, 2] >= self.z_min) & (points[:, 2] <= self.z_max)
        )
        return points[mask]

    def _compute_indices(self, points: np.ndarray):
        xi = ((points[:, 0] - self.x_min) / (self.x_max - self.x_min) * self.w).astype(np.int32)
        yi = ((points[:, 1] - self.y_min) / (self.y_max - self.y_min) * self.h).astype(np.int32)
        xi = np.clip(xi, 0, self.w - 1)
        yi = np.clip(yi, 0, self.h - 1)
        return xi, yi

    def _project(self, points: np.ndarray) -> np.ndarray:
        xi, yi = self._compute_indices(points)
        z = points[:, 2]
        intensity = points[:, 3]

        bev = np.zeros(self.observation_shape, dtype=np.float32)

        # Channel 0: Max Height
        bev[:, :, 0] = -np.inf
        np.maximum.at(bev[:, :, 0], (yi, xi), z)
        bev[:, :, 0][bev[:, :, 0] == -np.inf] = 0.0

        # Min Height (scratch) → Channel 1: Height Range
        min_z = np.full((self.h, self.w), np.inf, dtype=np.float32)
        np.minimum.at(min_z, (yi, xi), z)
        occupied = min_z < np.inf
        bev[:, :, 1][occupied] = bev[:, :, 0][occupied] - min_z[occupied]

        # Channel 2: Density (normalized)
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


if __name__ == '__main__':
    proc = BEVProcessor()
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
    print(f"Input {points.shape} → BEV {bev.shape}  "
          f"occupied={int(bev[:,:,4].sum())}  max_h={bev[:,:,0].max():.2f}")
