"""
LiDAR geometria: pontfelho <-> range image.

Forras: lidm/utils/lidar_utils.py (pcd2range, range2pcd)
        lidm/data/base.py         (process_scan, depth_thresh)

Fuggoseg: csak numpy. Nincs benne halo, ez tiszta geometria.
"""

import numpy as np


def pcd2range(pcd, size, fov, depth_range, remission=None, labels=None, **kwargs):
    """
    XYZ pontfelho -> range image (gombi projekcio).

    pcd         : (N, 3) float array, a szenzor koordinatarendszereben
    size        : (H, W) pl. (64, 1024)
    fov         : (fov_up, fov_down) fokban, pl. (3, -25)
    depth_range : (min, max) meterben, pl. (1.0, 56.0)

    return: proj_range (H, W) float32, ahol nincs pont ott -1
            proj_feature (H, W) vagy None
    """
    # laser parameters
    fov_up = fov[0] / 180.0 * np.pi  # field of view up in rad
    fov_down = fov[1] / 180.0 * np.pi  # field of view down in rad
    fov_range = abs(fov_down) + abs(fov_up)  # get field of view total in rad

    # get depth (distance) of all points
    depth = np.linalg.norm(pcd, 2, axis=1)

    # mask points out of range
    mask = np.logical_and(depth > depth_range[0], depth < depth_range[1])
    depth, pcd = depth[mask], pcd[mask]

    # get scan components
    scan_x, scan_y, scan_z = pcd[:, 0], pcd[:, 1], pcd[:, 2]

    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov_range  # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= size[1]  # in [0.0, W]
    proj_y *= size[0]  # in [0.0, H]

    # round and clamp for use as index
    proj_x = np.maximum(0, np.minimum(size[1] - 1, np.floor(proj_x))).astype(np.int32)  # in [0,W-1]
    proj_y = np.maximum(0, np.minimum(size[0] - 1, np.floor(proj_y))).astype(np.int32)  # in [0,H-1]

    # order in decreasing depth
    # Ez fontos: tobb pont eshet ugyanabba a cellaba. A csokkeno rendezes miatt
    # a KOZELEBBI pont irja felul a tavolabbit, ami fizikailag helyes (az takar).
    order = np.argsort(depth)[::-1]
    proj_x, proj_y = proj_x[order], proj_y[order]

    # project depth
    depth = depth[order]
    proj_range = np.full(size, -1, dtype=np.float32)
    proj_range[proj_y, proj_x] = depth

    # project point feature
    if remission is not None:
        remission = remission[mask][order]
        proj_feature = np.full(size, -1, dtype=np.float32)
        proj_feature[proj_y, proj_x] = remission
    elif labels is not None:
        labels = labels[mask][order]
        proj_feature = np.full(size, 0, dtype=np.float32)
        proj_feature[proj_y, proj_x] = labels
    else:
        proj_feature = None

    return proj_range, proj_feature


def range2pcd(range_img, fov, depth_range, depth_scale, log_scale=True, label=None, color=None, **kwargs):
    """
    Range image -> XYZ pontfelho. A pcd2range inverze.

    range_img : (H, W), [0, 1] tartomanyban (tehat a [-1,1]-bol mar vissza kell
                skalazni: x * 0.5 + 0.5)

    return: pcd (M, 3), color (M, 3), label
    """
    # laser parameters
    size = range_img.shape
    fov_up = fov[0] / 180.0 * np.pi  # field of view up in rad
    fov_down = fov[1] / 180.0 * np.pi  # field of view down in rad
    fov_range = abs(fov_down) + abs(fov_up)  # get field of view total in rad

    # inverse transform from depth
    depth = (range_img * depth_scale).flatten()
    if log_scale:
        depth = np.exp2(depth) - 1

    scan_x, scan_y = np.meshgrid(np.arange(size[1]), np.arange(size[0]))
    scan_x = scan_x.astype(np.float64) / size[1]
    scan_y = scan_y.astype(np.float64) / size[0]

    yaw = (np.pi * (scan_x * 2 - 1)).flatten()
    pitch = ((1.0 - scan_y) * fov_range - abs(fov_down)).flatten()

    pcd = np.zeros((len(yaw), 3))
    pcd[:, 0] = np.cos(yaw) * np.cos(pitch) * depth
    pcd[:, 1] = -np.sin(yaw) * np.cos(pitch) * depth
    pcd[:, 2] = np.sin(pitch) * depth

    # mask out invalid points
    mask = np.logical_and(depth > depth_range[0], depth < depth_range[1])
    pcd = pcd[mask, :]

    # label
    if label is not None:
        label = label.flatten()[mask]

    # default point color
    if color is not None:
        color = color.reshape(-1, 3)[mask, :]
    else:
        color = np.ones((pcd.shape[0], 3)) * [0.7, 0.7, 1]

    return pcd, color, label


def get_depth_thresh(depth_scale, log_scale=True):
    """
    Az ervenyessegi kuszob a maszkhoz.
    Forras: lidm/data/base.py:30-33
    """
    if log_scale:
        return (np.log2(1. / 255. + 1) / depth_scale) * 2. - 1 + 1e-6
    else:
        return (1. / 255. / depth_scale) * 2. - 1 + 1e-6


def process_scan(range_img, depth_scale, log_scale=True, depth_thresh=None):
    """
    Nyers range image -> normalizalt tensor + maszk.

    Az eredetiben ez a DatasetBase metodusa volt, de csak harom dolgot hasznalt
    self-bol (log_scale, depth_scale, depth_thresh), ezert itt sima fuggveny.
    Igy nem kell orokolni a DatasetBase-t, ami behozna az egesz
    augmentacio/kamera/degradation gepezetet.

    range_img : (H, W), a pcd2range kimenete (-1 ahol nincs pont)

    return: range_img (1, H, W) float, [-1, 1]
            range_mask (1, H, W), -1 ahol ervenytelen, kulonben 1
    """
    if depth_thresh is None:
        depth_thresh = get_depth_thresh(depth_scale, log_scale)

    range_img = np.where(range_img < 0, 0, range_img)

    if log_scale:
        # log scale: a kozeli tartomany felbontasa no
        range_img = np.log2(range_img + 0.0001 + 1)

    range_img = range_img / depth_scale
    range_img = range_img * 2. - 1.

    range_img = np.clip(range_img, -1, 1)
    range_img = np.expand_dims(range_img, axis=0)

    # mask
    range_mask = np.ones_like(range_img)
    range_mask[range_img < depth_thresh] = -1

    return range_img, range_mask
