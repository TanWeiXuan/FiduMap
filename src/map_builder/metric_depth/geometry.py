from __future__ import annotations

from typing import Any

import numpy as np


FORWARD_RAY_EPS = 1e-8


def pixels_and_rays(camera_model: Any, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    u, v = np.meshgrid(np.arange(width, dtype=float), np.arange(height, dtype=float))
    pixels = np.column_stack((u.ravel(), v.ravel()))
    return pixels, np.asarray(camera_model.unproject_many(pixels), dtype=float)


def z_depth_to_range(z_depth_m: np.ndarray, camera_model: Any, eps: float = FORWARD_RAY_EPS) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(z_depth_m, dtype=np.float32)
    height, width = z.shape
    _pixels, rays = pixels_and_rays(camera_model, width, height)
    ray_z = rays[:, 2].reshape(height, width)
    valid = np.isfinite(z) & (z > 0.0) & np.isfinite(ray_z) & (ray_z > eps)
    ranges = np.zeros_like(z, dtype=np.float32)
    ranges[valid] = z[valid] / ray_z[valid]
    valid &= np.isfinite(ranges) & (ranges > 0.0)
    ranges[~valid] = 0.0
    return ranges, valid


def deterministic_decimate(count: int, maximum: int) -> np.ndarray:
    if count <= 0 or maximum <= 0:
        return np.empty(0, dtype=np.int64)
    if count <= maximum:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, maximum, dtype=np.int64)
