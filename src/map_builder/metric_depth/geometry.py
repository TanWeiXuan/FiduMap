from __future__ import annotations

from typing import Any

import numpy as np


FORWARD_RAY_EPS = 1e-8


def _pose_part(pose: Any, name: str) -> Any:
    return getattr(pose, name) if hasattr(pose, name) else pose[name]


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


def range_to_world_points(
    range_m: np.ndarray,
    valid_mask: np.ndarray,
    camera_model: Any,
    T_W_C: Any,
) -> tuple[np.ndarray, np.ndarray]:
    ranges = np.asarray(range_m, dtype=float)
    mask = np.asarray(valid_mask, dtype=bool)
    height, width = ranges.shape
    pixels, rays = pixels_and_rays(camera_model, width, height)
    flat_valid = mask.ravel() & np.isfinite(ranges.ravel()) & (ranges.ravel() > 0.0)
    flat_valid &= np.all(np.isfinite(rays), axis=1) & (rays[:, 2] > FORWARD_RAY_EPS)
    points_c = rays[flat_valid] * ranges.ravel()[flat_valid, None]
    R = np.asarray(_pose_part(T_W_C, "R"), dtype=float)
    t = np.asarray(_pose_part(T_W_C, "t"), dtype=float)
    points_w = (R @ points_c.T).T + t
    return points_w.astype(np.float32), pixels[flat_valid].astype(np.int32)


def intersect_marker_plane(
    pixels: np.ndarray,
    camera_model: Any,
    T_W_C: Any,
    T_W_M: Any,
    marker_size_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Intersect rays with the finite marker square; return valid, z, range."""
    px = np.asarray(pixels, dtype=float).reshape(-1, 2)
    rays_c = np.asarray(camera_model.unproject_many(px), dtype=float)
    Rwc = np.asarray(_pose_part(T_W_C, "R"), dtype=float)
    twc = np.asarray(_pose_part(T_W_C, "t"), dtype=float)
    Rwm = np.asarray(_pose_part(T_W_M, "R"), dtype=float)
    twm = np.asarray(_pose_part(T_W_M, "t"), dtype=float)
    origin_m = Rwm.T @ (twc - twm)
    direction_m = (Rwm.T @ Rwc @ rays_c.T).T
    denom = direction_m[:, 2]
    valid = np.all(np.isfinite(rays_c), axis=1) & (rays_c[:, 2] > FORWARD_RAY_EPS)
    valid &= np.isfinite(denom) & (np.abs(denom) > FORWARD_RAY_EPS)
    distance = np.full(len(px), np.nan, dtype=float)
    distance[valid] = -origin_m[2] / denom[valid]
    valid &= np.isfinite(distance) & (distance > 0.0)
    hit_m = origin_m + direction_m * distance[:, None]
    half = float(marker_size_m) / 2.0 + 1e-9
    valid &= (np.abs(hit_m[:, 0]) <= half) & (np.abs(hit_m[:, 1]) <= half)
    z = np.zeros(len(px), dtype=np.float32)
    radial = np.zeros(len(px), dtype=np.float32)
    radial[valid] = distance[valid].astype(np.float32)
    z[valid] = (distance[valid] * rays_c[valid, 2]).astype(np.float32)
    valid &= np.isfinite(z) & (z > 0.0)
    z[~valid] = 0.0
    radial[~valid] = 0.0
    return valid, z, radial


def deterministic_decimate(count: int, maximum: int) -> np.ndarray:
    if count <= 0 or maximum <= 0:
        return np.empty(0, dtype=np.int64)
    if count <= maximum:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, maximum, dtype=np.int64)
