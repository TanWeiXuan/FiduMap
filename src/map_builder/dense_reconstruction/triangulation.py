from __future__ import annotations

from typing import Any

import numpy as np

from .models import TriangulationConfig


def triangulate_two_view(C1: np.ndarray, d1: np.ndarray, C2: np.ndarray, d2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    C1 = np.asarray(C1, dtype=float).reshape(3)
    C2 = np.asarray(C2, dtype=float).reshape(3)
    d1 = _normalize_rows(np.asarray(d1, dtype=float))
    d2 = _normalize_rows(np.asarray(d2, dtype=float))
    w0 = C1 - C2
    a = np.sum(d1 * d1, axis=1)
    b = np.sum(d1 * d2, axis=1)
    c = np.sum(d2 * d2, axis=1)
    d = np.sum(d1 * w0, axis=1)
    e = np.sum(d2 * w0, axis=1)
    den = np.maximum(a * c - b * b, 1e-12)
    s = (b * e - c * d) / den
    t = (a * e - b * d) / den
    p1 = C1 + s[:, None] * d1
    p2 = C2 + t[:, None] * d2
    return 0.5 * (p1 + p2), np.linalg.norm(p1 - p2, axis=1)


def triangulate_pair_matches(
    pixels1: np.ndarray,
    pixels2: np.ndarray,
    T_W_C1: dict[str, Any],
    T_W_C2: dict[str, Any],
    camera_model: Any,
    config: TriangulationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    R1, C1 = _pose_parts(T_W_C1)
    R2, C2 = _pose_parts(T_W_C2)
    f1 = camera_model.unproject_many(np.asarray(pixels1, dtype=float))
    f2 = camera_model.unproject_many(np.asarray(pixels2, dtype=float))
    d1 = _normalize_rows((R1 @ f1.T).T)
    d2 = _normalize_rows((R2 @ f2.T).T)
    X, gaps = triangulate_two_view(C1, d1, C2, d2)
    angles = ray_angles_deg(d1, d2)
    depths1 = np.sum((X - C1) * d1, axis=1)
    depths2 = np.sum((X - C2) * d2, axis=1)
    err1 = reprojection_errors(X, np.asarray(pixels1, dtype=float), T_W_C1, camera_model)
    err2 = reprojection_errors(X, np.asarray(pixels2, dtype=float), T_W_C2, camera_model)
    ranges = np.maximum(np.linalg.norm(X - C1, axis=1), np.linalg.norm(X - C2, axis=1))
    valid = (
        (depths1 > 0.0)
        & (depths2 > 0.0)
        & (angles >= config.min_triangulation_angle_deg)
        & (err1 <= config.max_reprojection_error_px)
        & (err2 <= config.max_reprojection_error_px)
        & (gaps <= config.max_ray_gap_m)
        & (ranges <= config.max_depth_m)
    )
    return X, valid


def triangulate_multiview(
    observations: list[tuple[int, int, float, float]],
    poses_by_image: dict[int, dict[str, Any]],
    camera_model: Any,
    config: TriangulationConfig,
) -> tuple[np.ndarray | None, dict[str, float]]:
    if len(observations) < config.min_observations:
        return None, {}
    image_ids = [obs[0] for obs in observations]
    if len(set(image_ids)) != len(image_ids):
        return None, {}
    centers = []
    dirs = []
    pixels = []
    for image_id, _feature_idx, x, y in observations:
        if image_id not in poses_by_image:
            return None, {}
        R, C = _pose_parts(poses_by_image[image_id])
        ray = camera_model.unproject(np.array([x, y], dtype=float))
        centers.append(C)
        dirs.append(R @ ray)
        pixels.append([x, y])
    C_arr = np.asarray(centers, dtype=float)
    d_arr = _normalize_rows(np.asarray(dirs, dtype=float))
    A = np.zeros((3, 3), dtype=float)
    b = np.zeros(3, dtype=float)
    eye = np.eye(3)
    for C, d in zip(C_arr, d_arr):
        P = eye - np.outer(d, d)
        A += P
        b += P @ C
    if np.linalg.cond(A) > 1e10:
        return None, {}
    X = np.linalg.solve(A, b)
    depths = np.sum((X[None, :] - C_arr) * d_arr, axis=1)
    if np.any(depths <= 0.0) or float(np.max(depths)) > config.max_depth_m:
        return None, {}
    errors = []
    for (image_id, _feature_idx, x, y) in observations:
        err = reprojection_errors(X[None, :], np.array([[x, y]], dtype=float), poses_by_image[image_id], camera_model)
        errors.append(float(err[0]))
    max_err = float(np.max(errors))
    mean_err = float(np.mean(errors))
    min_angle = min_pairwise_ray_angle_deg(d_arr)
    if min_angle < config.min_triangulation_angle_deg:
        return None, {}
    if mean_err > config.max_mean_reprojection_error_px or max_err > config.max_reprojection_error_px:
        return None, {}
    return X, {
        "mean_reprojection_error_px": mean_err,
        "max_reprojection_error_px": max_err,
        "min_triangulation_angle_deg": float(min_angle),
    }


def reprojection_errors(
    points_w: np.ndarray, pixels: np.ndarray, T_W_C: dict[str, Any], camera_model: Any
) -> np.ndarray:
    R, C = _pose_parts(T_W_C)
    X = np.asarray(points_w, dtype=float)
    rays_c = (R.T @ (X - C).T).T
    projected = camera_model.project_many(rays_c)
    return np.linalg.norm(projected - np.asarray(pixels, dtype=float), axis=1)


def ray_angles_deg(d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    dot = np.clip(np.sum(_normalize_rows(d1) * _normalize_rows(d2), axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def min_pairwise_ray_angle_deg(dirs: np.ndarray) -> float:
    d = _normalize_rows(np.asarray(dirs, dtype=float))
    best = 180.0
    for i in range(len(d)):
        for j in range(i + 1, len(d)):
            angle = float(np.degrees(np.arccos(np.clip(np.dot(d[i], d[j]), -1.0, 1.0))))
            best = min(best, angle)
    return best


def _pose_parts(T_W_C: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(T_W_C["R"], dtype=float), np.asarray(T_W_C["t"], dtype=float).reshape(3)


def _normalize_rows(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
