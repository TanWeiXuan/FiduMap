from __future__ import annotations

from typing import Any

import numpy as np

from .models import EpipolarFilterConfig, PairMatchRecord


def relative_pose_21(T_W_C1: dict[str, Any], T_W_C2: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    R1 = np.asarray(T_W_C1["R"], dtype=float)
    t1 = np.asarray(T_W_C1["t"], dtype=float).reshape(3)
    R2 = np.asarray(T_W_C2["R"], dtype=float)
    t2 = np.asarray(T_W_C2["t"], dtype=float).reshape(3)
    return R2.T @ R1, R2.T @ (t1 - t2)


def compute_ray_angular_errors(
    f1: np.ndarray, f2: np.ndarray, R21: np.ndarray, t21: np.ndarray, eps: float = 1e-12
) -> np.ndarray:
    f1 = f1 / np.linalg.norm(f1, axis=1, keepdims=True)
    f2 = f2 / np.linalg.norm(f2, axis=1, keepdims=True)
    rf1 = (R21 @ np.asarray(f1, dtype=float).T).T
    n2 = np.cross(np.broadcast_to(np.asarray(t21, dtype=float).reshape(3), (rf1.shape[0], 3)), rf1)
    return np.abs(np.sum(np.asarray(f2, dtype=float) * n2, axis=1)) / np.maximum(np.linalg.norm(n2, axis=1), eps)


def compute_sampson_errors(
    f1: np.ndarray, f2: np.ndarray, R21: np.ndarray, t21: np.ndarray, eps: float = 1e-12
) -> np.ndarray:
    f1 = f1 / np.linalg.norm(f1, axis=1, keepdims=True)
    f2 = f2 / np.linalg.norm(f2, axis=1, keepdims=True)
    E = _skew(np.asarray(t21, dtype=float).reshape(3)) @ np.asarray(R21, dtype=float)
    Ef1 = (E @ f1.T).T
    Etf2 = (E.T @ f2.T).T
    e = np.sum(f2 * Ef1, axis=1)
    denom = Ef1[:, 0] ** 2 + Ef1[:, 1] ** 2 + Etf2[:, 0] ** 2 + Etf2[:, 1] ** 2
    return (e * e) / np.maximum(denom, eps)


def epipolar_inlier_mask(
    f1: np.ndarray,
    f2: np.ndarray,
    R21: np.ndarray,
    t21: np.ndarray,
    config: EpipolarFilterConfig,
) -> tuple[np.ndarray, np.ndarray]:
    baseline = float(np.linalg.norm(t21))
    if baseline < config.min_baseline_m:
        errors = np.full((len(f1),), np.inf, dtype=float)
        return errors, np.zeros((len(f1),), dtype=bool)
    if config.error_type == "sampson":
        errors = compute_sampson_errors(f1, f2, R21, t21)
        return errors, errors <= float(config.max_sampson_error)
    errors = compute_ray_angular_errors(f1, f2, R21, t21)
    return errors, errors <= float(config.max_ray_angular_error)


def filter_pair_matches(
    matches: list[PairMatchRecord],
    T_W_C1: dict[str, Any],
    T_W_C2: dict[str, Any],
    camera_model: Any,
    config: EpipolarFilterConfig,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    if not matches:
        return [], np.empty((0,), dtype=float), np.empty((0,), dtype=bool)
    pixels1 = np.array([[m.x_a, m.y_a] for m in matches], dtype=float)
    pixels2 = np.array([[m.x_b, m.y_b] for m in matches], dtype=float)
    f1 = camera_model.unproject_many(pixels1)
    f2 = camera_model.unproject_many(pixels2)
    R21, t21 = relative_pose_21(T_W_C1, T_W_C2)
    errors, inliers = epipolar_inlier_mask(f1, f2, R21, t21, config)
    missing_ids = [m for m in matches if m.id is None]
    if missing_ids:
        raise ValueError("Epipolar filtering requires PairMatchRecord.id values from persisted pair_matches rows.")
    ids = [int(m.id) for m in matches]
    return ids, errors, inliers


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dtype=float)
