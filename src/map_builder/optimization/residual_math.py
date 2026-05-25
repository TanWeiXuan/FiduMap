"""Pure-Python residual math for marker BA.

Convention: ``T_A_B`` maps points from frame B into frame A. Pose parameter
blocks use ``xi = [rho_x, rho_y, rho_z, phi_x, phi_y, phi_z]`` and
``T = SE3.exp(xi)``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from map_builder.geometry import SE3


def compute_marker_observation_residual(
    camera_model: Any,
    camera_xi: np.ndarray,
    marker_xi: np.ndarray,
    object_points_marker: np.ndarray,
    observed_corners_px: np.ndarray,
    invalid_projection_penalty_px: float = 1e6,
) -> np.ndarray:
    T_W_C = SE3.exp(np.asarray(camera_xi, dtype=np.float64))
    T_W_M = SE3.exp(np.asarray(marker_xi, dtype=np.float64))
    T_C_W = T_W_C.inverse()
    object_points = np.asarray(object_points_marker, dtype=np.float64).reshape(4, 3)
    observed = np.asarray(observed_corners_px, dtype=np.float64).reshape(4, 2)
    eps = 1e-12

    points_w = T_W_M.transform_points(object_points)
    points_c = T_C_W.transform_points(points_w)
    norms = np.linalg.norm(points_c, axis=1)
    valid_mask = (points_c[:, 2] > eps) & (norms > eps)

    predicted = np.full((4, 2), invalid_projection_penalty_px, dtype=np.float64)
    if np.any(valid_mask):
        rays = points_c[valid_mask] / norms[valid_mask, None]
        projected = np.asarray(camera_model.project_many(rays), dtype=np.float64).reshape(-1, 2)
        finite_mask = np.all(np.isfinite(projected), axis=1)
        if np.any(finite_mask):
            valid_indices = np.flatnonzero(valid_mask)
            predicted[valid_indices[finite_mask]] = projected[finite_mask]

    residuals = np.full((4, 2), invalid_projection_penalty_px, dtype=np.float64)
    finite_pred_mask = np.all(np.isfinite(predicted), axis=1)
    if np.any(finite_pred_mask):
        residuals[finite_pred_mask] = observed[finite_pred_mask] - predicted[finite_pred_mask]

    return residuals.reshape(8)


def finite_difference_jacobian(
    func: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    step: float = 1e-6,
) -> np.ndarray:
    x0 = np.asarray(x, dtype=np.float64)
    r0 = np.asarray(func(x0), dtype=np.float64)
    J = np.zeros((r0.size, x0.size), dtype=np.float64)
    for j in range(x0.size):
        h = step * max(1.0, abs(float(x0[j])))
        xp = x0.copy()
        xm = x0.copy()
        xp[j] += h
        xm[j] -= h
        rp = np.asarray(func(xp), dtype=np.float64)
        rm = np.asarray(func(xm), dtype=np.float64)
        J[:, j] = (rp - rm) / (2.0 * h)
    return J


def write_jacobian(dst: Any, J: np.ndarray) -> None:
    arr = np.asarray(dst)
    J = np.asarray(J, dtype=np.float64)
    if arr.ndim == 1:
        arr[:] = J.reshape(-1)
    else:
        arr[:, :] = J
