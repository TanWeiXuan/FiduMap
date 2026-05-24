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
    residuals = np.empty(8, dtype=np.float64)
    eps = 1e-12

    for i, (X_M, observed_px) in enumerate(zip(object_points, observed)):
        X_W = T_W_M.transform_points(X_M)
        X_C = T_C_W.transform_points(X_W)
        norm = float(np.linalg.norm(X_C))
        if norm <= eps or X_C[2] <= eps:
            residuals[2 * i : 2 * i + 2] = invalid_projection_penalty_px
            continue
        ray = X_C / norm
        predicted_px = camera_model.project(ray)
        if not np.all(np.isfinite(predicted_px)):
            residuals[2 * i : 2 * i + 2] = invalid_projection_penalty_px
            continue
        residuals[2 * i : 2 * i + 2] = observed_px - predicted_px

    return residuals


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
