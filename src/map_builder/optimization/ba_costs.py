"""pyceres cost functions for marker BA."""

from __future__ import annotations

import numpy as np

from .residual_math import compute_marker_observation_residual, finite_difference_jacobian, write_jacobian

try:
    import pyceres  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - exercised by backend error path
    pyceres = None  # type: ignore[assignment]

_CostFunctionBase = object if pyceres is None else pyceres.CostFunction


class MarkerObservationReprojectionCost(_CostFunctionBase):  # type: ignore[misc, valid-type]
    def __init__(
        self,
        camera_model: object,
        marker_size_m: float,
        observed_corners_px: np.ndarray,
        detector_order_object_points: np.ndarray,
        finite_diff_step: float = 1e-6,
        invalid_projection_penalty_px: float = 1e6,
    ):
        if pyceres is None:
            raise RuntimeError("pyceres is required for MarkerObservationReprojectionCost.")
        pyceres.CostFunction.__init__(self)
        self.set_num_residuals(8)
        self.set_parameter_block_sizes([6, 6])
        self.camera_model = camera_model
        self.marker_size_m = float(marker_size_m)
        self.observed_corners_px = np.asarray(observed_corners_px, dtype=np.float64).reshape(4, 2)
        self.detector_order_object_points = np.asarray(detector_order_object_points, dtype=np.float64).reshape(4, 3)
        self.finite_diff_step = float(finite_diff_step)
        self.invalid_projection_penalty_px = float(invalid_projection_penalty_px)

    def compute_residual(self, camera_xi: np.ndarray, marker_xi: np.ndarray) -> np.ndarray:
        return compute_marker_observation_residual(
            self.camera_model,
            camera_xi,
            marker_xi,
            self.detector_order_object_points,
            self.observed_corners_px,
            self.invalid_projection_penalty_px,
        )

    def Evaluate(self, parameters: object, residuals: object, jacobians: object) -> bool:
        camera_xi = np.asarray(parameters[0], dtype=np.float64)  # type: ignore[index]
        marker_xi = np.asarray(parameters[1], dtype=np.float64)  # type: ignore[index]
        r = self.compute_residual(camera_xi, marker_xi)
        np.asarray(residuals)[:] = r

        if jacobians is not None:
            if jacobians[0] is not None:  # type: ignore[index]
                J_cam = finite_difference_jacobian(
                    lambda x: self.compute_residual(x, marker_xi),
                    camera_xi,
                    step=self.finite_diff_step,
                )
                write_jacobian(jacobians[0], J_cam)  # type: ignore[index]
            if jacobians[1] is not None:  # type: ignore[index]
                J_marker = finite_difference_jacobian(
                    lambda x: self.compute_residual(camera_xi, x),
                    marker_xi,
                    step=self.finite_diff_step,
                )
                write_jacobian(jacobians[1], J_marker)  # type: ignore[index]
        return True
