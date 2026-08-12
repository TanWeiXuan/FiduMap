"""Small public data models for runtime localisation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from map_builder.camera_models import CameraModel
from map_builder.geometry import SE3


@dataclass(frozen=True)
class CameraConfig:
    """A camera model and its camera-to-body transform ``T_B_C``."""

    model: CameraModel
    T_B_C: SE3

    def __post_init__(self) -> None:
        if not isinstance(self.model, CameraModel):
            raise TypeError("model must implement the FiduMap CameraModel interface.")
        if not isinstance(self.T_B_C, SE3):
            raise TypeError("T_B_C must be an SE3.")
        R = self.T_B_C.R
        t = self.T_B_C.t
        if not np.all(np.isfinite(R)) or not np.all(np.isfinite(t)):
            raise ValueError("T_B_C must contain only finite values.")
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-7) or not np.isclose(
            np.linalg.det(R), 1.0, atol=1e-7
        ):
            raise ValueError("T_B_C.R must be a proper rotation matrix.")


@dataclass(frozen=True)
class AbsolutePoseResult:
    """Result of one independent absolute-pose solve."""

    success: bool
    T_W_B: SE3 | None
    inlier_indices: np.ndarray
    num_correspondences: int
    num_inliers: int
    mean_reprojection_error_px: float | None
