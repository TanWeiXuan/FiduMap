"""Reprojection residuals for map refinement.

Convention: ``T_A_B`` maps points from frame B into frame A. Residuals compare
stored detector-order image corners against projections of fixed marker-local
detector-order corners transformed by optimized marker and camera poses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from map_builder.camera_models import CameraModel
from map_builder.geometry import SE3, marker_corners_for_detector_order
from map_builder.optimization.residual_math import _compute_residuals
from map_builder.project import ReprojectionErrorRecord


@dataclass(frozen=True)
class BAObservation:
    image_id: int
    marker_id: int
    corners_px: np.ndarray


def evaluate_marker_observation_residuals(
    camera_model: CameraModel,
    marker_size_m: float,
    observation: BAObservation,
    T_W_C: SE3,
    T_W_M: SE3,
    detector_corner_order: str = "opencv_tl_tr_br_bl",
    behind_camera_penalty_px: float = 1.0e6,
) -> np.ndarray:
    object_points = marker_corners_for_detector_order(marker_size_m, detector_corner_order)
    return _compute_residuals(
        camera_model,
        object_points,
        observation.corners_px,
        T_W_C,
        T_W_M,
        invalid_projection_penalty_px=behind_camera_penalty_px,
    )


def compute_reprojection_error_records(
    ba_run_id: int,
    camera_model: CameraModel,
    marker_size_m: float,
    observations: list[BAObservation],
    camera_poses: dict[int, SE3],
    marker_poses: dict[int, SE3],
    outlier_observation_keys: set[tuple[int, int]] | None = None,
) -> list[ReprojectionErrorRecord]:
    outliers = outlier_observation_keys or set()
    records: list[ReprojectionErrorRecord] = []
    for obs in observations:
        residuals = evaluate_marker_observation_residuals(
            camera_model,
            marker_size_m,
            obs,
            camera_poses[obs.image_id],
            marker_poses[obs.marker_id],
        ).reshape(4, 2)
        is_outlier = (obs.image_id, obs.marker_id) in outliers
        for corner_index, residual in enumerate(residuals):
            error = float(np.linalg.norm(residual))
            records.append(
                ReprojectionErrorRecord(
                    ba_run_id=ba_run_id,
                    image_id=obs.image_id,
                    marker_id=obs.marker_id,
                    corner_index_detector_order=corner_index,
                    error_px=error,
                    residual_x_px=float(residual[0]),
                    residual_y_px=float(residual[1]),
                    is_outlier=is_outlier,
                )
            )
    return records


def residual_statistics(records: list[ReprojectionErrorRecord]) -> dict[str, float | int | None]:
    if not records:
        return {"mean": None, "median": None, "max": None, "num_corners": 0}
    errors = np.asarray([record.error_px for record in records], dtype=float)
    return {
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "max": float(np.max(errors)),
        "num_corners": int(errors.size),
    }
