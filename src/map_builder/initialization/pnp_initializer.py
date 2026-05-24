"""Per-detection PnP initialization."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from map_builder.camera_models import CameraModel
from map_builder.geometry import SE3, marker_object_points_for_detector, validate_marker_size
from map_builder.project import PnPObservation, ProjectStore


ProgressCallback = Callable[[int, int, str], None]


class PnPInitializer:
    def __init__(
        self,
        project_folder: Path,
        store: ProjectStore,
        camera_model: CameraModel,
        marker_size_m: float,
        detector_corner_order: str = "opencv_tl_tr_br_bl",
        progress_callback: ProgressCallback | None = None,
    ):
        self.project_folder = Path(project_folder)
        self.store = store
        self.camera_model = camera_model
        self.marker_size_m = validate_marker_size(marker_size_m)
        self.detector_corner_order = detector_corner_order
        self.progress_callback = progress_callback

    def run(self) -> list[PnPObservation]:
        detections = self.store.list_detection_rows_for_initialization()
        total = len(detections)
        observations: list[PnPObservation] = []
        object_points = marker_object_points_for_detector(self.marker_size_m, self.detector_corner_order)
        for index, row in enumerate(detections, start=1):
            self._notify(index, total, f"image {row['image_id']} marker {row['marker_id']}")
            observations.append(self._solve_one(row, object_points))
        self.store.replace_pnp_observations(observations)
        return observations

    def _solve_one(self, row: dict[str, Any], object_points: np.ndarray) -> PnPObservation:
        try:
            normalized_points = self._corners_to_normalized_points(np.asarray(row["corners"], dtype=float))
            T_C_M, rvec, tvec = solve_marker_pnp(object_points, normalized_points)
            error = reprojection_error_px(self.camera_model, object_points, np.asarray(row["corners"], dtype=float), T_C_M)
            return PnPObservation(
                image_id=int(row["image_id"]),
                detection_id=int(row["detection_id"]),
                marker_id=int(row["marker_id"]),
                success=True,
                rvec=rvec.reshape(3).astype(float).tolist(),
                tvec=tvec.reshape(3).astype(float).tolist(),
                T_C_M=T_C_M.to_json_dict(),
                reprojection_error_px=error,
            )
        except Exception as exc:
            return PnPObservation(
                image_id=int(row["image_id"]),
                detection_id=int(row["detection_id"]),
                marker_id=int(row["marker_id"]),
                success=False,
                error_message=str(exc),
            )

    def _corners_to_normalized_points(self, corners_px: np.ndarray) -> np.ndarray:
        if corners_px.shape != (4, 2):
            raise ValueError(f"Expected four 2D marker corners, got shape {corners_px.shape}.")
        rays = self.camera_model.unproject_many(corners_px)
        if np.any(rays[:, 2] <= 1e-9):
            raise ValueError("At least one detected corner unprojects to a non-forward camera ray.")
        return rays[:, :2] / rays[:, 2:3]

    def _notify(self, index: int, total: int, message: str) -> None:
        if self.progress_callback is not None:
            self.progress_callback(index, total, message)


def solve_marker_pnp(object_points: np.ndarray, normalized_image_points: np.ndarray) -> tuple[SE3, np.ndarray, np.ndarray]:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for PnP initialization.") from exc

    obj = np.asarray(object_points, dtype=np.float64).reshape(4, 3)
    img = np.asarray(normalized_image_points, dtype=np.float64).reshape(4, 2)
    camera_matrix = np.eye(3, dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    flags = getattr(cv2, "SOLVEPNP_IPPE_SQUARE", getattr(cv2, "SOLVEPNP_ITERATIVE", 0))
    success, rvec, tvec = cv2.solvePnP(obj, img, camera_matrix, dist_coeffs, flags=flags)
    if not success:
        raise RuntimeError("cv2.solvePnP failed for marker observation.")
    T_C_M = SE3.from_rvec_tvec(rvec, tvec)
    return T_C_M, rvec, tvec


def reprojection_error_px(
    camera_model: CameraModel,
    object_points: np.ndarray,
    image_points_px: np.ndarray,
    T_C_M: SE3,
) -> float:
    points_c = T_C_M.transform_points(object_points)
    if np.any(points_c[:, 2] <= 1e-9):
        raise ValueError("PnP solution places at least one marker corner behind the camera.")
    projected = camera_model.project_many(points_c)
    if np.any(~np.isfinite(projected)):
        raise ValueError("PnP solution produced invalid reprojections.")
    errors = np.linalg.norm(projected - np.asarray(image_points_px, dtype=float), axis=1)
    return float(np.mean(errors))
