"""OpenGV-backed absolute-pose solver for FiduMap CSV maps."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from map_builder.geometry import SE3
from map_builder.project.models import MarkerDetection

from .models import AbsolutePoseResult, CameraConfig


# OpenCV detector order TL, TR, BR, BL -> CSV order BL, TL, TR, BR.
DETECTOR_TO_MAP_CORNER = np.array([1, 2, 3, 0], dtype=np.int64)


def load_marker_map_csv(path: str | Path) -> dict[int, np.ndarray]:
    """Load a FiduMap ``id,x,y,z`` corner map grouped by marker ID."""

    map_path = Path(path)
    corners: dict[int, dict[int, np.ndarray]] = {}
    with map_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "x", "y", "z"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("Marker map CSV must contain id, x, y, and z fields.")

        for row_number, row in enumerate(reader, start=2):
            raw_id = row.get("id")
            try:
                point_id = int(raw_id) if raw_id is not None else -1
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid point ID on CSV row {row_number}: {raw_id!r}.") from exc
            if point_id < 0:
                raise ValueError(f"Point ID on CSV row {row_number} must be non-negative.")

            marker_id = point_id >> 2
            corner_index = point_id & 0b11
            try:
                point = np.array([float(row[axis]) for axis in ("x", "y", "z")], dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid XYZ coordinate on CSV row {row_number}.") from exc
            if not np.all(np.isfinite(point)):
                raise ValueError(f"XYZ coordinates on CSV row {row_number} must be finite.")

            marker_corners = corners.setdefault(marker_id, {})
            if corner_index in marker_corners:
                raise ValueError(
                    f"Duplicate marker corner ({marker_id}, {corner_index}) on CSV row {row_number}."
                )
            marker_corners[corner_index] = point

    marker_map: dict[int, np.ndarray] = {}
    for marker_id, marker_corners in corners.items():
        missing = sorted(set(range(4)) - marker_corners.keys())
        if missing:
            raise ValueError(f"Marker {marker_id} is missing corner indices {missing}.")
        marker_map[marker_id] = np.stack([marker_corners[index] for index in range(4)])
    return marker_map


class AbsolutePoseSolver:
    """Estimate the body pose ``T_W_B`` from one or more rigid cameras."""

    def __init__(
        self,
        map_path: str | Path,
        cameras: Mapping[str, CameraConfig],
        ransac_threshold_deg: float = 1.0,
        ransac_max_iterations: int = 1000,
        ransac_probability: float = 0.999,
    ) -> None:
        if not cameras:
            raise ValueError("At least one camera must be configured.")
        if any(not isinstance(camera_id, str) or not camera_id for camera_id in cameras):
            raise ValueError("Camera IDs must be non-empty strings.")
        if any(not isinstance(config, CameraConfig) for config in cameras.values()):
            raise TypeError("Every camera value must be a CameraConfig.")

        threshold_deg = float(ransac_threshold_deg)
        if not np.isfinite(threshold_deg) or not 0.0 < threshold_deg < 180.0:
            raise ValueError("ransac_threshold_deg must be finite and between 0 and 180.")
        if not isinstance(ransac_max_iterations, int) or ransac_max_iterations <= 0:
            raise ValueError("ransac_max_iterations must be a positive integer.")
        probability = float(ransac_probability)
        if not np.isfinite(probability) or not 0.0 < probability < 1.0:
            raise ValueError("ransac_probability must be finite and between 0 and 1.")

        self.marker_map = load_marker_map_csv(map_path)
        self.cameras = dict(cameras)
        self.ransac_threshold = 1.0 - np.cos(np.deg2rad(threshold_deg))
        self.ransac_max_iterations = ransac_max_iterations
        self.ransac_probability = probability

    def solve(
        self,
        detections_by_camera: Mapping[str, Sequence[MarkerDetection]],
    ) -> AbsolutePoseResult:
        if any(camera_id not in self.cameras for camera_id in detections_by_camera):
            return self._failure()

        built = self._build_correspondences(detections_by_camera)
        bearings_C, points_W, offsets_B, rotations_B_C, camera_ids, pixels = built
        count = len(points_W)
        if count < 4:
            return self._failure(num_correspondences=count)

        active_camera_count = len(set(camera_ids))
        use_generalized = active_camera_count > 1
        native = _load_native()
        success, candidates, inlier_indices = native.solve_ransac_upnp(
            bearings_C,
            points_W,
            offsets_B,
            rotations_B_C,
            use_generalized,
            float(self.ransac_threshold),
            self.ransac_max_iterations,
            self.ransac_probability,
        )
        inliers = np.asarray(inlier_indices, dtype=np.int64)
        if not success or len(inliers) < 3:
            return self._failure(count, inliers)

        best_pose: SE3 | None = None
        best_error: float | None = None
        for candidate in np.asarray(candidates, dtype=float):
            pose = self._valid_pose(candidate)
            if pose is None:
                continue
            error = self._mean_reprojection_error(
                pose, inliers, points_W, camera_ids, pixels
            )
            if error is not None and (best_error is None or error < best_error):
                best_pose = pose
                best_error = error

        if best_pose is None:
            return self._failure(count, inliers)
        return AbsolutePoseResult(
            success=True,
            T_W_B=best_pose,
            inlier_indices=inliers,
            num_correspondences=count,
            num_inliers=len(inliers),
            mean_reprojection_error_px=best_error,
        )

    def _build_correspondences(
        self,
        detections_by_camera: Mapping[str, Sequence[MarkerDetection]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray]:
        bearing_batches: list[np.ndarray] = []
        point_batches: list[np.ndarray] = []
        offset_batches: list[np.ndarray] = []
        rotation_batches: list[np.ndarray] = []
        camera_ids: list[str] = []
        pixel_batches: list[np.ndarray] = []

        for camera_id, detections in detections_by_camera.items():
            camera = self.cameras[camera_id]
            for detection in detections:
                marker_corners = self.marker_map.get(int(detection.marker_id))
                if marker_corners is None:
                    continue
                pixels = np.asarray(detection.corners, dtype=float)
                if pixels.shape != (4, 2) or not np.all(np.isfinite(pixels)):
                    raise ValueError("Each MarkerDetection must contain four finite pixel corners.")
                bearings = np.asarray(camera.model.unproject_many(pixels), dtype=float)
                if bearings.shape != (4, 3) or not np.all(np.isfinite(bearings)):
                    raise ValueError("CameraModel.unproject_many() returned invalid bearings.")
                norms = np.linalg.norm(bearings, axis=1)
                if not np.allclose(norms, 1.0, atol=1e-7):
                    raise ValueError("CameraModel.unproject_many() must return unit bearings.")

                bearing_batches.append(bearings)
                point_batches.append(marker_corners[DETECTOR_TO_MAP_CORNER])
                offset_batches.append(np.repeat(camera.T_B_C.t[None, :], 4, axis=0))
                rotation_batches.append(np.repeat(camera.T_B_C.R[None, :, :], 4, axis=0))
                camera_ids.extend([camera_id] * 4)
                pixel_batches.append(pixels)

        if not bearing_batches:
            return (
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 3, 3), dtype=np.float64),
                [],
                np.empty((0, 2), dtype=np.float64),
            )
        return (
            np.ascontiguousarray(np.concatenate(bearing_batches), dtype=np.float64),
            np.ascontiguousarray(np.concatenate(point_batches), dtype=np.float64),
            np.ascontiguousarray(np.concatenate(offset_batches), dtype=np.float64),
            np.ascontiguousarray(np.concatenate(rotation_batches), dtype=np.float64),
            camera_ids,
            np.ascontiguousarray(np.concatenate(pixel_batches), dtype=np.float64),
        )

    def _mean_reprojection_error(
        self,
        T_W_B: SE3,
        inliers: np.ndarray,
        points_W: np.ndarray,
        camera_ids: list[str],
        observed_pixels: np.ndarray,
    ) -> float | None:
        T_B_W = T_W_B.inverse()
        errors: list[np.ndarray] = []
        for camera_id in dict.fromkeys(camera_ids[index] for index in inliers):
            indices = np.array(
                [index for index in inliers if camera_ids[index] == camera_id], dtype=np.int64
            )
            camera = self.cameras[camera_id]
            points_B = T_B_W.transform_points(points_W[indices])
            points_C = camera.T_B_C.inverse().transform_points(points_B)
            norms = np.linalg.norm(points_C, axis=1)
            if np.any(~np.isfinite(norms)) or np.any(norms <= 1e-12):
                return None
            rays_C = points_C / norms[:, None]
            try:
                predicted = np.asarray(camera.model.project_many(rays_C), dtype=float)
            except (ValueError, FloatingPointError):
                return None
            if predicted.shape != (len(indices), 2) or not np.all(np.isfinite(predicted)):
                return None
            errors.append(np.linalg.norm(predicted - observed_pixels[indices], axis=1))
        if not errors:
            return None
        return float(np.mean(np.concatenate(errors)))

    @staticmethod
    def _valid_pose(candidate: Any) -> SE3 | None:
        transform = np.asarray(candidate, dtype=float)
        if transform.shape != (3, 4) or not np.all(np.isfinite(transform)):
            return None
        R = transform[:, :3]
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-5) or not np.isclose(
            np.linalg.det(R), 1.0, atol=1e-5
        ):
            return None
        return SE3(R, transform[:, 3])

    @staticmethod
    def _failure(
        num_correspondences: int = 0,
        inlier_indices: np.ndarray | None = None,
    ) -> AbsolutePoseResult:
        inliers = (
            np.empty(0, dtype=np.int64)
            if inlier_indices is None
            else np.asarray(inlier_indices, dtype=np.int64)
        )
        return AbsolutePoseResult(
            success=False,
            T_W_B=None,
            inlier_indices=inliers,
            num_correspondences=num_correspondences,
            num_inliers=len(inliers),
            mean_reprojection_error_px=None,
        )


def _load_native() -> Any:
    try:
        from . import _opengv_native
    except ImportError as exc:
        raise RuntimeError(
            "The OpenGV extension is not built. Install the project with "
            "`python -m pip install .` before using AbsolutePoseSolver.solve()."
        ) from exc
    return _opengv_native
