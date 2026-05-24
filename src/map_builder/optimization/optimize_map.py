"""High-level BA-like map optimizer."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from map_builder.camera_models import CameraModel
from map_builder.geometry import SE3
from map_builder.project import (
    BAConfig,
    BAResult,
    BARunSummary,
    OptimizedCameraPose,
    OptimizedMarkerPose,
    ProjectStore,
)

from .ba_problem import BAProblem
from .outlier_rejection import find_outlier_observations
from .pyceres_backend import solve_marker_ba
from .residuals import BAObservation, compute_reprojection_error_records, residual_statistics


ProgressCallback = Callable[[str], None]


class MapOptimizer:
    def __init__(self, store: ProjectStore, camera_model: CameraModel):
        self.store = store
        self.camera_model = camera_model

    def run_ba(self, config: BAConfig, progress_callback: ProgressCallback | None = None) -> BAResult:
        ba_run_id = self.store.create_ba_run(config)
        try:
            self._notify(progress_callback, "Loading seed poses and detections")
            problem = self._build_problem()
            if not problem.camera_ids or not problem.marker_ids:
                raise RuntimeError("No seed camera/marker poses are available for BA.")
            if config.backend_name.strip().lower() != "pyceres":
                raise RuntimeError("Only backend_name='pyceres' is supported by the default BA backend.")

            self._notify(progress_callback, "Running first pyceres BA pass")
            first = solve_marker_ba(problem, config)
            camera_poses, marker_poses = first.camera_poses, first.marker_poses
            all_records = compute_reprojection_error_records(
                ba_run_id,
                self.camera_model,
                problem.marker_size_m,
                problem.observations,
                camera_poses,
                marker_poses,
            )
            outliers = find_outlier_observations(
                all_records,
                config.corner_outlier_threshold_px,
                config.marker_observation_outlier_threshold_px,
            )

            final = first
            if config.run_outlier_second_pass and outliers:
                self._notify(progress_callback, f"Running second pyceres BA pass without {len(outliers)} outlier observation(s)")
                second_problem = problem.with_initial_poses(camera_poses, marker_poses)
                final = solve_marker_ba(second_problem, config, excluded_observations=outliers)
                camera_poses, marker_poses = final.camera_poses, final.marker_poses

            records = compute_reprojection_error_records(
                ba_run_id,
                self.camera_model,
                problem.marker_size_m,
                problem.observations,
                camera_poses,
                marker_poses,
                outlier_observation_keys=outliers,
            )
            stats = residual_statistics([record for record in records if not record.is_outlier])
            optimized_cameras = [
                OptimizedCameraPose(image_id=image_id, ba_run_id=ba_run_id, T_W_C=pose.to_json_dict())
                for image_id, pose in sorted(camera_poses.items())
            ]
            optimized_markers = [
                OptimizedMarkerPose(
                    marker_id=marker_id,
                    ba_run_id=ba_run_id,
                    T_W_M=pose.to_json_dict(),
                    is_anchor=marker_id == problem.anchor_marker_id,
                )
                for marker_id, pose in sorted(marker_poses.items())
            ]
            self.store.replace_optimized_camera_poses(ba_run_id, optimized_cameras)
            self.store.replace_optimized_marker_poses(ba_run_id, optimized_markers)
            self.store.replace_reprojection_errors(ba_run_id, records)
            self.store.complete_ba_run(
                ba_run_id,
                success=bool(final.success),
                num_iterations=final.num_iterations,
                initial_cost=final.initial_cost,
                final_cost=final.final_cost,
                mean_reprojection_error_px=_float_or_none(stats["mean"]),
                median_reprojection_error_px=_float_or_none(stats["median"]),
                max_reprojection_error_px=_float_or_none(stats["max"]),
                num_observations=len(problem.observations),
                num_corners=int(stats["num_corners"] or 0),
                num_outlier_observations=len(outliers),
                solver_report=final.solver_report,
                error_message=None if final.success else final.solver_report,
            )
            summary = self.store.get_ba_summary(ba_run_id)
            assert summary is not None
            self._notify(progress_callback, "BA complete")
            return BAResult(ba_run_id, bool(final.success), summary, optimized_markers, optimized_cameras, records, "pyceres")
        except Exception as exc:
            self.store.complete_ba_run(ba_run_id, success=False, error_message=str(exc))
            summary = self.store.get_ba_summary(ba_run_id)
            assert summary is not None
            raise

    def _build_problem(self) -> BAProblem:
        marker_size = self.store.get_marker_size_m()
        if marker_size is None:
            raise RuntimeError("Set marker side length before running BA.")
        anchor_marker_id = self.store.get_anchor_marker_id()
        seed_cameras = {pose.image_id: SE3.from_json_dict(pose.T_W_C) for pose in self.store.get_seed_camera_poses()}
        seed_markers = {pose.marker_id: SE3.from_json_dict(pose.T_W_M) for pose in self.store.get_seed_marker_poses()}
        seed_markers[anchor_marker_id] = SE3.identity()
        if not seed_cameras:
            raise RuntimeError("No seed camera poses are available. Run seed initialization first.")
        if not seed_markers:
            raise RuntimeError("No seed marker poses are available. Run seed initialization first.")

        observations: list[BAObservation] = []
        for row in self.store.list_detection_rows_for_initialization():
            image_id = int(row["image_id"])
            marker_id = int(row["marker_id"])
            if image_id not in seed_cameras or marker_id not in seed_markers:
                continue
            observations.append(
                BAObservation(
                    image_id=image_id,
                    marker_id=marker_id,
                    corners_px=np.asarray(row["corners"], dtype=float).reshape(4, 2),
                )
            )
        if not observations:
            raise RuntimeError("No detections connect initialized cameras and markers.")

        camera_ids = sorted({obs.image_id for obs in observations})
        marker_ids = sorted({obs.marker_id for obs in observations} | {anchor_marker_id})
        return BAProblem(
            camera_model=self.camera_model,
            marker_size_m=marker_size,
            observations=observations,
            camera_ids=camera_ids,
            marker_ids=marker_ids,
            anchor_marker_id=anchor_marker_id,
            initial_camera_poses=seed_cameras,
            initial_marker_poses=seed_markers,
        )

    def _notify(self, callback: ProgressCallback | None, message: str) -> None:
        if callback is not None:
            callback(message)


def _float_or_none(value: Any) -> float | None:
    return None if value is None else float(value)
