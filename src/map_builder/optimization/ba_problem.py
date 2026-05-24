"""Packing and residual evaluation for BA-like map refinement."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from map_builder.camera_models import CameraModel
from map_builder.geometry import SE3

from .residuals import BAObservation, evaluate_marker_observation_residuals


@dataclass
class BAProblem:
    camera_model: CameraModel
    marker_size_m: float
    observations: list[BAObservation]
    camera_ids: list[int]
    marker_ids: list[int]
    anchor_marker_id: int
    initial_camera_poses: dict[int, SE3]
    initial_marker_poses: dict[int, SE3]

    def __post_init__(self) -> None:
        self.optimized_marker_ids = [marker_id for marker_id in self.marker_ids if marker_id != self.anchor_marker_id]

    def pack_initial(self) -> np.ndarray:
        blocks: list[np.ndarray] = []
        for image_id in self.camera_ids:
            blocks.append(self.initial_camera_poses[image_id].log())
        for marker_id in self.optimized_marker_ids:
            blocks.append(self.initial_marker_poses[marker_id].log())
        return np.concatenate(blocks) if blocks else np.zeros(0, dtype=float)

    def unpack(self, x: np.ndarray) -> tuple[dict[int, SE3], dict[int, SE3]]:
        x = np.asarray(x, dtype=float)
        offset = 0
        cameras: dict[int, SE3] = {}
        markers: dict[int, SE3] = {self.anchor_marker_id: SE3.identity()}
        for image_id in self.camera_ids:
            cameras[image_id] = SE3.exp(x[offset : offset + 6])
            offset += 6
        for marker_id in self.optimized_marker_ids:
            markers[marker_id] = SE3.exp(x[offset : offset + 6])
            offset += 6
        return cameras, markers

    def residuals(self, x: np.ndarray, excluded_observations: set[tuple[int, int]] | None = None) -> np.ndarray:
        cameras, markers = self.unpack(x)
        excluded = excluded_observations or set()
        blocks: list[np.ndarray] = []
        for obs in self.observations:
            if (obs.image_id, obs.marker_id) in excluded:
                continue
            if obs.image_id not in cameras or obs.marker_id not in markers:
                continue
            blocks.append(
                evaluate_marker_observation_residuals(
                    self.camera_model,
                    self.marker_size_m,
                    obs,
                    cameras[obs.image_id],
                    markers[obs.marker_id],
                )
            )
        return np.concatenate(blocks) if blocks else np.zeros(0, dtype=float)

    def with_initial_poses(
        self,
        camera_poses: dict[int, SE3],
        marker_poses: dict[int, SE3],
    ) -> "BAProblem":
        return BAProblem(
            camera_model=self.camera_model,
            marker_size_m=self.marker_size_m,
            observations=self.observations,
            camera_ids=self.camera_ids,
            marker_ids=self.marker_ids,
            anchor_marker_id=self.anchor_marker_id,
            initial_camera_poses=camera_poses,
            initial_marker_poses=marker_poses,
        )
