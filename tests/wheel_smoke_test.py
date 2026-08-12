"""Installed-wheel smoke test used by cibuildwheel."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np

from absolute_pose_solver import AbsolutePoseSolver, CameraConfig
from absolute_pose_solver import _opengv_native
from map_builder.camera_models import PinholeRadTanCameraModel
from map_builder.geometry import SE3
from map_builder.project import MarkerDetection


def main() -> None:
    assert _opengv_native.__doc__
    camera = PinholeRadTanCameraModel(
        1280, 960, 700.0, 700.0, 640.0, 480.0, 0.0, 0.0, 0.0, 0.0, 0.0
    )
    T_W_C = SE3(
        _rotation_y(np.deg2rad(8.0)),
        np.array([1.0, -0.5, 2.0]),
    )
    marker_map = _marker_map(T_W_C)

    with tempfile.TemporaryDirectory() as folder:
        map_path = Path(folder) / "map.csv"
        _write_map(map_path, marker_map)
        solver = AbsolutePoseSolver(
            map_path,
            {"camera": CameraConfig(camera, SE3.identity())},
            ransac_threshold_deg=0.2,
        )
        detections = _project_detections(marker_map, camera, T_W_C)
        result = solver.solve({"camera": detections})

    assert result.success
    assert result.T_W_B is not None
    np.testing.assert_allclose(result.T_W_B.t, T_W_C.t, atol=2e-3)
    rotation_error = result.T_W_B.R.T @ T_W_C.R
    angle = np.arccos(np.clip((np.trace(rotation_error) - 1.0) / 2.0, -1.0, 1.0))
    assert angle < 2e-3


def _rotation_y(angle: float) -> np.ndarray:
    sine, cosine = np.sin(angle), np.cos(angle)
    return np.array([[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]])


def _marker_map(T_W_C: SE3) -> dict[int, np.ndarray]:
    half = 0.2
    offsets = np.array(
        [[-half, -half, 0.0], [-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0]]
    )
    centers_C = [(-0.8, -0.4, 4.5), (0.7, 0.5, 5.2), (-0.2, 0.9, 6.0), (0.9, -0.7, 6.8)]
    return {
        marker_id: T_W_C.transform_points(offsets + np.asarray(center))
        for marker_id, center in enumerate(centers_C)
    }


def _write_map(path: Path, marker_map: dict[int, np.ndarray]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "x", "y", "z"])
        for marker_id, corners in marker_map.items():
            for corner_index, point in enumerate(corners):
                writer.writerow([(marker_id << 2) | corner_index, *point])


def _project_detections(
    marker_map: dict[int, np.ndarray],
    camera: PinholeRadTanCameraModel,
    T_W_C: SE3,
) -> list[MarkerDetection]:
    detections = []
    T_C_W = T_W_C.inverse()
    for marker_id, corners_W in marker_map.items():
        points_C = T_C_W.transform_points(corners_W[[1, 2, 3, 0]])
        pixels = camera.project_many(points_C)
        detections.append(
            MarkerDetection(
                marker_family="aruco",
                dictionary_name="DICT_6X6_250",
                marker_id=marker_id,
                corners=pixels.tolist(),
                corner_refinement_method="synthetic",
            )
        )
    return detections


if __name__ == "__main__":
    main()
