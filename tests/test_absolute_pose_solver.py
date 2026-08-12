from __future__ import annotations

import csv

import numpy as np
import pytest

import absolute_pose_solver.solver as solver_module
from absolute_pose_solver import AbsolutePoseSolver, CameraConfig
from map_builder.camera_models import OmniRadTanCameraModel, PinholeRadTanCameraModel
from map_builder.geometry import SE3
from map_builder.project.models import MarkerDetection


def _rotation_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    sx, cx = np.sin(rx), np.cos(rx)
    sy, cy = np.sin(ry), np.cos(ry)
    sz, cz = np.sin(rz), np.cos(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=float)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=float)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=float)
    return Rz @ Ry @ Rx


def _pinhole() -> PinholeRadTanCameraModel:
    return PinholeRadTanCameraModel(
        1280, 960, 700.0, 710.0, 640.0, 480.0, 0.002, -0.0002, 0.0001, -0.0001, 0.0
    )


def _omni() -> OmniRadTanCameraModel:
    return OmniRadTanCameraModel(
        1280, 960, 620.0, 625.0, 640.0, 480.0, 0.65, 0.001, -0.0001, 0.0, 0.0, 0.0
    )


T_W_B_GROUND_TRUTH = SE3(
    _rotation_xyz(np.deg2rad(8), np.deg2rad(-13), np.deg2rad(21)),
    np.array([2.0, -3.0, 5.0]),
)


def _make_map(tmp_path, T_W_B: SE3 = T_W_B_GROUND_TRUTH):
    centers_B = [
        (-1.0, -0.5, 4.8),
        (0.8, 0.6, 5.7),
        (-0.3, 1.0, 6.6),
        (1.2, -0.8, 7.3),
        (0.1, -1.2, 5.9),
    ]
    marker_map = {}
    half = 0.28
    export_offsets = np.array(
        [[-half, -half, 0], [-half, half, 0], [half, half, 0], [half, -half, 0]],
        dtype=float,
    )
    for marker_id, center in enumerate(centers_B, start=10):
        marker_map[marker_id] = T_W_B.transform_points(export_offsets + np.asarray(center))

    path = tmp_path / "marker_map.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "x", "y", "z"])
        for marker_id, corners in marker_map.items():
            for corner_index, point in enumerate(corners):
                writer.writerow([(marker_id << 2) | corner_index, *point])
    return path, marker_map


def _detections(marker_map, model, T_B_C, T_W_B=T_W_B_GROUND_TRUTH):
    detections = []
    T_C_B = T_B_C.inverse()
    T_B_W = T_W_B.inverse()
    for marker_id, corners_W in marker_map.items():
        detector_corners_W = corners_W[[1, 2, 3, 0]]
        points_B = T_B_W.transform_points(detector_corners_W)
        points_C = T_C_B.transform_points(points_B)
        rays_C = points_C / np.linalg.norm(points_C, axis=1, keepdims=True)
        pixels = model.project_many(rays_C)
        assert np.all(np.isfinite(pixels))
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


def _assert_pose_close(actual: SE3, expected: SE3, translation_atol=2e-3, angle_atol=2e-3):
    np.testing.assert_allclose(actual.t, expected.t, atol=translation_atol)
    rotation_error = actual.R.T @ expected.R
    angle = np.arccos(np.clip((np.trace(rotation_error) - 1.0) / 2.0, -1.0, 1.0))
    assert angle < angle_atol


def test_single_camera_synthetic_pose(tmp_path) -> None:
    map_path, marker_map = _make_map(tmp_path)
    camera = CameraConfig(_pinhole(), SE3.identity())
    solver = AbsolutePoseSolver(map_path, {"front": camera}, ransac_threshold_deg=0.2)

    result = solver.solve({"front": _detections(marker_map, camera.model, camera.T_B_C)})

    assert result.success
    assert result.num_inliers == result.num_correspondences == 20
    assert result.mean_reprojection_error_px is not None
    assert result.mean_reprojection_error_px < 1e-3
    _assert_pose_close(result.T_W_B, T_W_B_GROUND_TRUTH)


def test_nonidentity_extrinsic_and_transform_direction(tmp_path) -> None:
    map_path, marker_map = _make_map(tmp_path)
    T_B_C = SE3(
        _rotation_xyz(np.deg2rad(4), np.deg2rad(-7), np.deg2rad(11)),
        np.array([0.35, -0.12, 0.18]),
    )
    camera = CameraConfig(_pinhole(), T_B_C)
    solver = AbsolutePoseSolver(map_path, {"front": camera}, ransac_threshold_deg=0.2)

    result = solver.solve({"front": _detections(marker_map, camera.model, T_B_C)})

    assert result.success
    _assert_pose_close(result.T_W_B, T_W_B_GROUND_TRUTH)
    body_origin_in_world = result.T_W_B.transform_points(np.zeros(3))
    np.testing.assert_allclose(body_origin_in_world, [2, -3, 5], atol=2e-3)
    assert np.linalg.norm(result.T_W_B.t - T_W_B_GROUND_TRUTH.inverse().t) > 1.0


def test_multi_camera_uses_generalized_ransac(tmp_path, monkeypatch) -> None:
    map_path, marker_map = _make_map(tmp_path)
    cameras = {
        "front": CameraConfig(_pinhole(), SE3.identity()),
        "offset": CameraConfig(
            _omni(),
            SE3(
                _rotation_xyz(np.deg2rad(-3), np.deg2rad(9), np.deg2rad(-12)),
                np.array([0.55, 0.15, -0.08]),
            ),
        ),
    }
    real_native = solver_module._load_native()
    calls = []

    class NativeSpy:
        def solve_ransac_upnp(self, *args):
            calls.append(args[4])
            return real_native.solve_ransac_upnp(*args)

    monkeypatch.setattr(solver_module, "_load_native", lambda: NativeSpy())
    solver = AbsolutePoseSolver(map_path, cameras, ransac_threshold_deg=0.2)
    observations = {
        camera_id: _detections(marker_map, config.model, config.T_B_C)
        for camera_id, config in cameras.items()
    }

    result = solver.solve(observations)

    assert calls == [True]
    assert result.success
    assert result.num_correspondences == 40
    _assert_pose_close(result.T_W_B, T_W_B_GROUND_TRUTH)


def test_ransac_rejects_coherent_bad_marker(tmp_path) -> None:
    map_path, marker_map = _make_map(tmp_path)
    camera = CameraConfig(_pinhole(), SE3.identity())
    detections = _detections(marker_map, camera.model, camera.T_B_C)
    corrupted_marker_id = detections[1].marker_id
    corrupted = np.asarray(detections[1].corners) + np.array([260.0, -190.0])
    detections[1] = MarkerDetection(
        marker_family="aruco",
        dictionary_name="DICT_6X6_250",
        marker_id=corrupted_marker_id,
        corners=corrupted.tolist(),
        corner_refinement_method="synthetic-outlier",
    )
    solver = AbsolutePoseSolver(map_path, {"front": camera}, ransac_threshold_deg=0.4)

    result = solver.solve({"front": detections})

    assert result.success
    _assert_pose_close(result.T_W_B, T_W_B_GROUND_TRUTH, translation_atol=5e-3, angle_atol=5e-3)
    corrupted_indices = set(range(4, 8))
    assert corrupted_indices.isdisjoint(result.inlier_indices.tolist())


def test_normal_failure_cases_do_not_call_native(tmp_path, monkeypatch) -> None:
    map_path, marker_map = _make_map(tmp_path)
    camera = CameraConfig(_pinhole(), SE3.identity())
    solver = AbsolutePoseSolver(map_path, {"front": camera})
    monkeypatch.setattr(
        solver_module,
        "_load_native",
        lambda: pytest.fail("native solver should not be loaded for insufficient data"),
    )
    unknown_detection = MarkerDetection(
        marker_family="aruco",
        dictionary_name="DICT_6X6_250",
        marker_id=999,
        corners=[[0, 0], [1, 0], [1, 1], [0, 1]],
        corner_refinement_method="synthetic",
    )

    assert not solver.solve({}).success
    assert not solver.solve({"front": []}).success
    assert not solver.solve({"front": [unknown_detection]}).success
    assert not solver.solve({"unknown-camera": []}).success
