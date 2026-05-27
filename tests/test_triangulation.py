import numpy as np

from map_builder.camera_models import PinholeRadTanCameraModel
from map_builder.dense_reconstruction.models import TriangulationConfig
from map_builder.dense_reconstruction.triangulation import triangulate_multiview, triangulate_pair_matches, triangulate_two_view


def _camera():
    return PinholeRadTanCameraModel(640, 480, 400, 400, 320, 240, 0, 0, 0, 0, 0)


def _project(camera, point, T):
    R = np.asarray(T["R"], dtype=float)
    C = np.asarray(T["t"], dtype=float)
    return camera.project(R.T @ (point - C))


def test_triangulate_shape():
    C1 = np.array([0.0, 0, 0])
    C2 = np.array([1.0, 0, 0])
    d1 = np.array([[0, 0, 1.0]])
    d2 = np.array([[-0.1, 0, 1.0]])
    d1 = d1 / np.linalg.norm(d1, axis=1, keepdims=True)
    d2 = d2 / np.linalg.norm(d2, axis=1, keepdims=True)
    X, g = triangulate_two_view(C1, d1, C2, d2)
    assert X.shape == (1, 3)
    assert g.shape == (1,)


def test_triangulate_pair_and_multiview_recover_world_point():
    camera = _camera()
    T1 = {"R": np.eye(3).tolist(), "t": [0, 0, 0]}
    T2 = {"R": np.eye(3).tolist(), "t": [1, 0, 0]}
    X_true = np.array([0.2, 0.1, 5.0])
    p1 = _project(camera, X_true, T1)
    p2 = _project(camera, X_true, T2)
    cfg = TriangulationConfig(min_triangulation_angle_deg=0.1, max_reprojection_error_px=0.25)
    X, valid = triangulate_pair_matches(np.array([p1]), np.array([p2]), T1, T2, camera, cfg)
    assert valid.tolist() == [True]
    assert np.allclose(X[0], X_true, atol=1e-6)

    X_mv, metrics = triangulate_multiview(
        [(1, 0, p1[0], p1[1]), (2, 0, p2[0], p2[1])],
        {1: T1, 2: T2},
        camera,
        cfg,
    )
    assert X_mv is not None
    assert np.allclose(X_mv, X_true, atol=1e-6)
    assert metrics["max_reprojection_error_px"] < 1e-6


def test_triangulation_rejects_large_reprojection_error():
    camera = _camera()
    T1 = {"R": np.eye(3).tolist(), "t": [0, 0, 0]}
    T2 = {"R": np.eye(3).tolist(), "t": [1, 0, 0]}
    X_true = np.array([0.2, 0.1, 5.0])
    p1 = _project(camera, X_true, T1)
    p2 = _project(camera, X_true, T2) + np.array([100.0, 0.0])
    cfg = TriangulationConfig(min_triangulation_angle_deg=0.1, max_reprojection_error_px=1.0)
    _X, valid = triangulate_pair_matches(np.array([p1]), np.array([p2]), T1, T2, camera, cfg)
    assert valid.tolist() == [False]
