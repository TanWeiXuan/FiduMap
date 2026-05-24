import numpy as np
import pytest

from map_builder.camera_models import PinholeRadTanCameraModel
from map_builder.geometry import SE3, marker_corners_for_detector_order
from map_builder.optimization.ba_costs import MarkerObservationReprojectionCost


def test_marker_observation_cost_residual_and_jacobians_at_ground_truth() -> None:
    pytest.importorskip("pyceres")
    camera = PinholeRadTanCameraModel(640, 480, 500, 500, 320, 240, 0, 0, 0, 0, 0)
    marker_size = 0.2
    T_W_C = SE3(np.eye(3), np.array([0.0, 0.0, -1.5]))
    T_W_M = SE3.identity()
    object_points = marker_corners_for_detector_order(marker_size)
    corners_px = camera.project_many(T_W_C.inverse().transform_points(T_W_M.transform_points(object_points)))
    cost = MarkerObservationReprojectionCost(camera, marker_size, corners_px, object_points)
    residuals = np.zeros(8, dtype=np.float64)
    J_cam = np.zeros((8, 6), dtype=np.float64)
    J_marker = np.zeros((8, 6), dtype=np.float64)

    assert cost.Evaluate([T_W_C.log(), T_W_M.log()], residuals, [J_cam, J_marker])
    np.testing.assert_allclose(residuals, np.zeros(8), atol=1e-9)
    assert np.linalg.norm(J_cam) > 0.0
    assert np.linalg.norm(J_marker) > 0.0
