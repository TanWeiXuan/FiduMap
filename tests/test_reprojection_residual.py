import numpy as np

from map_builder.camera_models import PinholeRadTanCameraModel
from map_builder.geometry import SE3, marker_corners_for_detector_order
from map_builder.optimization.residuals import BAObservation, evaluate_marker_observation_residuals


def test_reprojection_residual_is_zero_at_ground_truth() -> None:
    camera = PinholeRadTanCameraModel(640, 480, 500, 500, 320, 240, 0, 0, 0, 0, 0)
    marker_size = 0.2
    T_W_M = SE3(np.eye(3), np.array([0.5, 0.0, 0.0]))
    T_W_C = SE3(np.eye(3), np.array([0.0, 0.0, -1.5]))
    object_points = marker_corners_for_detector_order(marker_size)
    corners_px = camera.project_many(T_W_C.inverse().transform_points(T_W_M.transform_points(object_points)))
    obs = BAObservation(image_id=1, marker_id=2, corners_px=corners_px)
    residuals = evaluate_marker_observation_residuals(camera, marker_size, obs, T_W_C, T_W_M)
    np.testing.assert_allclose(residuals, np.zeros(8), atol=1e-9)
