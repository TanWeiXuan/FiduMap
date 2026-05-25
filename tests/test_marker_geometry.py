import numpy as np

from map_builder.geometry import marker_corners_y_up, marker_corners_for_detector_order


def test_marker_corners_y_up() -> None:
    corners = marker_corners_y_up(2.0)
    np.testing.assert_allclose(corners[0], [-1.0, -1.0, 0.0])
    np.testing.assert_allclose(corners[2], [1.0, 1.0, 0.0])


def test_marker_object_points_match_opencv_corner_order() -> None:
    points = marker_corners_for_detector_order(2.0, "opencv_tl_tr_br_bl")
    np.testing.assert_allclose(
        points,
        [
            [-1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
            [-1.0, -1.0, 0.0],
        ],
    )
