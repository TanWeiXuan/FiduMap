"""Marker-local geometry helpers.

Marker-local coordinates use a y-up convention with the marker center at the
origin and the marker plane at z=0. OpenCV ArUco/AprilTag detections are stored
in top-left, top-right, bottom-right, bottom-left image-corner order, so PnP
must use object points in that same order.
"""

from __future__ import annotations

import numpy as np


def validate_marker_size(marker_size_m: float) -> float:
    size = float(marker_size_m)
    if not np.isfinite(size) or size <= 0.0:
        raise ValueError("Marker side length must be a positive finite value in meters.")
    return size


def marker_corners_y_up(marker_size_m: float) -> np.ndarray:
    """Return marker corners as bottom-left, bottom-right, top-right, top-left."""

    s = validate_marker_size(marker_size_m)
    half = s / 2.0
    return np.array(
        [
            [-half, -half, 0.0],
            [half, -half, 0.0],
            [half, half, 0.0],
            [-half, half, 0.0],
        ],
        dtype=float,
    )


def marker_corners_for_detector_order(
    marker_size_m: float,
    detector_corner_order: str = "opencv_tl_tr_br_bl",
) -> np.ndarray:
    """Return object points ordered to match stored detector image corners.

    The default ``opencv_tl_tr_br_bl`` matches OpenCV marker order:
    top-left, top-right, bottom-right, bottom-left. With y-up marker-local
    coordinates, that means:

    - top-left: ``[-s/2,  s/2, 0]``
    - top-right: ``[ s/2,  s/2, 0]``
    - bottom-right: ``[ s/2, -s/2, 0]``
    - bottom-left: ``[-s/2, -s/2, 0]``
    """

    orders = {
        "opencv_tl_tr_br_bl": [3, 2, 1, 0],
        "y_up_bl_br_tr_tl": [0, 1, 2, 3],
    }
    if detector_corner_order not in orders:
        raise ValueError(f"Unsupported detector corner order: {detector_corner_order}")

    corners = marker_corners_y_up(marker_size_m)
    return corners[orders[detector_corner_order], :]



def marker_corners_for_export_order(marker_size_m: float) -> np.ndarray:
    """Return marker corners in CSV export order.

    Export order is:

    0. bottom-left: ``[-s/2, -s/2, 0]``
    1. top-left: ``[-s/2,  s/2, 0]``
    2. top-right: ``[ s/2,  s/2, 0]``
    3. bottom-right: ``[ s/2, -s/2, 0]``
    """

    corners = marker_corners_y_up(marker_size_m)
    return corners[[0, 3, 2, 1], :]
