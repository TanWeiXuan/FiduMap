from pathlib import Path

import pytest


def test_opencv_aruco_detector_smoke() -> None:
    cv2 = pytest.importorskip("cv2")
    if not hasattr(cv2, "aruco"):
        pytest.skip("OpenCV aruco module is unavailable")
    aruco = cv2.aruco
    if not hasattr(aruco, "DICT_4X4_50"):
        pytest.skip("OpenCV aruco dictionary constants are unavailable")

    dictionary_id = getattr(aruco, "DICT_4X4_50")
    if hasattr(aruco, "getPredefinedDictionary"):
        dictionary = aruco.getPredefinedDictionary(dictionary_id)
    else:
        dictionary = aruco.Dictionary_get(dictionary_id)

    if hasattr(aruco, "generateImageMarker"):
        marker = aruco.generateImageMarker(dictionary, 0, 160)
    elif hasattr(aruco, "drawMarker"):
        import numpy as np

        marker = np.zeros((160, 160), dtype="uint8")
        aruco.drawMarker(dictionary, 0, 160, marker, 1)
    else:
        pytest.skip("OpenCV aruco marker generation is unavailable")

    import numpy as np

    canvas = np.full((260, 260), 255, dtype="uint8")
    canvas[50:210, 50:210] = marker

    from map_builder.detection import OpenCVArucoMarkerDetector

    detector = OpenCVArucoMarkerDetector("DICT_4X4_50")
    detections = detector.detect(canvas)
    assert len(detections) >= 1
    assert detections[0].marker_id == 0


def test_marker_detector_import_does_not_require_cv2() -> None:
    from map_builder.detection import DICTIONARY_CHOICES

    assert "DICT_4X4_50" in DICTIONARY_CHOICES
