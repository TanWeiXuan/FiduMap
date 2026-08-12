"""Runtime-facing fiducial detector."""

from __future__ import annotations

from typing import Any

from map_builder.detection.opencv_aruco_detector import OpenCVArucoMarkerDetector
from map_builder.project.models import MarkerDetection


class FiducialDetector:
    """Thin wrapper around FiduMap's existing OpenCV marker detector."""

    def __init__(
        self,
        dictionary_name: str = "DICT_6X6_250",
        corner_refinement: str = "auto",
    ) -> None:
        self._detector = OpenCVArucoMarkerDetector(
            dictionary_name=dictionary_name,
            corner_refinement=corner_refinement,
        )

    def detect(self, image: Any) -> list[MarkerDetection]:
        return self._detector.detect(image)
