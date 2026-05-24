"""OpenCV aruco/apriltag marker detection."""

from __future__ import annotations

from typing import Any

from map_builder.project.models import MarkerDetection

from .marker_detector import marker_family_for_dictionary


class OpenCVArucoMarkerDetector:
    def __init__(
        self,
        dictionary_name: str,
        marker_family: str | None = None,
        corner_refinement: str = "auto",
    ):
        self.dictionary_name = dictionary_name
        self.marker_family = marker_family or marker_family_for_dictionary(dictionary_name)
        self.corner_refinement = corner_refinement

        self.cv2 = _load_cv2_with_aruco()
        self.aruco = self.cv2.aruco
        self.dictionary = self._get_dictionary(dictionary_name)
        self.parameters = self._create_detector_parameters()
        self.corner_refinement_method = self._configure_corner_refinement(corner_refinement)

    def detect(self, image_bgr_or_gray: Any) -> list[MarkerDetection]:
        gray = self._as_gray(image_bgr_or_gray)
        corners, ids = self._detect_markers(gray)
        if ids is None or len(ids) == 0:
            return []

        marker_ids = ids.reshape(-1).tolist()
        detections: list[MarkerDetection] = []
        for marker_id, marker_corners in zip(marker_ids, corners):
            points = marker_corners.reshape(4, 2).astype(float).tolist()
            detections.append(
                MarkerDetection(
                    marker_family=self.marker_family,
                    dictionary_name=self.dictionary_name,
                    marker_id=int(marker_id),
                    corners=[[float(u), float(v)] for u, v in points],
                    corner_refinement_method=self.corner_refinement_method,
                )
            )
        return detections

    def _get_dictionary(self, dictionary_name: str) -> Any:
        if not hasattr(self.aruco, dictionary_name):
            raise ValueError(f"OpenCV aruco dictionary is not available: {dictionary_name}")
        dictionary_id = getattr(self.aruco, dictionary_name)
        if hasattr(self.aruco, "getPredefinedDictionary"):
            return self.aruco.getPredefinedDictionary(dictionary_id)
        return self.aruco.Dictionary_get(dictionary_id)

    def _create_detector_parameters(self) -> Any:
        if hasattr(self.aruco, "DetectorParameters"):
            return self.aruco.DetectorParameters()
        if hasattr(self.aruco, "DetectorParameters_create"):
            return self.aruco.DetectorParameters_create()
        raise RuntimeError("OpenCV aruco DetectorParameters API is unavailable.")

    def _configure_corner_refinement(self, corner_refinement: str) -> str:
        method_name = self._resolve_corner_refinement_name(corner_refinement)
        method_value = getattr(self.aruco, method_name, None)
        if method_value is not None:
            self.parameters.cornerRefinementMethod = method_value
        return method_name

    def _resolve_corner_refinement_name(self, corner_refinement: str) -> str:
        normalized = corner_refinement.strip().lower()
        if normalized == "auto":
            if self.marker_family == "apriltag" and hasattr(self.aruco, "CORNER_REFINE_APRILTAG"):
                return "CORNER_REFINE_APRILTAG"
            return "CORNER_REFINE_SUBPIX"

        aliases = {
            "none": "CORNER_REFINE_NONE",
            "subpix": "CORNER_REFINE_SUBPIX",
            "contour": "CORNER_REFINE_CONTOUR",
            "apriltag": "CORNER_REFINE_APRILTAG",
            "corner_refine_none": "CORNER_REFINE_NONE",
            "corner_refine_subpix": "CORNER_REFINE_SUBPIX",
            "corner_refine_contour": "CORNER_REFINE_CONTOUR",
            "corner_refine_apriltag": "CORNER_REFINE_APRILTAG",
        }
        method_name = aliases.get(normalized, corner_refinement)
        if not hasattr(self.aruco, method_name):
            raise ValueError(f"OpenCV corner refinement method is not available: {corner_refinement}")
        return method_name

    def _as_gray(self, image: Any) -> Any:
        if image is None:
            raise ValueError("Cannot detect markers in an empty image.")
        if len(image.shape) == 2:
            return image
        if len(image.shape) == 3:
            return self.cv2.cvtColor(image, self.cv2.COLOR_BGR2GRAY)
        raise ValueError(f"Unsupported image shape for marker detection: {image.shape}")

    def _detect_markers(self, gray: Any) -> tuple[Any, Any]:
        if hasattr(self.aruco, "ArucoDetector"):
            detector = self.aruco.ArucoDetector(self.dictionary, self.parameters)
            corners, ids, _rejected = detector.detectMarkers(gray)
            return corners, ids
        corners, ids, _rejected = self.aruco.detectMarkers(gray, self.dictionary, parameters=self.parameters)
        return corners, ids


def _load_cv2_with_aruco() -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for marker detection. Install opencv-contrib-python.") from exc

    if not hasattr(cv2, "aruco"):
        raise RuntimeError("This OpenCV build does not include cv2.aruco. Install opencv-contrib-python.")
    return cv2
