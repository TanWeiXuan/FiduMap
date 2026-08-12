"""Online absolute-pose localisation against a FiduMap marker map."""

from .detector import FiducialDetector
from .models import AbsolutePoseResult, CameraConfig
from .solver import AbsolutePoseSolver, DETECTOR_TO_MAP_CORNER, load_marker_map_csv

__all__ = [
    "AbsolutePoseResult",
    "AbsolutePoseSolver",
    "CameraConfig",
    "DETECTOR_TO_MAP_CORNER",
    "FiducialDetector",
    "load_marker_map_csv",
]
