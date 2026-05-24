"""Detector interfaces and dictionary metadata."""

from __future__ import annotations

from typing import Protocol

from map_builder.project.models import MarkerDetection


ARUCO_DICTIONARY_CHOICES = [
    "DICT_4X4_50",
    "DICT_4X4_100",
    "DICT_4X4_250",
    "DICT_4X4_1000",
    "DICT_5X5_50",
    "DICT_5X5_100",
    "DICT_5X5_250",
    "DICT_5X5_1000",
    "DICT_6X6_50",
    "DICT_6X6_100",
    "DICT_6X6_250",
    "DICT_6X6_1000",
    "DICT_7X7_50",
    "DICT_7X7_100",
    "DICT_7X7_250",
    "DICT_7X7_1000",
    "DICT_ARUCO_ORIGINAL",
]

APRILTAG_DICTIONARY_CHOICES = [
    "DICT_APRILTAG_16h5",
    "DICT_APRILTAG_25h9",
    "DICT_APRILTAG_36h10",
    "DICT_APRILTAG_36h11",
]

DICTIONARY_CHOICES = ARUCO_DICTIONARY_CHOICES + APRILTAG_DICTIONARY_CHOICES


class MarkerDetector(Protocol):
    def detect(self, image_bgr_or_gray: object) -> list[MarkerDetection]:
        ...


def marker_family_for_dictionary(dictionary_name: str, detector_type: str | None = None) -> str:
    if detector_type and detector_type.lower() == "apriltag":
        return "apriltag"
    if dictionary_name.upper().startswith("DICT_APRILTAG"):
        return "apriltag"
    return "aruco"
