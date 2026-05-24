"""Dataclasses shared by project storage, indexing, and detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ImageRecord:
    id: int
    rel_path: str
    size_bytes: int
    mtime_ns: int
    width: int | None
    height: int | None
    ignored: bool
    missing: bool
    detection_status: str
    modified_since_detection: bool
    indexed_at: str
    marker_count: int = 0

    def absolute_path(self, project_folder: Path) -> Path:
        return project_folder / self.rel_path


@dataclass(frozen=True)
class MarkerDetection:
    marker_family: str
    dictionary_name: str
    marker_id: int
    corners: list[list[float]]
    corner_refinement_method: str


@dataclass(frozen=True)
class DetectorRunConfig:
    detector_type: str
    dictionary_name: str
    detector_params: dict[str, Any] = field(default_factory=dict)
    marker_family: str | None = None


@dataclass(frozen=True)
class IndexSummary:
    num_found: int = 0
    num_new: int = 0
    num_updated: int = 0
    num_missing: int = 0
    num_unchanged: int = 0


@dataclass(frozen=True)
class PnPObservation:
    image_id: int
    detection_id: int
    marker_id: int
    success: bool
    rvec: list[float] | None = None
    tvec: list[float] | None = None
    T_C_M: dict[str, Any] | None = None
    reprojection_error_px: float | None = None
    error_message: str | None = None
    id: int | None = None
    created_at: str | None = None


@dataclass(frozen=True)
class SeedCameraPose:
    image_id: int
    T_W_C: dict[str, Any]
    source_marker_id: int | None = None
    reprojection_error_px: float | None = None
    created_at: str | None = None


@dataclass(frozen=True)
class SeedMarkerPose:
    marker_id: int
    T_W_M: dict[str, Any]
    source_image_id: int | None = None
    reprojection_error_px: float | None = None
    is_anchor: bool = False
    created_at: str | None = None


@dataclass(frozen=True)
class GraphDiagnostics:
    values: dict[str, Any]


@dataclass(frozen=True)
class ObservationGraphSummary:
    num_camera_nodes: int
    num_marker_nodes: int
    num_camera_marker_edges: int
    num_marker_overlap_edges: int
    connected_components: int
    anchor_marker_exists: bool
    markers_connected_to_anchor: int
    disconnected_markers: list[int]
    observations_per_marker: dict[int, int] = field(default_factory=dict)
    observations_per_image: dict[int, int] = field(default_factory=dict)
