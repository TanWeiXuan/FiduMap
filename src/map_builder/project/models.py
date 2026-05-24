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
