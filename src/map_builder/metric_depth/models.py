from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


BACKEND_DAV2 = "depth_anything_v2_aligned"
BACKEND_PROMPT_DA = "prompt_depth_anything"


@dataclass(frozen=True)
class MetricDepthConfig:
    backend: str = BACKEND_DAV2
    model_id_or_path: str = "depth-anything/Depth-Anything-V2-Small-hf"
    device: str = "auto"
    allow_download: bool = False
    inference_size: int = 518
    include_dense_track_points: bool = True
    include_marker_surfaces: bool = True
    minimum_anchor_count: int = 12
    minimum_anchor_grid_cells: int = 3
    recompute: bool = False
    maximum_alignment_median_relative_error: float = 0.25
    maximum_anchor_error_m: float = 0.50

    def validate(self) -> None:
        if self.backend not in {BACKEND_DAV2, BACKEND_PROMPT_DA}:
            raise ValueError(f"Unsupported metric-depth backend: {self.backend}")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("Device must be auto, cpu, or cuda.")
        if not self.model_id_or_path.strip():
            raise ValueError("Model ID or local path is required.")
        if self.inference_size <= 0:
            raise ValueError("Inference size must be positive.")
        if self.minimum_anchor_count < 2:
            raise ValueError("Minimum anchor count must be at least 2.")
        if not 1 <= self.minimum_anchor_grid_cells <= 16:
            raise ValueError("Minimum anchor grid cells must be between 1 and 16.")
        if self.maximum_alignment_median_relative_error <= 0:
            raise ValueError("Maximum median relative error must be positive.")
        if self.maximum_anchor_error_m <= 0:
            raise ValueError("Maximum anchor error must be positive.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PromptAnchor:
    u: float
    v: float
    z_depth_m: float
    range_m: float
    confidence: float
    provenance: str


@dataclass
class PromptRaster:
    depth_z_m: np.ndarray
    mask: np.ndarray
    confidence: np.ndarray
    provenance: np.ndarray
    anchor_count: int
    pixel_count: int
    occupied_grid_cells: int
    spatial_coverage: float


@dataclass
class MetricDepthMetrics:
    prompt_point_count: int = 0
    prompt_pixel_count: int = 0
    prompt_spatial_coverage: float = 0.0
    valid_output_fraction: float = 0.0
    median_anchor_absolute_error_m: float | None = None
    median_anchor_relative_error: float | None = None
    alignment_inlier_count: int = 0
    processing_duration_s: float = 0.0
    status: str = "pending"
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MetricDepthArtifact:
    image_id: int
    backend: str
    width: int
    height: int
    z_depth_m: np.ndarray
    range_m: np.ndarray
    valid_mask: np.ndarray
    confidence: np.ndarray
    prompt_depth_z_m: np.ndarray
    prompt_mask: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    metrics: MetricDepthMetrics = field(default_factory=MetricDepthMetrics)


@dataclass(frozen=True)
class MetricDepthProgress:
    stage: str
    message: str
    image_index: int = 0
    total_images: int = 0
    image_id: int | None = None
    fraction: float | None = None


@dataclass
class MetricDepthRunSummary:
    run_id: int
    total: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    cancelled: bool = False
    details: str = ""

