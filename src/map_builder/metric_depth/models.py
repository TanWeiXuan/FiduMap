from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


BACKEND_DAV2 = "depth_anything_v2_aligned"
ALIGNMENT_AFFINE = "affine_inverse_depth"
ALIGNMENT_SPLINE_SPATIAL = "monotonic_spline_spatial"
ARTIFACT_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class MetricDepthConfig:
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
    alignment_mode: str = ALIGNMENT_SPLINE_SPATIAL
    spline_knot_count: int = 12
    spatial_grid_columns: int = 8
    spatial_grid_rows: int = 6
    maximum_log_depth_correction: float = 0.4
    holdout_fraction: float = 0.2
    marker_sample_grid_size: int = 6
    spatial_smoothness: float = 0.02
    spatial_prior: float = 0.002
    maximum_correction_saturation_fraction: float = 0.25

    def validate(self) -> None:
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
        if self.alignment_mode not in {ALIGNMENT_AFFINE, ALIGNMENT_SPLINE_SPATIAL}:
            raise ValueError("Alignment mode must be affine_inverse_depth or monotonic_spline_spatial.")
        if not 2 <= self.spline_knot_count <= 64:
            raise ValueError("Spline knot count must be between 2 and 64.")
        if not 2 <= self.spatial_grid_columns <= 32 or not 2 <= self.spatial_grid_rows <= 32:
            raise ValueError("Spatial grid dimensions must each be between 2 and 32.")
        if self.spatial_grid_columns * self.spatial_grid_rows > 256:
            raise ValueError("Spatial grid must contain at most 256 control values.")
        if not np.isfinite(self.maximum_log_depth_correction) or self.maximum_log_depth_correction <= 0:
            raise ValueError("Maximum log-depth correction must be positive.")
        if not 0.0 < self.holdout_fraction < 0.5:
            raise ValueError("Holdout fraction must be greater than 0 and less than 0.5.")
        if not 2 <= self.marker_sample_grid_size <= 20:
            raise ValueError("Marker sample grid size must be between 2 and 20.")
        if self.spatial_smoothness < 0 or self.spatial_prior <= 0:
            raise ValueError("Spatial regularization values must be non-negative, with a positive prior.")
        if not 0.0 <= self.maximum_correction_saturation_fraction <= 1.0:
            raise ValueError("Maximum correction saturation fraction must be between 0 and 1.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "MetricDepthConfig":
        """Load saved settings; pre-v2 configurations retain affine behavior."""
        data = dict(values)
        data.setdefault("alignment_mode", ALIGNMENT_AFFINE)
        allowed = cls.__dataclass_fields__
        return cls(**{key: value for key, value in data.items() if key in allowed})


@dataclass(frozen=True)
class MetricAnchor:
    u: float
    v: float
    z_depth_m: float
    range_m: float
    confidence: float
    provenance: str
    group_id: str = ""
    source: str = ""
    raw_confidence: float | None = None
    fit_weight: float = 1.0


@dataclass
class AlignmentAnchorRaster:
    depth_z_m: np.ndarray
    mask: np.ndarray
    confidence: np.ndarray
    provenance: np.ndarray
    anchor_count: int
    pixel_count: int
    occupied_grid_cells: int
    spatial_coverage: float
    anchors: tuple[MetricAnchor, ...] = ()
    image_id: int = 0


@dataclass
class MetricDepthMetrics:
    anchor_point_count: int = 0
    anchor_pixel_count: int = 0
    anchor_spatial_coverage: float = 0.0
    valid_output_fraction: float = 0.0
    median_anchor_absolute_error_m: float | None = None
    median_anchor_relative_error: float | None = None
    alignment_inlier_count: int = 0
    processing_duration_s: float = 0.0
    status: str = "pending"
    error_message: str | None = None
    alignment_mode: str = ALIGNMENT_AFFINE
    training_anchor_count: int = 0
    holdout_anchor_count: int = 0
    marker_group_count: int = 0
    dense_track_anchor_count: int = 0
    training_median_absolute_error_m: float | None = None
    training_median_relative_error: float | None = None
    holdout_median_absolute_error_m: float | None = None
    holdout_median_relative_error: float | None = None
    marker_holdout_median_relative_error: float | None = None
    dense_track_holdout_median_relative_error: float | None = None
    affine_holdout_median_relative_error: float | None = None
    spline_knot_count: int = 0
    spatial_correction_rms: float | None = None
    spatial_correction_maximum: float | None = None
    correction_saturation_fraction: float | None = None
    prediction_extrapolation_fraction: float | None = None
    alignment_duration_s: float = 0.0
    warnings: list[str] = field(default_factory=list)

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
    metadata: dict[str, Any] = field(default_factory=dict)
    metrics: MetricDepthMetrics = field(default_factory=MetricDepthMetrics)
    global_spline_z_depth_m: np.ndarray | None = None
    spatial_log_correction: np.ndarray | None = None
    alignment_extrapolation_mask: np.ndarray | None = None
    anchor_mask: np.ndarray | None = None
    anchor_residual_m: np.ndarray | None = None
    anchor_split: np.ndarray | None = None
    dav2_prediction: np.ndarray | None = None


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
