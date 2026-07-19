"""Optional per-reference-image metric depth generation."""

from .models import (
    ALIGNMENT_AFFINE,
    ALIGNMENT_SPLINE_SPATIAL,
    ARTIFACT_SCHEMA_VERSION,
    MetricDepthArtifact,
    MetricDepthConfig,
    MetricDepthMetrics,
    MetricDepthProgress,
    MetricDepthRunSummary,
    MetricAnchor,
    AlignmentAnchorRaster,
)
from .pipeline import MetricDepthPipeline

__all__ = [
    "ALIGNMENT_AFFINE",
    "ALIGNMENT_SPLINE_SPATIAL",
    "ARTIFACT_SCHEMA_VERSION",
    "MetricDepthArtifact",
    "MetricDepthConfig",
    "MetricDepthMetrics",
    "MetricDepthPipeline",
    "MetricDepthProgress",
    "MetricDepthRunSummary",
    "MetricAnchor",
    "AlignmentAnchorRaster",
]
