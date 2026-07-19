from __future__ import annotations

import logging
import time
import numpy as np

from ..alignment import monotonic_spline_spatial_alignment, robust_affine_inverse_depth_alignment
from ..geometry import z_depth_to_range
from ..models import (
    ALIGNMENT_AFFINE,
    ALIGNMENT_SPLINE_SPATIAL,
    ARTIFACT_SCHEMA_VERSION,
    MetricDepthArtifact,
    MetricDepthMetrics,
    MetricDepthProgress,
)


LOG = logging.getLogger(__name__)


class DepthAnythingV2AlignedBackend:
    def __init__(self) -> None:
        self.model = None
        self.processor = None
        self.device = "cpu"

    def close(self) -> None:
        self.model = None
        self.processor = None

    def load(self, config: object) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as exc:
            raise RuntimeError("missing optional metric-depth inference dependency; install src/map_builder/requirements-depth.txt") from exc
        local_only = not bool(getattr(config, "allow_download"))
        reference = str(getattr(config, "model_id_or_path"))
        try:
            self.processor = AutoImageProcessor.from_pretrained(reference, local_files_only=local_only)
            self.model = AutoModelForDepthEstimation.from_pretrained(reference, local_files_only=local_only)
        except Exception as exc:
            mode = "local cache/directory" if local_only else "configured model reference"
            raise RuntimeError(f"Depth Anything V2 model unavailable from {mode}: {reference}: {exc}") from exc
        requested = str(getattr(config, "device"))
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was selected but torch.cuda.is_available() is false.")
        self.device = "cuda" if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()) else "cpu"
        self.model.to(self.device).eval()

    def predict_relative(self, image_rgb: np.ndarray, config: object) -> np.ndarray:
        import torch
        from PIL import Image

        assert self.processor is not None and self.model is not None
        image = Image.fromarray(np.asarray(image_rgb, dtype=np.uint8), mode="RGB")
        size = int(getattr(config, "inference_size"))
        inputs = self.processor(images=image, return_tensors="pt", size={"height": size, "width": size})
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        processed = self.processor.post_process_depth_estimation(outputs, target_sizes=[(image.height, image.width)])
        return processed[0]["predicted_depth"].detach().float().cpu().numpy().astype(np.float32)

    def infer(self, image_rgb: np.ndarray, anchors: object, camera_model: object, config: object, progress: object = None) -> MetricDepthArtifact:
        inference_started = time.perf_counter()
        relative = self.predict_relative(image_rgb, config)
        inference_duration = time.perf_counter() - inference_started
        mode = str(getattr(config, "alignment_mode", ALIGNMENT_AFFINE))
        if mode == ALIGNMENT_SPLINE_SPATIAL:
            artifact = self._infer_spline_spatial(relative, anchors, camera_model, config, progress)
            artifact.metadata["dav2_inference_duration_s"] = inference_duration
            return artifact
        if mode != ALIGNMENT_AFFINE:
            raise RuntimeError(f"unsupported metric-depth alignment mode: {mode}")
        started = time.perf_counter()
        result = robust_affine_inverse_depth_alignment(
            relative,
            anchors.depth_z_m,
            anchors.mask,
            anchors.confidence,
            int(getattr(config, "minimum_anchor_count")),
            int(getattr(config, "minimum_anchor_grid_cells")),
            float(getattr(config, "maximum_alignment_median_relative_error")),
        )
        ranges, ray_valid = z_depth_to_range(result.z_depth_m, camera_model)
        valid = result.valid_mask & ray_valid & bool(result.success)
        confidence = _confidence_from_support(
            valid,
            anchors.mask,
            anchors.confidence,
            result.median_relative_error,
            int(np.count_nonzero(result.inlier_mask)),
            max(anchors.pixel_count, 1),
        )
        metrics = MetricDepthMetrics(
            anchor_point_count=anchors.anchor_count,
            anchor_pixel_count=anchors.pixel_count,
            anchor_spatial_coverage=anchors.spatial_coverage,
            valid_output_fraction=float(np.mean(valid)),
            median_anchor_absolute_error_m=result.median_absolute_error_m,
            median_anchor_relative_error=result.median_relative_error,
            alignment_inlier_count=int(np.count_nonzero(result.inlier_mask)),
            status="success" if result.success else "failed",
            error_message=result.error_message,
            alignment_mode=ALIGNMENT_AFFINE,
            training_anchor_count=anchors.pixel_count,
            training_median_absolute_error_m=result.median_absolute_error_m,
            training_median_relative_error=result.median_relative_error,
            alignment_duration_s=time.perf_counter() - started,
        )
        h, w = relative.shape
        artifact = MetricDepthArtifact(
            0,
            "depth_anything_v2_aligned",
            w,
            h,
            result.z_depth_m,
            ranges,
            valid,
            confidence,
            {
                "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                "alignment_mode": ALIGNMENT_AFFINE,
                "alignment_a": result.coefficient_a,
                "alignment_b": result.coefficient_b,
            },
            metrics,
        )
        artifact.metadata["dav2_inference_duration_s"] = inference_duration
        return artifact

    def _infer_spline_spatial(self, relative: np.ndarray, anchors: object, camera_model: object, config: object, progress: object) -> MetricDepthArtifact:
        started = time.perf_counter()
        _alignment_progress(progress, "building_balanced_anchors", "Building balanced metric anchors")
        _alignment_progress(progress, "splitting_anchors", "Splitting training and holdout anchors")
        _alignment_progress(progress, "fitting_spline", "Fitting monotonic depth spline")
        stage_messages = {
            "fitting_spatial_correction": "Fitting spatial correction field",
            "evaluating_holdout": "Evaluating held-out anchors",
            "generating_confidence": "Generating confidence map",
        }
        result = monotonic_spline_spatial_alignment(
            relative, anchors.anchors, int(getattr(anchors, "image_id", 0)), config,
            lambda stage: _alignment_progress(progress, stage, stage_messages[stage]),
        )
        ranges, ray_valid = z_depth_to_range(result.final_z_depth_m, camera_model)
        valid = result.valid_mask & ray_valid & bool(result.success)
        confidence = np.asarray(result.confidence, dtype=np.float32).copy()
        confidence[~valid] = 0.0
        training = result.training_metrics
        holdout = result.holdout_metrics
        correction = result.correction_statistics
        spline_holdout = holdout.get("median_relative_error")
        if result.affine_holdout_median_relative_error is None or spline_holdout is None:
            comparison = {"outcome": "unavailable", "relative_error_delta": None}
        else:
            delta = float(result.affine_holdout_median_relative_error) - float(spline_holdout)
            comparison = {"outcome": "improvement" if delta > 0.0 else "regression", "relative_error_delta": delta}
        marker_groups = {a.group_id for a in anchors.anchors if (a.source or a.provenance) == "marker_surface"}
        dense_count = sum((a.source or a.provenance) == "dense_track" for a in anchors.anchors)
        training_count = int(training.get("count") or 0)
        metrics = MetricDepthMetrics(
            anchor_point_count=anchors.anchor_count,
            anchor_pixel_count=anchors.pixel_count,
            anchor_spatial_coverage=anchors.spatial_coverage,
            valid_output_fraction=float(np.mean(valid)),
            median_anchor_absolute_error_m=training.get("median_absolute_error_m"),
            median_anchor_relative_error=training.get("median_relative_error"),
            alignment_inlier_count=int(training.get("robust_inlier_count") or 0),
            status="success" if result.success else "failed",
            error_message=result.error_message,
            alignment_mode=ALIGNMENT_SPLINE_SPATIAL,
            training_anchor_count=training_count,
            holdout_anchor_count=int(holdout.get("count") or 0),
            marker_group_count=len(marker_groups),
            dense_track_anchor_count=int(dense_count),
            training_median_absolute_error_m=training.get("median_absolute_error_m"),
            training_median_relative_error=training.get("median_relative_error"),
            holdout_median_absolute_error_m=holdout.get("median_absolute_error_m"),
            holdout_median_relative_error=holdout.get("median_relative_error"),
            marker_holdout_median_relative_error=holdout.get("marker_median_relative_error"),
            dense_track_holdout_median_relative_error=holdout.get("dense_track_median_relative_error"),
            affine_holdout_median_relative_error=result.affine_holdout_median_relative_error,
            spline_direction=result.spline_direction,
            spline_knot_count=len(result.spline_knots_x),
            spatial_correction_rms=correction.get("unclamped_rms"),
            spatial_correction_maximum=correction.get("unclamped_maximum_absolute"),
            correction_saturation_fraction=correction.get("saturation_fraction"),
            prediction_extrapolation_fraction=correction.get("extrapolation_fraction"),
            alignment_duration_s=time.perf_counter() - started,
            warnings=list(result.warnings),
        )
        h, w = relative.shape
        metadata = {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "alignment_mode": ALIGNMENT_SPLINE_SPATIAL,
            "prediction_normalization": result.prediction_normalization,
            "spline_knots_x": result.spline_knots_x.tolist(),
            "spline_knots_log_z": result.spline_knots_y.tolist(),
            "spline_direction": result.spline_direction,
            "spatial_grid_dimensions": list(result.spatial_grid_coefficients.shape[::-1]),
            "spatial_grid_coefficients": result.spatial_grid_coefficients.tolist(),
            "spatial_regularization": {
                "smoothness": float(getattr(config, "spatial_smoothness", 0.02)),
                "prior": float(getattr(config, "spatial_prior", 0.002)),
            },
            "maximum_log_depth_correction": float(getattr(config, "maximum_log_depth_correction", 0.4)),
            "correction_statistics": correction,
            "training_metrics": training,
            "holdout_metrics": holdout,
            "affine_baseline_metrics": {
                "holdout_median_relative_error": result.affine_holdout_median_relative_error,
                "comparison": comparison,
            },
            "warnings": result.warnings,
            "per_image_alignment": True,
        }
        return MetricDepthArtifact(
            0, "depth_anything_v2_aligned", w, h, result.final_z_depth_m, ranges, valid,
            confidence, metadata, metrics, result.global_spline_z_depth_m,
            result.spatial_log_correction, result.extrapolation_mask, result.anchor_mask,
            result.anchor_residual_m, result.anchor_split, relative.astype(np.float32),
        )


def _confidence_from_support(valid: np.ndarray, support: np.ndarray, support_confidence: np.ndarray, median_relative_error: float | None, inliers: int, count: int) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for metric-depth confidence maps.") from exc
    support_u8 = np.asarray(support, dtype=np.uint8)
    distance = cv2.distanceTransform(1 - support_u8, cv2.DIST_L2, 3)
    scale = max(min(valid.shape) * 0.25, 1.0)
    decay = np.exp(-distance / scale)
    quality = max(0.0, 1.0 - float(median_relative_error or 0.0)) * min(float(inliers) / max(count, 1), 1.0)
    confidence = (quality * decay).astype(np.float32)
    confidence[support] = np.maximum(confidence[support], np.asarray(support_confidence)[support] * quality)
    confidence[~valid] = 0.0
    return np.clip(confidence, 0.0, 1.0)


def _alignment_progress(progress: object, stage: str, message: str) -> None:
    LOG.info(message)
    if callable(progress):
        progress(MetricDepthProgress(stage, message))
