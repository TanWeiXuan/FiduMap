from __future__ import annotations

import numpy as np

from ..alignment import robust_affine_inverse_depth_alignment
from ..geometry import z_depth_to_range
from ..models import MetricDepthArtifact, MetricDepthMetrics
from .base import MetricDepthBackend


class DepthAnythingV2AlignedBackend(MetricDepthBackend):
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

    def infer(self, image_rgb: np.ndarray, prompt: object, camera_model: object, config: object, progress: object = None) -> MetricDepthArtifact:
        relative = self.predict_relative(image_rgb, config)
        result = robust_affine_inverse_depth_alignment(
            relative,
            prompt.depth_z_m,
            prompt.mask,
            prompt.confidence,
            int(getattr(config, "minimum_anchor_count")),
            int(getattr(config, "minimum_anchor_grid_cells")),
            float(getattr(config, "maximum_alignment_median_relative_error")),
        )
        ranges, ray_valid = z_depth_to_range(result.z_depth_m, camera_model)
        valid = result.valid_mask & ray_valid & bool(result.success)
        confidence = _confidence_from_support(
            valid,
            prompt.mask,
            prompt.confidence,
            result.median_relative_error,
            int(np.count_nonzero(result.inlier_mask)),
            max(prompt.pixel_count, 1),
        )
        metrics = MetricDepthMetrics(
            prompt_point_count=prompt.anchor_count,
            prompt_pixel_count=prompt.pixel_count,
            prompt_spatial_coverage=prompt.spatial_coverage,
            valid_output_fraction=float(np.mean(valid)),
            median_anchor_absolute_error_m=result.median_absolute_error_m,
            median_anchor_relative_error=result.median_relative_error,
            alignment_inlier_count=int(np.count_nonzero(result.inlier_mask)),
            status="success" if result.success else "failed",
            error_message=result.error_message,
        )
        h, w = relative.shape
        return MetricDepthArtifact(0, "depth_anything_v2_aligned", w, h, result.z_depth_m, ranges, valid, confidence, prompt.depth_z_m, prompt.mask, {"alignment_a": result.coefficient_a, "alignment_b": result.coefficient_b}, metrics)


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
