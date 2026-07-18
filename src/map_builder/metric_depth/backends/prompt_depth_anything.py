from __future__ import annotations

import numpy as np

from ..alignment import validate_anchor_coverage, verify_metric_prediction
from ..geometry import z_depth_to_range
from ..models import MetricDepthArtifact, MetricDepthMetrics
from .base import MetricDepthBackend
from .depth_anything_v2 import _confidence_from_support


class PromptDepthAnythingBackend(MetricDepthBackend):
    def load(self, config: object) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as exc:
            raise RuntimeError("missing optional metric-depth inference dependency; install src/map_builder/requirements-depth.txt") from exc
        reference = str(getattr(config, "model_id_or_path"))
        local_only = not bool(getattr(config, "allow_download"))
        try:
            self.processor = AutoImageProcessor.from_pretrained(reference, local_files_only=local_only)
            self.model = AutoModelForDepthEstimation.from_pretrained(reference, local_files_only=local_only)
        except Exception as exc:
            mode = "local cache/directory" if local_only else "configured model reference"
            raise RuntimeError(f"Prompt Depth Anything model unavailable from {mode}: {reference}: {exc}") from exc
        requested = str(getattr(config, "device"))
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was selected but torch.cuda.is_available() is false.")
        self.device = "cuda" if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()) else "cpu"
        self.model.to(self.device).eval()

    def predict_metric(self, image_rgb: np.ndarray, prompt_depth: np.ndarray, prompt_mask: np.ndarray, config: object) -> np.ndarray:
        import torch
        from PIL import Image

        assert self.processor is not None and self.model is not None
        image = Image.fromarray(np.asarray(image_rgb, dtype=np.uint8), mode="RGB")
        size = int(getattr(config, "inference_size"))
        adapted_depth, _adapted_mask = adapt_sparse_prompt(prompt_depth, prompt_mask, (size, size))
        inputs = self.processor(
            images=image,
            prompt_depth=adapted_depth,
            prompt_scale_to_meter=1.0,
            size={"height": size, "width": size},
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        processed = self.processor.post_process_depth_estimation(outputs, target_sizes=[(image.height, image.width)])
        return processed[0]["predicted_depth"].detach().float().cpu().numpy().astype(np.float32)

    def infer(self, image_rgb: np.ndarray, prompt: object, camera_model: object, config: object, progress: object = None) -> MetricDepthArtifact:
        coverage_ok, _cells, coverage_error = validate_anchor_coverage(prompt.mask, int(getattr(config, "minimum_anchor_count")), int(getattr(config, "minimum_anchor_grid_cells")))
        if coverage_ok:
            z = self.predict_metric(image_rgb, prompt.depth_z_m, prompt.mask, config)
            verified, median_abs, median_rel, inliers, error = verify_metric_prediction(z, prompt.depth_z_m, prompt.mask, float(getattr(config, "maximum_anchor_error_m")), float(getattr(config, "maximum_alignment_median_relative_error")))
            positive_fraction = float(np.mean(np.isfinite(z) & (z > 0.0)))
            if verified and positive_fraction < 0.01:
                verified, error = False, f"too few finite positive output values: {positive_fraction:.3%}"
        else:
            z = np.zeros_like(prompt.depth_z_m, dtype=np.float32)
            verified, median_abs, median_rel, inliers, error = False, None, None, 0, coverage_error
        ranges, ray_valid = z_depth_to_range(z, camera_model)
        valid = np.isfinite(z) & (z > 0.0) & ray_valid & bool(verified)
        confidence = _confidence_from_support(valid, prompt.mask, prompt.confidence, median_rel, inliers, max(prompt.pixel_count, 1))
        metrics = MetricDepthMetrics(prompt.anchor_count, prompt.pixel_count, prompt.spatial_coverage, float(np.mean(valid)), median_abs, median_rel, inliers, status="success" if verified else "failed", error_message=error)
        h, w = z.shape
        return MetricDepthArtifact(0, "prompt_depth_anything", w, h, z, ranges, valid, confidence, prompt.depth_z_m, prompt.mask, {}, metrics)


def adapt_sparse_prompt(depth: np.ndarray, mask: np.ndarray, target_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Nearest valid-aware resize that never blends valid depth with missing zeros."""
    source_depth = np.asarray(depth, dtype=np.float32)
    source_mask = np.asarray(mask, dtype=bool)
    target_h, target_w = target_shape
    if source_depth.shape == (target_h, target_w):
        out = source_depth.copy()
        out[~source_mask] = 0.0
        return out, source_mask.copy()
    source_h, source_w = source_depth.shape
    if target_h <= source_h and target_w <= source_w:
        # Sparse valid-aware median aggregation. Iterate over support only, so
        # missing background is neither averaged nor allowed to erase prompts.
        buckets: dict[tuple[int, int], list[float]] = {}
        ys, xs = np.nonzero(source_mask & np.isfinite(source_depth) & (source_depth > 0.0))
        target_y = np.minimum(((ys + 0.5) * target_h / source_h).astype(int), target_h - 1)
        target_x = np.minimum(((xs + 0.5) * target_w / source_w).astype(int), target_w - 1)
        for y, x, value in zip(target_y, target_x, source_depth[ys, xs]):
            buckets.setdefault((int(y), int(x)), []).append(float(value))
        out_depth = np.zeros((target_h, target_w), dtype=np.float32)
        out_mask = np.zeros((target_h, target_w), dtype=bool)
        for (y, x), values in buckets.items():
            out_depth[y, x] = np.float32(np.median(values)); out_mask[y, x] = True
        return out_depth, out_mask
    y = np.minimum((np.arange(target_h) * source_h / target_h).astype(int), source_h - 1)
    x = np.minimum((np.arange(target_w) * source_w / target_w).astype(int), source_w - 1)
    out_mask = source_mask[np.ix_(y, x)]
    out_depth = np.where(out_mask, source_depth[np.ix_(y, x)], 0.0).astype(np.float32)
    return out_depth, out_mask
