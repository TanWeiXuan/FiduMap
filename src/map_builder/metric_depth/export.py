from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from .models import MetricDepthArtifact


def export_artifact(artifact: MetricDepthArtifact, output_folder: Path, stem: str | None = None) -> dict[str, Path]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for portable metric-depth export.") from exc
    folder = Path(output_folder)
    folder.mkdir(parents=True, exist_ok=True)
    prefix = stem or str(artifact.image_id)
    npz_path = folder / f"{prefix}.npz"
    np.savez_compressed(
        npz_path,
        z_depth_m=np.asarray(artifact.z_depth_m, dtype=np.float32),
        range_m=np.asarray(artifact.range_m, dtype=np.float32),
        valid_mask=np.asarray(artifact.valid_mask, dtype=np.uint8),
        confidence=np.asarray(artifact.confidence, dtype=np.float32),
        prompt_depth_z_m=np.asarray(artifact.prompt_depth_z_m, dtype=np.float32),
        prompt_mask=np.asarray(artifact.prompt_mask, dtype=np.uint8),
    )
    valid = np.asarray(artifact.valid_mask, dtype=bool) & np.isfinite(artifact.range_m) & (artifact.range_m > 0.0)
    overflow = valid & (artifact.range_m * 1000.0 > np.iinfo(np.uint16).max)
    range_mm = np.zeros(artifact.range_m.shape, dtype=np.uint16)
    representable = valid & ~overflow
    range_mm[representable] = np.rint(artifact.range_m[representable] * 1000.0).astype(np.uint16)
    confidence = np.clip(np.asarray(artifact.confidence) * 255.0, 0, 255).astype(np.uint8)
    prompt_mask = np.asarray(artifact.prompt_mask, dtype=np.uint8) * 255
    range_path = folder / f"{prefix}_range_mm.png"
    confidence_path = folder / f"{prefix}_confidence.png"
    prompt_path = folder / f"{prefix}_prompt_mask.png"
    for path, image in ((range_path, range_mm), (confidence_path, confidence), (prompt_path, prompt_mask)):
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Could not write portable depth image: {path}")
    metadata = dict(artifact.metadata)
    metadata.update({
        "image_id": artifact.image_id,
        "backend": artifact.backend,
        "width": artifact.width,
        "height": artifact.height,
        "quality_metrics": artifact.metrics.to_dict(),
        "portable_range_mm_overflow": bool(np.any(overflow)),
        "portable_range_mm_overflow_pixel_count": int(np.count_nonzero(overflow)),
        "portable_range_mm_zero_means_invalid": True,
    })
    json_path = folder / f"{prefix}.json"
    json_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return {"npz": npz_path, "metadata": json_path, "range_mm": range_path, "confidence": confidence_path, "prompt_mask": prompt_path}


def export_stored_artifact(store: Any, run_id: int, image_id: int, output_folder: Path, stem: str | None = None) -> dict[str, Path]:
    artifact = store.load_artifact(run_id, image_id)
    if artifact is None:
        raise RuntimeError(f"No successful metric-depth artifact for image {image_id} in run {run_id}.")
    return export_artifact(artifact, output_folder, stem)
