from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from .models import DenseFeatureRecord, XFeatExtractionConfig
from .vendor_xfeat import ensure_vendor_parent_on_path


class XFeatSemiDenseExtractor:
    def __init__(self, config: XFeatExtractionConfig):
        self.config = config
        torch = importlib.import_module("torch")
        ensure_vendor_parent_on_path()
        xfeat_mod = importlib.import_module("vendor.xfeat.xfeat")
        self.torch = torch
        self.model = xfeat_mod.XFeat(top_k=int(config.max_keypoints), detection_threshold=float(config.detection_threshold))
        device = _resolve_device(torch, config.device, config.use_cuda)
        self.model.dev = device
        self.model.net.to(device).eval()

    def extract(self, image_bgr_or_gray: np.ndarray) -> DenseFeatureRecord:
        image, scale = _prepare_image(image_bgr_or_gray, self.config.resize_max_side)
        tensor = _to_xfeat_tensor(self.torch, image, self.model.dev)
        with self.torch.no_grad():
            out = self.model.detectAndComputeDense(tensor, top_k=int(self.config.max_keypoints), multiscale=True)
        keypoints = _to_numpy(out["keypoints"])
        descriptors = _to_numpy(out["descriptors"])
        scores = _to_numpy(out.get("scores", out.get("scales")))
        if keypoints.ndim == 3:
            keypoints = keypoints[0]
        if descriptors.ndim == 3:
            descriptors = descriptors[0]
        if scores is not None and scores.ndim == 2:
            scores = scores[0]
        if scale != 1.0:
            keypoints = keypoints / scale
        return DenseFeatureRecord(
            image_id=0,
            keypoints=keypoints.astype(np.float32, copy=False),
            descriptors=descriptors.astype(np.float32, copy=False),
            scores=None if scores is None else scores.astype(np.float32, copy=False),
            status="success",
            num_keypoints=int(len(keypoints)),
            height=int(image_bgr_or_gray.shape[0]),
            width=int(image_bgr_or_gray.shape[1]),
        )


def _resolve_device(torch: Any, requested: str, use_cuda: bool) -> Any:
    normalized = requested.strip().lower()
    cuda_ok = bool(use_cuda and torch.cuda.is_available())
    if normalized == "cuda" and cuda_ok:
        return torch.device("cuda")
    if normalized == "auto" and cuda_ok:
        return torch.device("cuda")
    return torch.device("cpu")


def _prepare_image(image: np.ndarray, resize_max_side: int) -> tuple[np.ndarray, float]:
    arr = np.asarray(image)
    if arr.ndim not in {2, 3}:
        raise ValueError(f"Expected grayscale or color image, got shape {arr.shape}.")
    if arr.ndim == 3 and arr.shape[2] == 3:
        try:
            import cv2  # type: ignore[import-not-found]

            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        except Exception:
            arr = arr[..., ::-1]
    max_side = int(resize_max_side)
    scale = 1.0
    if max_side > 0 and max(arr.shape[:2]) > max_side:
        try:
            import cv2  # type: ignore[import-not-found]

            scale = max_side / float(max(arr.shape[:2]))
            size = (max(int(round(arr.shape[1] * scale)), 1), max(int(round(arr.shape[0] * scale)), 1))
            arr = cv2.resize(arr, size, interpolation=cv2.INTER_AREA)
        except ImportError as exc:
            raise RuntimeError("OpenCV is required when resize_max_side downsizes images.") from exc
    return np.ascontiguousarray(arr), scale


def _to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_xfeat_tensor(torch: Any, image: np.ndarray, device: Any) -> Any:
    arr = np.asarray(image)
    if arr.ndim == 2:
        tensor = torch.as_tensor(arr[None, None, ...], device=device)
    elif arr.ndim == 3:
        tensor = torch.as_tensor(arr, device=device).permute(2, 0, 1)[None]
    else:
        raise ValueError(f"Expected grayscale or color image, got shape {arr.shape}.")
    return tensor.float() / 255.0
