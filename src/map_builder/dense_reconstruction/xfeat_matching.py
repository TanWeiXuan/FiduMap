from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from .models import DenseFeatureRecord, MatchingConfig, PairMatchRecord
from .vendor_xfeat import ensure_vendor_parent_on_path

SEMI_DENSE_EXTRACTION_MODE = "semi_dense_xfeat"
SEMI_DENSE_DESCRIPTOR_SOURCE = "detectAndComputeDense"


class IncompatibleDenseFeatureError(RuntimeError):
    """Raised when stored dense features cannot be matched by the semi-dense XFeat path."""


class XFeatSemiDenseMatcher:
    def __init__(self, config: MatchingConfig):
        self.config = config
        self.torch = importlib.import_module("torch")
        ensure_vendor_parent_on_path()
        xfeat_mod = importlib.import_module("vendor.xfeat.xfeat")
        self.model = xfeat_mod.XFeat()
        device = _resolve_device(self.torch, config.device)
        self.model.dev = device
        self.model.net.to(device).eval()

    def match(
        self,
        features_a: DenseFeatureRecord,
        features_b: DenseFeatureRecord,
        pair_id: int,
    ) -> list[PairMatchRecord]:
        if int(pair_id) <= 0:
            raise ValueError("pair_id must be the persisted frame_pairs.id for this match set.")
        _validate_compatible_feature(features_a, "a")
        _validate_compatible_feature(features_b, "b")

        keypoints_a, descriptors_a, scales_a = self._feature_tensors(features_a)
        keypoints_b, descriptors_b, scales_b = self._feature_tensors(features_b)
        min_cossim = float(self.config.min_match_score)

        with self.torch.no_grad():
            matched_indices = self.model.batch_match(
                descriptors_a[None, ...],
                descriptors_b[None, ...],
                min_cossim=min_cossim,
            )
            refined = self._try_refine_matches(
                keypoints_a,
                descriptors_a,
                scales_a,
                keypoints_b,
                descriptors_b,
                scales_b,
                matched_indices,
            )

        idx_a = _to_numpy_indices(matched_indices[0][0])
        idx_b = _to_numpy_indices(matched_indices[0][1])
        if len(idx_a) != len(idx_b):
            raise RuntimeError("XFeat semi-dense matching returned misaligned match index arrays.")

        kpts_a = np.asarray(features_a.keypoints, dtype=np.float32)
        kpts_b = np.asarray(features_b.keypoints, dtype=np.float32)
        coords = _to_numpy(refined) if refined is not None else None
        if coords is not None and coords.shape != (len(idx_a), 4):
            coords = None

        scores = self._match_scores(descriptors_a, descriptors_b, idx_a, idx_b)
        rows: list[PairMatchRecord] = []
        for row_index, (ia, ib) in enumerate(zip(idx_a, idx_b)):
            if coords is None:
                x_a, y_a = kpts_a[int(ia), :2]
                x_b, y_b = kpts_b[int(ib), :2]
            else:
                x_a, y_a, x_b, y_b = coords[row_index]
            rows.append(
                PairMatchRecord(
                    pair_id=int(pair_id),
                    feature_idx_a=int(ia),
                    feature_idx_b=int(ib),
                    x_a=float(x_a),
                    y_a=float(y_a),
                    x_b=float(x_b),
                    y_b=float(y_b),
                    match_score=None if scores is None else float(scores[row_index]),
                )
            )
        return rows

    def _feature_tensors(self, rec: DenseFeatureRecord) -> tuple[Any, Any, Any]:
        keypoints = self.torch.as_tensor(np.asarray(rec.keypoints, dtype=np.float32), device=self.model.dev)
        descriptors = self.torch.as_tensor(np.asarray(rec.descriptors, dtype=np.float32), device=self.model.dev)
        scales_np = np.ones((len(np.asarray(rec.keypoints)),), dtype=np.float32)
        if rec.scores is not None:
            scales_np = np.asarray(rec.scores, dtype=np.float32)
        scales = self.torch.as_tensor(scales_np, device=self.model.dev)
        return keypoints, descriptors, scales

    def _try_refine_matches(
        self,
        keypoints_a: Any,
        descriptors_a: Any,
        scales_a: Any,
        keypoints_b: Any,
        descriptors_b: Any,
        scales_b: Any,
        matched_indices: list[tuple[Any, Any]],
    ) -> Any | None:
        if not hasattr(self.model, "refine_matches"):
            return None
        d0 = {"keypoints": keypoints_a[None, ...], "descriptors": descriptors_a[None, ...], "scales": scales_a[None, ...]}
        d1 = {"keypoints": keypoints_b[None, ...], "descriptors": descriptors_b[None, ...], "scales": scales_b[None, ...]}
        try:
            return self.model.refine_matches(d0, d1, matches=matched_indices, batch_idx=0)
        except Exception:
            return None

    def _match_scores(self, descriptors_a: Any, descriptors_b: Any, idx_a: np.ndarray, idx_b: np.ndarray) -> np.ndarray | None:
        if len(idx_a) == 0:
            return np.empty((0,), dtype=np.float32)
        try:
            idx_a_tensor = self.torch.as_tensor(idx_a, device=self.model.dev)
            idx_b_tensor = self.torch.as_tensor(idx_b, device=self.model.dev)
            selected_a = descriptors_a[idx_a_tensor]
            selected_b = descriptors_b[idx_b_tensor]
            scores = self.torch.sum(selected_a * selected_b, dim=1)
            return np.asarray(_to_numpy(scores), dtype=np.float32)
        except Exception:
            return None


def _validate_compatible_feature(feature: DenseFeatureRecord, label: str) -> None:
    prefix = f"Feature record {label}"
    if feature.extraction_mode != SEMI_DENSE_EXTRACTION_MODE or feature.descriptor_source != SEMI_DENSE_DESCRIPTOR_SOURCE:
        raise IncompatibleDenseFeatureError(
            "Features were extracted by an older/incompatible dense feature format. Re-run feature extraction."
        )
    if feature.status != "success":
        raise IncompatibleDenseFeatureError(f"{prefix} is not a successful dense feature record.")
    if feature.keypoints is None or feature.descriptors is None:
        raise IncompatibleDenseFeatureError(f"{prefix} is missing keypoints or descriptors.")
    keypoints = np.asarray(feature.keypoints)
    descriptors = np.asarray(feature.descriptors)
    if keypoints.ndim != 2 or keypoints.shape[1] < 2:
        raise IncompatibleDenseFeatureError(f"{prefix} has invalid keypoint shape {keypoints.shape}; expected Nx2.")
    if descriptors.ndim != 2:
        raise IncompatibleDenseFeatureError(f"{prefix} has invalid descriptor shape {descriptors.shape}; expected NxD.")
    if int(feature.num_keypoints) <= 0 or len(keypoints) == 0:
        raise IncompatibleDenseFeatureError(f"{prefix} has no semi-dense keypoints.")
    if len(keypoints) != len(descriptors):
        raise IncompatibleDenseFeatureError(
            f"{prefix} has {len(keypoints)} keypoints but {len(descriptors)} descriptors."
        )
    if feature.scores is not None and len(np.asarray(feature.scores)) != len(keypoints):
        raise IncompatibleDenseFeatureError(f"{prefix} has score/scale values that do not align with keypoints.")


def _resolve_device(torch: Any, requested: str) -> Any:
    normalized = requested.strip().lower()
    if normalized in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_numpy_indices(value: Any) -> np.ndarray:
    return np.asarray(_to_numpy(value), dtype=np.int64).reshape(-1)
