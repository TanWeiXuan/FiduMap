from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from .models import DenseFeatureRecord, MatchingConfig, PairMatchRecord
from .vendor_xfeat import ensure_vendor_parent_on_path


class XFeatLighterGlueMatcher:
    def __init__(self, config: MatchingConfig):
        self.config = config
        self.torch = importlib.import_module("torch")
        importlib.import_module("kornia")
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
        if features_a.keypoints is None or features_b.keypoints is None:
            return []
        if features_a.descriptors is None or features_b.descriptors is None:
            return []
        d0 = self._feature_dict(features_a)
        d1 = self._feature_dict(features_b)
        with self.torch.no_grad():
            _mk0, _mk1, idx = self.model.match_lighterglue(d0, d1, min_conf=float(self.config.min_match_score))
        matches: list[PairMatchRecord] = []
        k0 = np.asarray(features_a.keypoints, dtype=float)
        k1 = np.asarray(features_b.keypoints, dtype=float)
        for ia, ib in np.asarray(idx, dtype=int):
            x_a, y_a = k0[ia, :2]
            x_b, y_b = k1[ib, :2]
            matches.append(
                PairMatchRecord(
                    pair_id=int(pair_id),
                    feature_idx_a=int(ia),
                    feature_idx_b=int(ib),
                    x_a=float(x_a),
                    y_a=float(y_a),
                    x_b=float(x_b),
                    y_b=float(y_b),
                    match_score=None,
                )
            )
        return matches

    def _feature_dict(self, rec: DenseFeatureRecord) -> dict[str, Any]:
        keypoints = self.torch.as_tensor(np.asarray(rec.keypoints, dtype=np.float32), device=self.model.dev)
        descriptors = self.torch.as_tensor(np.asarray(rec.descriptors, dtype=np.float32), device=self.model.dev)
        scores_np = np.ones((keypoints.shape[0],), dtype=np.float32) if rec.scores is None else np.asarray(rec.scores, dtype=np.float32)
        scores = self.torch.as_tensor(scores_np, device=self.model.dev)
        width = rec.width or int(np.ceil(float(np.max(np.asarray(rec.keypoints)[:, 0])) + 1.0))
        height = rec.height or int(np.ceil(float(np.max(np.asarray(rec.keypoints)[:, 1])) + 1.0))
        return {
            "keypoints": keypoints,
            "descriptors": descriptors,
            "scores": scores,
            "image_size": (int(width), int(height)),
        }


def _resolve_device(torch: Any, requested: str) -> Any:
    normalized = requested.strip().lower()
    if normalized in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
