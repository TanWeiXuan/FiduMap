from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from map_builder.dense_reconstruction.models import FramePairRecord, PairSelectionConfig
from map_builder.project.models import MarkerDetection, OptimizedCameraPose


def _camera_center_and_axis(T_W_C: dict) -> tuple[np.ndarray, np.ndarray]:
    R = np.asarray(T_W_C["R"], dtype=float)
    t = np.asarray(T_W_C["t"], dtype=float)
    z = R @ np.array([0.0, 0.0, 1.0])
    return t, z / max(np.linalg.norm(z), 1e-12)


def _count_common_markers(a: Iterable[MarkerDetection], b: Iterable[MarkerDetection]) -> int:
    am = {d.marker_id for d in a}
    bm = {d.marker_id for d in b}
    return len(am & bm)


def select_frame_pairs(
    camera_poses: list[OptimizedCameraPose],
    detections_by_image: dict[int, list[MarkerDetection]],
    config: PairSelectionConfig,
) -> list[FramePairRecord]:
    per_image: dict[int, list[tuple[float, FramePairRecord]]] = {}

    for i, pa in enumerate(camera_poses):
        C1, z1 = _camera_center_and_axis(pa.T_W_C)
        for pb in camera_poses[i + 1 :]:
            C2, z2 = _camera_center_and_axis(pb.T_W_C)
            baseline = float(np.linalg.norm(C2 - C1))
            if baseline < config.min_baseline_m or baseline > config.max_baseline_m:
                continue
            dot = float(np.clip(np.dot(z1, z2), -1.0, 1.0))
            angle_deg = math.degrees(math.acos(dot))
            if angle_deg > config.max_optical_axis_angle_deg:
                continue
            common = _count_common_markers(detections_by_image.get(pa.image_id, []), detections_by_image.get(pb.image_id, []))
            if common < config.min_common_markers:
                continue
            overlap = 1.0 / (1.0 + baseline) + (0.2 * common if config.use_common_markers_bonus else 0.0)
            rec = FramePairRecord(
                image_id_a=min(pa.image_id, pb.image_id),
                image_id_b=max(pa.image_id, pb.image_id),
                status="candidate",
                baseline_m=baseline,
                optical_axis_angle_deg=angle_deg,
                common_marker_count=common,
                estimated_overlap_score=float(overlap),
            )
            per_image.setdefault(pa.image_id, []).append((overlap, rec))
            per_image.setdefault(pb.image_id, []).append((overlap, rec))

    keep: set[tuple[int, int]] = set()
    for image_id, ranked in per_image.items():
        del image_id
        ranked.sort(key=lambda x: x[0], reverse=True)
        for _, rec in ranked[: config.max_pairs_per_image]:
            keep.add((rec.image_id_a, rec.image_id_b))

    out: list[FramePairRecord] = []
    for ranked in per_image.values():
        for _, rec in ranked:
            key = (rec.image_id_a, rec.image_id_b)
            if key in keep:
                keep.remove(key)
                out.append(rec)
    out.sort(key=lambda r: (r.image_id_a, r.image_id_b))
    return out
