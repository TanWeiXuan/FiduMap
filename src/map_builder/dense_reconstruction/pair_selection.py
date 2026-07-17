from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from map_builder.dense_reconstruction.models import FramePairRecord, PairSelectionConfig
from map_builder.project.models import MarkerDetection, OptimizedCameraPose


def _camera_center_and_axis(T_W_C: dict) -> tuple[np.ndarray, np.ndarray]:
    R = np.asarray(T_W_C["R"], dtype=float)
    t = np.asarray(T_W_C["t"], dtype=float).reshape(3)
    z = R @ np.array([0.0, 0.0, 1.0])
    return t, z / max(np.linalg.norm(z), 1e-12)


def _marker_ids(detections: Iterable[MarkerDetection]) -> set[int]:
    return {int(d.marker_id) for d in detections}


def _baseline_quality(baseline: float, config: PairSelectionConfig) -> float:
    lower = max(float(config.min_baseline_m), 1e-6)
    upper = max(float(config.max_baseline_m), lower)
    preferred = math.sqrt(lower * upper)
    return float(min(baseline / preferred, preferred / baseline))


def _pair_score(
    baseline: float,
    optical_axis_angle_deg: float,
    marker_ids_a: set[int],
    marker_ids_b: set[int],
    config: PairSelectionConfig,
) -> tuple[float, int]:
    common = len(marker_ids_a & marker_ids_b)
    union = len(marker_ids_a | marker_ids_b)
    marker_overlap = common / union if union else 0.0
    baseline_quality = _baseline_quality(baseline, config)
    max_angle = max(float(config.max_optical_axis_angle_deg), 1e-6)
    orientation_quality = max(0.0, 1.0 - optical_axis_angle_deg / max_angle)

    score = 0.55 * marker_overlap + 0.30 * baseline_quality + 0.15 * orientation_quality
    if config.use_common_markers_bonus and common == 0:
        # Marker co-visibility is the only overlap signal currently available.
        # Keep no-common-marker pairs possible when explicitly allowed, but rank
        # them well below pairs with direct evidence of shared scene content.
        score *= 0.25
    return float(score), common


def select_frame_pairs(
    camera_poses: list[OptimizedCameraPose],
    detections_by_image: dict[int, list[MarkerDetection]],
    config: PairSelectionConfig,
) -> list[FramePairRecord]:
    if config.max_pairs_per_image <= 0:
        return []

    candidates: list[tuple[float, FramePairRecord]] = []
    marker_ids_by_image = {
        int(pose.image_id): _marker_ids(detections_by_image.get(pose.image_id, []))
        for pose in camera_poses
    }

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

            score, common = _pair_score(
                baseline,
                angle_deg,
                marker_ids_by_image.get(int(pa.image_id), set()),
                marker_ids_by_image.get(int(pb.image_id), set()),
                config,
            )
            if common < config.min_common_markers:
                continue
            candidates.append(
                (
                    score,
                    FramePairRecord(
                        image_id_a=min(pa.image_id, pb.image_id),
                        image_id_b=max(pa.image_id, pb.image_id),
                        status="candidate",
                        baseline_m=baseline,
                        optical_axis_angle_deg=angle_deg,
                        common_marker_count=common,
                        estimated_overlap_score=score,
                    ),
                )
            )

    # A single global greedy pass enforces the configured degree cap. The old
    # per-image top-k union could exceed max_pairs_per_image substantially.
    candidates.sort(
        key=lambda item: (
            -item[0],
            item[1].image_id_a,
            item[1].image_id_b,
        )
    )
    degree: dict[int, int] = {int(pose.image_id): 0 for pose in camera_poses}
    selected: list[FramePairRecord] = []
    for _score, rec in candidates:
        if degree[rec.image_id_a] >= config.max_pairs_per_image:
            continue
        if degree[rec.image_id_b] >= config.max_pairs_per_image:
            continue
        selected.append(rec)
        degree[rec.image_id_a] += 1
        degree[rec.image_id_b] += 1

    selected.sort(key=lambda rec: (rec.image_id_a, rec.image_id_b))
    return selected
