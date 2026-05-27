from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from .models import DensePointRecord, DuplicateMergeConfig, TrackObservationRecord


def merge_duplicate_points(
    points: list[DensePointRecord],
    observations_by_track: dict[int, list[TrackObservationRecord]] | None,
    config: DuplicateMergeConfig,
) -> tuple[list[DensePointRecord], int]:
    if len(points) < 2:
        return points, 0
    coords = np.array([[p.x, p.y, p.z] for p in points], dtype=float)
    candidates = _candidate_pairs(coords, config.duplicate_merge_radius_m)
    parent = list(range(len(points)))
    obs_by_track = observations_by_track or {}

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def component_indices(root: int) -> list[int]:
        return [i for i in range(len(points)) if find(i) == root]

    accepted = 0
    for i, j in candidates:
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        merged_indices = component_indices(ri) + component_indices(rj)
        if not _can_merge([points[k] for k in merged_indices], obs_by_track, config):
            continue
        parent[rj] = ri
        accepted += 1

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(points)):
        groups[find(i)].append(i)
    merged: list[DensePointRecord] = []
    for indices in groups.values():
        selected = [points[i] for i in indices]
        weights = np.array([max(float(p.num_observations or 1), 1.0) for p in selected], dtype=float)
        xyz = np.average(np.array([[p.x, p.y, p.z] for p in selected], dtype=float), axis=0, weights=weights)
        track_id = selected[0].track_id
        num_obs = sum(int(p.num_observations or 0) for p in selected) or None
        mean_err = _weighted_optional([p.mean_reprojection_error_px for p in selected], weights)
        max_err = max([p.max_reprojection_error_px or 0.0 for p in selected]) if selected else None
        merged.append(
            DensePointRecord(
                track_id=track_id,
                x=float(xyz[0]),
                y=float(xyz[1]),
                z=float(xyz[2]),
                mean_reprojection_error_px=mean_err,
                max_reprojection_error_px=max_err,
                num_observations=num_obs,
                source="merged" if len(indices) > 1 else selected[0].source,
                is_active=1,
            )
        )
    return merged, accepted


def points_from_rows(rows: list[Any]) -> list[DensePointRecord]:
    return [
        DensePointRecord(
            id=int(row["id"]),
            track_id=None if row["track_id"] is None else int(row["track_id"]),
            x=float(row["x"]),
            y=float(row["y"]),
            z=float(row["z"]),
            mean_reprojection_error_px=None
            if row["mean_reprojection_error_px"] is None
            else float(row["mean_reprojection_error_px"]),
            max_reprojection_error_px=None
            if row["max_reprojection_error_px"] is None
            else float(row["max_reprojection_error_px"]),
            num_observations=None if row["num_observations"] is None else int(row["num_observations"]),
            source=str(row["source"]),
            is_active=int(row["is_active"]),
        )
        for row in rows
    ]


def _candidate_pairs(coords: np.ndarray, radius: float) -> list[tuple[int, int]]:
    try:
        from scipy.spatial import cKDTree  # type: ignore[import-not-found]

        return [(int(i), int(j)) for i, j in cKDTree(coords).query_pairs(float(radius))]
    except Exception:
        return _grid_candidate_pairs(coords, float(radius))


def _grid_candidate_pairs(coords: np.ndarray, radius: float) -> list[tuple[int, int]]:
    if radius <= 0.0:
        return []
    buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    pairs: set[tuple[int, int]] = set()
    for i, xyz in enumerate(coords):
        cell = tuple(np.floor(xyz / radius).astype(int))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for j in buckets.get((cell[0] + dx, cell[1] + dy, cell[2] + dz), []):
                        if np.linalg.norm(coords[i] - coords[j]) <= radius:
                            pairs.add((min(i, j), max(i, j)))
        buckets[cell].append(i)
    return sorted(pairs)


def _can_merge(
    points: list[DensePointRecord],
    observations_by_track: dict[int, list[TrackObservationRecord]],
    config: DuplicateMergeConfig,
) -> bool:
    seen_images: set[int] = set()
    for point in points:
        if point.mean_reprojection_error_px is not None and point.mean_reprojection_error_px > config.max_merged_mean_reprojection_error_px:
            return False
        if point.max_reprojection_error_px is not None and point.max_reprojection_error_px > config.max_merged_reprojection_error_px:
            return False
        if point.track_id is None:
            continue
        for obs in observations_by_track.get(point.track_id, []):
            if obs.image_id in seen_images:
                return False
            seen_images.add(obs.image_id)
    return True


def _weighted_optional(values: list[float | None], weights: np.ndarray) -> float | None:
    valid = np.array([v is not None for v in values], dtype=bool)
    if not np.any(valid):
        return None
    vals = np.array([0.0 if v is None else float(v) for v in values], dtype=float)
    return float(np.average(vals[valid], weights=weights[valid]))
