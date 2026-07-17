from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from .models import (
    DensePointRecord,
    FramePairRecord,
    PairMatchRecord,
    TrackObservationRecord,
    TrackRecord,
    TriangulationConfig,
)
from .triangulation import triangulate_multiview_robust

ObservationKey = tuple[int, int]


def build_tracks_union_find(edges: list[tuple[ObservationKey, ObservationKey]]) -> list[set[ObservationKey]]:
    parent: dict[ObservationKey, ObservationKey] = {}
    members: dict[ObservationKey, set[ObservationKey]] = {}
    image_sets: dict[ObservationKey, set[int]] = {}

    def add(x: ObservationKey) -> None:
        if x not in parent:
            parent[x] = x
            members[x] = {x}
            image_sets[x] = {x[0]}

    def find(x: ObservationKey) -> ObservationKey:
        add(x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != x:
            nxt = parent[x]
            parent[x] = root
            x = nxt
        return root

    def union(a: ObservationKey, b: ObservationKey) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return True
        if image_sets[ra] & image_sets[rb]:
            return False
        if len(members[rb]) > len(members[ra]):
            ra, rb = rb, ra
        parent[rb] = ra
        members[ra] |= members.pop(rb)
        image_sets[ra] |= image_sets.pop(rb)
        return True

    for a, b in edges:
        union(a, b)
    comps: dict[ObservationKey, set[ObservationKey]] = defaultdict(set)
    for node in list(parent):
        comps[find(node)].add(node)
    return [component for component in comps.values() if len(component) >= 2]


def build_tracks_from_matches(
    pairs: list[FramePairRecord],
    matches_by_pair: dict[int, list[PairMatchRecord]],
    poses_by_image: dict[int, dict[str, Any]],
    camera_model: Any,
    config: TriangulationConfig,
) -> list[tuple[TrackRecord, list[TrackObservationRecord], DensePointRecord]]:
    pair_by_id = {int(pair.id): pair for pair in pairs if pair.id is not None}
    ranked_edges: list[tuple[tuple[float, float, int, int, int, int], ObservationKey, ObservationKey, PairMatchRecord]] = []
    for pair_id, matches in matches_by_pair.items():
        pair = pair_by_id.get(int(pair_id))
        if pair is None:
            continue
        for match in matches:
            if not match.is_epipolar_inlier:
                continue
            a = (pair.image_id_a, match.feature_idx_a)
            b = (pair.image_id_b, match.feature_idx_b)
            ranked_edges.append((_edge_sort_key(match, pair_id), a, b, match))

    # Ambiguous edges can compete for the same image inside one component. Feed
    # the union-find highest-confidence, lowest-epipolar-error edges first so a
    # weak match cannot block a stronger consistent track merely due to DB order.
    ranked_edges.sort(key=lambda item: item[0])
    edges = [(a, b) for _key, a, b, _match in ranked_edges]
    xy_by_obs: dict[ObservationKey, tuple[float, float]] = {}
    for _key, a, b, match in ranked_edges:
        xy_by_obs.setdefault(a, (match.x_a, match.y_a))
        xy_by_obs.setdefault(b, (match.x_b, match.y_b))

    components = build_tracks_union_find(edges)
    output: list[tuple[TrackRecord, list[TrackObservationRecord], DensePointRecord]] = []
    for component in components:
        observations_raw: list[tuple[int, int, float, float]] = []
        for image_id, feature_idx in sorted(component):
            coordinates = xy_by_obs.get((image_id, feature_idx))
            if coordinates is None:
                continue
            x, y = coordinates
            observations_raw.append((image_id, feature_idx, x, y))
        X, metrics, inlier_mask = triangulate_multiview_robust(
            observations_raw,
            poses_by_image,
            camera_model,
            config,
        )
        if X is None:
            continue
        inlier_observations = [
            observation
            for observation, is_inlier in zip(observations_raw, np.asarray(inlier_mask, dtype=bool))
            if bool(is_inlier)
        ]
        if len(inlier_observations) < config.min_observations:
            continue
        track = TrackRecord(
            status="active",
            num_observations=len(inlier_observations),
            num_images=len({obs[0] for obs in inlier_observations}),
            x=float(X[0]),
            y=float(X[1]),
            z=float(X[2]),
            mean_reprojection_error_px=metrics["mean_reprojection_error_px"],
            max_reprojection_error_px=metrics["max_reprojection_error_px"],
            min_triangulation_angle_deg=metrics["min_triangulation_angle_deg"],
        )
        observations = [
            TrackObservationRecord(track_id=0, image_id=image_id, feature_idx=feature_idx, x=float(x), y=float(y))
            for image_id, feature_idx, x, y in inlier_observations
        ]
        point = DensePointRecord(
            x=float(X[0]),
            y=float(X[1]),
            z=float(X[2]),
            mean_reprojection_error_px=track.mean_reprojection_error_px,
            max_reprojection_error_px=track.max_reprojection_error_px,
            num_observations=track.num_observations,
            source="triangulated",
            is_active=1,
        )
        output.append((track, observations, point))
    return output


def _edge_sort_key(match: PairMatchRecord, pair_id: int) -> tuple[float, float, int, int, int, int]:
    score = float(match.match_score) if match.match_score is not None and np.isfinite(match.match_score) else -np.inf
    epipolar_error = (
        float(match.epipolar_error)
        if match.epipolar_error is not None and np.isfinite(match.epipolar_error)
        else np.inf
    )
    return (
        -score,
        epipolar_error,
        int(pair_id),
        int(match.feature_idx_a),
        int(match.feature_idx_b),
        int(match.id or 0),
    )
