from __future__ import annotations

from collections import defaultdict
from typing import Any

from .models import DensePointRecord, FramePairRecord, PairMatchRecord, TrackObservationRecord, TrackRecord
from .triangulation import triangulate_multiview

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
    config: Any,
) -> list[tuple[TrackRecord, list[TrackObservationRecord], DensePointRecord]]:
    pair_by_id = {int(pair.id): pair for pair in pairs if pair.id is not None}
    edges: list[tuple[ObservationKey, ObservationKey]] = []
    xy_by_obs: dict[ObservationKey, tuple[float, float]] = {}
    for pair_id, matches in matches_by_pair.items():
        pair = pair_by_id.get(int(pair_id))
        if pair is None:
            continue
        for match in matches:
            if not match.is_epipolar_inlier:
                continue
            a = (pair.image_id_a, match.feature_idx_a)
            b = (pair.image_id_b, match.feature_idx_b)
            edges.append((a, b))
            xy_by_obs.setdefault(a, (match.x_a, match.y_a))
            xy_by_obs.setdefault(b, (match.x_b, match.y_b))
    components = build_tracks_union_find(edges)
    output: list[tuple[TrackRecord, list[TrackObservationRecord], DensePointRecord]] = []
    for component in components:
        observations_raw = []
        for image_id, feature_idx in sorted(component):
            if (image_id, feature_idx) not in xy_by_obs:
                continue
            x, y = xy_by_obs[(image_id, feature_idx)]
            observations_raw.append((image_id, feature_idx, x, y))
        X, metrics = triangulate_multiview(observations_raw, poses_by_image, camera_model, config)
        if X is None:
            continue
        track = TrackRecord(
            status="active",
            num_observations=len(observations_raw),
            num_images=len({obs[0] for obs in observations_raw}),
            x=float(X[0]),
            y=float(X[1]),
            z=float(X[2]),
            mean_reprojection_error_px=metrics["mean_reprojection_error_px"],
            max_reprojection_error_px=metrics["max_reprojection_error_px"],
            min_triangulation_angle_deg=metrics["min_triangulation_angle_deg"],
        )
        observations = [
            TrackObservationRecord(track_id=0, image_id=image_id, feature_idx=feature_idx, x=float(x), y=float(y))
            for image_id, feature_idx, x, y in observations_raw
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
