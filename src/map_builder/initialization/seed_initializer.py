"""Seed pose initialization by graph traversal."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from map_builder.geometry import SE3
from map_builder.project import ProjectStore, SeedCameraPose, SeedMarkerPose

from .observation_graph import ObservationGraph, build_graph_from_store


@dataclass(frozen=True)
class SeedInitializationResult:
    camera_poses: list[SeedCameraPose]
    marker_poses: list[SeedMarkerPose]
    disconnected_markers: list[int]


def initialize_seed_poses(graph: ObservationGraph, anchor_marker_id: int = 0) -> SeedInitializationResult:
    if anchor_marker_id not in graph.marker_to_edges:
        return SeedInitializationResult([], [], sorted(graph.marker_to_edges.keys()))

    marker_poses: dict[int, tuple[SE3, int | None, float, bool]] = {
        anchor_marker_id: (SE3.identity(), None, 0.0, True)
    }
    camera_poses: dict[int, tuple[SE3, int | None, float]] = {}
    queue: deque[tuple[str, int]] = deque([("marker", anchor_marker_id)])

    while queue:
        kind, node_id = queue.popleft()
        if kind == "marker":
            T_W_M = marker_poses[node_id][0]
            for edge in graph.marker_to_edges.get(node_id, []):
                candidate = T_W_M @ edge.T_C_M.inverse()
                score = _score(edge.reprojection_error_px)
                if _is_better_camera_candidate(camera_poses, edge.image_id, score):
                    camera_poses[edge.image_id] = (candidate, node_id, score)
                    queue.append(("camera", edge.image_id))
        else:
            T_W_C = camera_poses[node_id][0]
            for edge in graph.camera_to_edges.get(node_id, []):
                candidate = T_W_C @ edge.T_C_M
                score = _score(edge.reprojection_error_px)
                if _is_better_marker_candidate(marker_poses, edge.marker_id, score):
                    marker_poses[edge.marker_id] = (candidate, node_id, score, edge.marker_id == anchor_marker_id)
                    queue.append(("marker", edge.marker_id))

    seed_cameras = [
        SeedCameraPose(
            image_id=image_id,
            T_W_C=pose.to_json_dict(),
            source_marker_id=source_marker_id,
            reprojection_error_px=None if score == float("inf") else score,
        )
        for image_id, (pose, source_marker_id, score) in sorted(camera_poses.items())
    ]
    seed_markers = [
        SeedMarkerPose(
            marker_id=marker_id,
            T_W_M=pose.to_json_dict(),
            source_image_id=source_image_id,
            reprojection_error_px=None if score == float("inf") else score,
            is_anchor=is_anchor,
        )
        for marker_id, (pose, source_image_id, score, is_anchor) in sorted(marker_poses.items())
    ]
    disconnected = sorted(set(graph.marker_to_edges.keys()) - set(marker_poses.keys()))
    return SeedInitializationResult(seed_cameras, seed_markers, disconnected)


def initialize_seed_poses_from_store(store: ProjectStore) -> SeedInitializationResult:
    graph = build_graph_from_store(store)
    anchor = store.get_anchor_marker_id()
    result = initialize_seed_poses(graph, anchor)
    store.replace_seed_camera_poses(result.camera_poses)
    store.replace_seed_marker_poses(result.marker_poses)
    diagnostics = store.get_graph_diagnostics()
    diagnostics["seed_camera_poses"] = len(result.camera_poses)
    diagnostics["seed_marker_poses"] = len(result.marker_poses)
    diagnostics["seed_disconnected_markers"] = result.disconnected_markers
    store.set_graph_diagnostics(diagnostics)
    return result


def _score(value: float | None) -> float:
    return float("inf") if value is None else float(value)


def _is_better_camera_candidate(cameras: dict[int, tuple[SE3, int | None, float]], image_id: int, score: float) -> bool:
    return image_id not in cameras or score < cameras[image_id][2]


def _is_better_marker_candidate(
    markers: dict[int, tuple[SE3, int | None, float, bool]], marker_id: int, score: float
) -> bool:
    return marker_id not in markers or score < markers[marker_id][2]
