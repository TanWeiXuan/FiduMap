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

    marker_poses: dict[int, tuple[SE3, int | None, float | None, bool]] = {
        anchor_marker_id: (SE3.identity(), None, 0.0, True)
    }
    camera_poses: dict[int, tuple[SE3, int | None, float | None]] = {}
    queue: deque[tuple[str, int]] = deque([("marker", anchor_marker_id)])

    while queue:
        kind, node_id = queue.popleft()
        if kind == "marker":
            T_W_M = marker_poses[node_id][0]
            for edge in graph.marker_to_edges.get(node_id, []):
                candidate = T_W_M @ edge.T_C_M.inverse()
                score = edge.reprojection_error_px
                existing_score = camera_poses.get(edge.image_id, (None, None, None))[2]
                if _is_better(existing_score, score):
                    camera_poses[edge.image_id] = (candidate, node_id, score)
                    queue.append(("camera", edge.image_id))
        else:
            T_W_C = camera_poses[node_id][0]
            for edge in graph.camera_to_edges.get(node_id, []):
                candidate = T_W_C @ edge.T_C_M
                score = edge.reprojection_error_px
                existing_score = marker_poses.get(edge.marker_id, (None, None, None, False))[2]
                if _is_better(existing_score, score):
                    marker_poses[edge.marker_id] = (candidate, node_id, score, edge.marker_id == anchor_marker_id)
                    queue.append(("marker", edge.marker_id))

    seed_cameras = [
        SeedCameraPose(
            image_id=image_id,
            T_W_C=pose.to_json_dict(),
            source_marker_id=source_marker_id,
            reprojection_error_px=score,
        )
        for image_id, (pose, source_marker_id, score) in sorted(camera_poses.items())
    ]
    seed_markers = [
        SeedMarkerPose(
            marker_id=marker_id,
            T_W_M=pose.to_json_dict(),
            source_image_id=source_image_id,
            reprojection_error_px=score,
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


def _is_better(existing_score: float | None, new_score: float | None) -> bool:
    existing = float("inf") if existing_score is None else float(existing_score)
    new = float("inf") if new_score is None else float(new_score)
    return new < existing
