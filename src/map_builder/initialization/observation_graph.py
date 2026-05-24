"""Camera-marker and marker-overlap graph construction."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

from map_builder.geometry import SE3
from map_builder.project import ObservationGraphSummary, PnPObservation, ProjectStore


@dataclass(frozen=True)
class CameraMarkerEdge:
    image_id: int
    marker_id: int
    pnp_observation_id: int | None
    T_C_M: SE3
    reprojection_error_px: float | None


@dataclass
class ObservationGraph:
    camera_to_edges: dict[int, list[CameraMarkerEdge]] = field(default_factory=dict)
    marker_to_edges: dict[int, list[CameraMarkerEdge]] = field(default_factory=dict)
    marker_overlap_edges: dict[tuple[int, int], dict[str, Any]] = field(default_factory=dict)
    summary: ObservationGraphSummary | None = None


def build_observation_graph(
    observations: list[PnPObservation],
    anchor_marker_id: int = 0,
) -> ObservationGraph:
    graph = ObservationGraph(defaultdict(list), defaultdict(list), {})
    by_image: dict[int, set[int]] = defaultdict(set)
    for obs in observations:
        if not obs.success or obs.T_C_M is None:
            continue
        edge = CameraMarkerEdge(
            image_id=obs.image_id,
            marker_id=obs.marker_id,
            pnp_observation_id=obs.id,
            T_C_M=SE3.from_json_dict(obs.T_C_M),
            reprojection_error_px=obs.reprojection_error_px,
        )
        graph.camera_to_edges[edge.image_id].append(edge)
        graph.marker_to_edges[edge.marker_id].append(edge)
        by_image[edge.image_id].add(edge.marker_id)

    overlap: dict[tuple[int, int], dict[str, Any]] = {}
    for image_id, markers in by_image.items():
        sorted_markers = sorted(markers)
        for i, marker_a in enumerate(sorted_markers):
            for marker_b in sorted_markers[i + 1 :]:
                key = (marker_a, marker_b)
                data = overlap.setdefault(key, {"shared_image_ids": [], "num_shared_observations": 0})
                data["shared_image_ids"].append(image_id)
                data["num_shared_observations"] += 1
    graph.marker_overlap_edges = overlap
    graph.summary = summarize_observation_graph(graph, anchor_marker_id)
    return graph


def summarize_observation_graph(graph: ObservationGraph, anchor_marker_id: int = 0) -> ObservationGraphSummary:
    camera_nodes = set(graph.camera_to_edges.keys())
    marker_nodes = set(graph.marker_to_edges.keys())
    adjacency: dict[str, set[str]] = defaultdict(set)
    observations_per_marker: dict[int, int] = {}
    observations_per_image: dict[int, int] = {}
    for image_id, edges in graph.camera_to_edges.items():
        observations_per_image[image_id] = len(edges)
        c_node = f"C{image_id}"
        for edge in edges:
            m_node = f"M{edge.marker_id}"
            adjacency[c_node].add(m_node)
            adjacency[m_node].add(c_node)
    for marker_id, edges in graph.marker_to_edges.items():
        observations_per_marker[marker_id] = len(edges)

    components = _connected_components(adjacency)
    anchor_node = f"M{anchor_marker_id}"
    anchor_component = next((component for component in components if anchor_node in component), set())
    connected_markers = {int(node[1:]) for node in anchor_component if node.startswith("M")}
    disconnected_markers = sorted(marker_nodes - connected_markers)
    return ObservationGraphSummary(
        num_camera_nodes=len(camera_nodes),
        num_marker_nodes=len(marker_nodes),
        num_camera_marker_edges=sum(len(edges) for edges in graph.camera_to_edges.values()),
        num_marker_overlap_edges=len(graph.marker_overlap_edges),
        connected_components=len(components),
        anchor_marker_exists=anchor_marker_id in marker_nodes,
        markers_connected_to_anchor=len(connected_markers),
        disconnected_markers=disconnected_markers,
        observations_per_marker=observations_per_marker,
        observations_per_image=observations_per_image,
    )


def build_graph_from_store(store: ProjectStore) -> ObservationGraph:
    anchor_marker_id = store.get_anchor_marker_id()
    graph = build_observation_graph(store.list_pnp_observations(success_only=True), anchor_marker_id)
    if graph.summary is not None:
        diagnostics = observation_graph_summary_to_dict(graph.summary)
        diagnostics["anchor_marker_id"] = anchor_marker_id
        store.set_graph_diagnostics(diagnostics)
    return graph


def observation_graph_summary_to_dict(summary: ObservationGraphSummary) -> dict[str, Any]:
    return {
        "num_camera_nodes": summary.num_camera_nodes,
        "num_marker_nodes": summary.num_marker_nodes,
        "num_camera_marker_edges": summary.num_camera_marker_edges,
        "num_marker_overlap_edges": summary.num_marker_overlap_edges,
        "connected_components": summary.connected_components,
        "anchor_marker_exists": summary.anchor_marker_exists,
        "markers_connected_to_anchor": summary.markers_connected_to_anchor,
        "disconnected_markers": summary.disconnected_markers,
        "observations_per_marker": summary.observations_per_marker,
        "observations_per_image": summary.observations_per_image,
    }


def _connected_components(adjacency: dict[str, set[str]]) -> list[set[str]]:
    seen: set[str] = set()
    components: list[set[str]] = []
    for start in sorted(adjacency):
        if start in seen:
            continue
        component: set[str] = set()
        queue: deque[str] = deque([start])
        seen.add(start)
        while queue:
            node = queue.popleft()
            component.add(node)
            for neighbor in adjacency[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        components.append(component)
    return components
