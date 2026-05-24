"""Text helpers for initialization diagnostics."""

from __future__ import annotations

from typing import Any


def format_graph_diagnostics(values: dict[str, Any]) -> str:
    if not values:
        return "No graph diagnostics yet"
    lines = [
        f"Cameras: {values.get('num_camera_nodes', 0)}",
        f"Markers: {values.get('num_marker_nodes', 0)}",
        f"Camera-marker edges: {values.get('num_camera_marker_edges', 0)}",
        f"Marker-overlap edges: {values.get('num_marker_overlap_edges', 0)}",
        f"Components: {values.get('connected_components', 0)}",
        f"Anchor exists: {values.get('anchor_marker_exists', False)}",
        f"Markers connected to anchor: {values.get('markers_connected_to_anchor', 0)}",
        f"Disconnected markers: {values.get('disconnected_markers', [])}",
    ]
    if "seed_camera_poses" in values:
        lines.append(f"Seed cameras: {values.get('seed_camera_poses', 0)}")
        lines.append(f"Seed markers: {values.get('seed_marker_poses', 0)}")
    return "\n".join(lines)
