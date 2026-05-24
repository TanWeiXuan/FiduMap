"""CSV export for optimized marker corner points."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from map_builder.geometry import SE3, marker_corners_for_export_order, validate_marker_size
from map_builder.project import OptimizedMarkerPose, ProjectStore


MAX_MARKER_ID_FOR_UINT32_CORNER_IDS = (1 << 30) - 1


def export_optimized_marker_map_csv(
    store: ProjectStore,
    output_path: Path,
    ba_run_id: int | None = None,
) -> int:
    marker_size = store.get_marker_size_m()
    if marker_size is None:
        raise RuntimeError("Marker side length is not set.")
    poses = store.get_optimized_marker_poses(ba_run_id)
    if not poses:
        raise RuntimeError("No successful optimized BA marker poses are available to export.")
    return write_marker_map_csv(output_path, poses, marker_size)


def write_marker_map_csv(
    output_path: Path,
    marker_poses: Iterable[OptimizedMarkerPose],
    marker_size_m: float,
) -> int:
    marker_size = validate_marker_size(marker_size_m)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    corners_m = marker_corners_for_export_order(marker_size)
    count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "x", "y", "z"])
        for pose in sorted(marker_poses, key=lambda p: p.marker_id):
            _validate_marker_id(pose.marker_id)
            T_W_M = SE3.from_json_dict(pose.T_W_M)
            corners_w = T_W_M.transform_points(corners_m)
            for corner_index, point in enumerate(corners_w):
                point_id = (pose.marker_id << 2) | corner_index
                writer.writerow([point_id, f"{point[0]:.6f}", f"{point[1]:.6f}", f"{point[2]:.6f}"])
                count += 1
    return count


def _validate_marker_id(marker_id: int) -> None:
    if marker_id < 0:
        raise ValueError("Marker IDs must be non-negative.")
    if marker_id > MAX_MARKER_ID_FOR_UINT32_CORNER_IDS:
        raise ValueError("Marker ID is too large to encode corner IDs as uint32.")
