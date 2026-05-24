"""Simple reprojection-error outlier rejection."""

from __future__ import annotations

from collections import defaultdict

from map_builder.project import ReprojectionErrorRecord


def find_outlier_observations(
    records: list[ReprojectionErrorRecord],
    corner_threshold_px: float,
    marker_threshold_px: float,
) -> set[tuple[int, int]]:
    grouped: dict[tuple[int, int], list[ReprojectionErrorRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.image_id, record.marker_id)].append(record)

    outliers: set[tuple[int, int]] = set()
    for key, group in grouped.items():
        max_corner = max(record.error_px for record in group)
        mean_marker = sum(record.error_px for record in group) / len(group)
        if max_corner > corner_threshold_px or mean_marker > marker_threshold_px:
            outliers.add(key)
    return outliers
