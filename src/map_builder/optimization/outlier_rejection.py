"""Simple reprojection-error outlier rejection."""

from __future__ import annotations

import statistics

from map_builder.project import ReprojectionErrorRecord


def find_outlier_observations(
    records: list[ReprojectionErrorRecord],
    corner_threshold_px: float,
    marker_threshold_px: float,
) -> set[tuple[int, int]]:
    grouped_errors = {
        key: [
            record.error_px
            for record in records
            if (record.image_id, record.marker_id) == key
        ]
        for key in {(record.image_id, record.marker_id) for record in records}
    }

    outliers: set[tuple[int, int]] = set()
    for key, errors in grouped_errors.items():
        max_corner = max(errors)
        mean_marker = statistics.mean(errors)
        if max_corner > corner_threshold_px or mean_marker > marker_threshold_px:
            outliers.add(key)
    return outliers
