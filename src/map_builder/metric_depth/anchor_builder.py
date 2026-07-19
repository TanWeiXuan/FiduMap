from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from typing import Any

import numpy as np

from map_builder.geometry.se3 import SE3

from .alignment import occupied_grid_cells
from .geometry import FORWARD_RAY_EPS
from .models import AlignmentAnchorRaster, MetricAnchor


MARKER_SURFACE = "marker_surface"
DENSE_TRACK = "dense_track"


def dense_track_anchors(
    image_id: int,
    T_W_C: SE3,
    dense_points: Iterable[Any],
    observations: Iterable[Any],
    tracks: Iterable[Any] = (),
) -> list[MetricAnchor]:
    """Use only active points whose track is actually observed in this image."""
    point_by_track = {
        int(_value(p, "track_id")): p
        for p in dense_points
        if _value(p, "track_id") is not None and bool(_value(p, "is_active", 1))
    }
    track_by_id = {int(_value(t, "id")): t for t in tracks if _value(t, "id") is not None}
    anchors: list[MetricAnchor] = []
    Rcw = T_W_C.R.T
    for obs in observations:
        if int(_value(obs, "image_id")) != int(image_id):
            continue
        track_id = int(_value(obs, "track_id"))
        point = point_by_track.get(track_id)
        if point is None:
            continue
        track = track_by_id.get(track_id)
        if track is not None and str(_value(track, "status", "active")) != "active":
            continue
        X_W = np.array([_value(point, "x"), _value(point, "y"), _value(point, "z")], dtype=float)
        X_C = Rcw @ (X_W - T_W_C.t)
        if not np.all(np.isfinite(X_C)) or X_C[2] <= FORWARD_RAY_EPS:
            continue
        radial = float(np.linalg.norm(X_C))
        if not np.isfinite(radial) or radial <= 0.0:
            continue
        error = _value(point, "mean_reprojection_error_px")
        if error is None and track is not None:
            error = _value(track, "mean_reprojection_error_px")
        count = _value(point, "num_observations") or (None if track is None else _value(track, "num_observations")) or 1
        confidence = float(np.clip((1.0 - min(float(error or 0.0), 10.0) / 12.0) * min(float(count) / 4.0, 1.0), 0.1, 0.9))
        u, v = float(_value(obs, "x")), float(_value(obs, "y"))
        if not np.isfinite(u) or not np.isfinite(v):
            continue
        anchors.append(MetricAnchor(
            u, v, float(X_C[2]), radial, confidence, DENSE_TRACK,
            f"track:{track_id}", DENSE_TRACK, confidence, confidence,
        ))
    return anchors


def marker_surface_anchors(
    detections: Iterable[Any],
    marker_poses: Iterable[Any],
    T_W_C: SE3,
    camera_model: Any,
    marker_size_m: float,
    width: int,
    height: int,
    sample_grid_size: int = 6,
) -> list[MetricAnchor]:
    pose_by_marker = {int(_value(p, "marker_id")): SE3.from_json_dict(_value(p, "T_W_M")) for p in marker_poses}
    anchors: list[MetricAnchor] = []
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to sample visible marker surfaces.") from exc
    half = float(marker_size_m) / 2.0
    coordinate = np.linspace(-half, half, int(sample_grid_size), dtype=float)
    gx, gy = np.meshgrid(coordinate, coordinate)
    local_points = np.column_stack((gx.ravel(), gy.ravel(), np.zeros(gx.size, dtype=float)))
    R_C_W = T_W_C.R.T
    for detection in detections:
        marker_id = int(_value(detection, "marker_id"))
        pose = pose_by_marker.get(marker_id)
        corners = np.asarray(_value(detection, "corners"), dtype=float)
        if pose is None or corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
            continue
        X_W = pose.transform_points(local_points)
        X_C = (R_C_W @ (X_W - T_W_C.t).T).T
        finite_forward = np.all(np.isfinite(X_C), axis=1) & (X_C[:, 2] > FORWARD_RAY_EPS)
        try:
            pixels = np.asarray(camera_model.project_many(X_C), dtype=float)
        except (ValueError, FloatingPointError):
            continue
        finite_pixels = np.all(np.isfinite(pixels), axis=1)
        in_bounds = finite_pixels & (pixels[:, 0] >= 0.0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0.0) & (pixels[:, 1] < height)
        polygon = corners.astype(np.float32)
        inside = np.array([
            bool(ok and cv2.pointPolygonTest(polygon, (float(pixel[0]), float(pixel[1])), False) >= 0)
            for ok, pixel in zip(in_bounds, pixels)
        ])
        valid = finite_forward & in_bounds & inside
        for pixel, point in zip(pixels[valid], X_C[valid]):
            radial = float(np.linalg.norm(point))
            if np.isfinite(radial) and radial > 0.0:
                anchors.append(MetricAnchor(
                    float(pixel[0]), float(pixel[1]), float(point[2]), radial, 1.0, MARKER_SURFACE,
                    f"marker:{marker_id}", MARKER_SURFACE, 1.0, 1.0,
                ))
    return anchors


def balance_anchor_weights(
    anchors: Iterable[MetricAnchor],
    marker_source_total: float = 1.0,
    dense_source_total: float = 1.0,
) -> list[MetricAnchor]:
    """Normalize source/group weights without allowing sample count to dominate."""
    values = list(anchors)
    marker = [a for a in values if (a.source or a.provenance) == MARKER_SURFACE]
    dense = [a for a in values if (a.source or a.provenance) == DENSE_TRACK]
    present = int(bool(marker)) + int(bool(dense))
    if present == 2:
        marker_total = float(marker_source_total) / max(float(marker_source_total + dense_source_total), 1e-12)
        dense_total = 1.0 - marker_total
    else:
        marker_total, dense_total = (1.0, 0.0) if marker else (0.0, 1.0 if dense else 0.0)

    marker_groups: dict[str, list[MetricAnchor]] = {}
    for anchor in marker:
        marker_groups.setdefault(anchor.group_id or "marker:unknown", []).append(anchor)
    weights: dict[int, float] = {}
    for group in marker_groups.values():
        group_total = marker_total / max(len(marker_groups), 1)
        for anchor in group:
            weights[id(anchor)] = group_total / len(group)
    dense_raw = np.array([max(_raw_confidence(a), 0.0) for a in dense], dtype=float)
    if len(dense):
        if not np.any(dense_raw > 0.0):
            dense_raw[:] = 1.0
        dense_raw /= float(np.sum(dense_raw))
        for anchor, weight in zip(dense, dense_raw):
            weights[id(anchor)] = dense_total * float(weight)
    return [replace(
        anchor,
        source=anchor.source or anchor.provenance,
        group_id=anchor.group_id or f"{anchor.provenance}:unknown",
        raw_confidence=_raw_confidence(anchor),
        fit_weight=float(weights.get(id(anchor), 0.0)),
    ) for anchor in values]


def rasterize_anchors(anchors: Iterable[MetricAnchor], width: int, height: int, grid_size: int = 4) -> AlignmentAnchorRaster:
    depth = np.zeros((height, width), dtype=np.float32)
    mask = np.zeros((height, width), dtype=bool)
    confidence = np.zeros((height, width), dtype=np.float32)
    provenance = np.zeros((height, width), dtype=np.uint8)
    anchor_list = list(anchors)
    for anchor in anchor_list:
        u, v = int(round(anchor.u)), int(round(anchor.v))
        if not (0 <= u < width and 0 <= v < height) or not np.isfinite(anchor.z_depth_m) or anchor.z_depth_m <= 0.0:
            continue
        priority = 2 if anchor.provenance == MARKER_SURFACE else 1
        replace = not mask[v, u]
        if mask[v, u]:
            current_priority = int(provenance[v, u])
            replace = priority > current_priority
            if priority == current_priority:
                current_confidence = float(confidence[v, u])
                replace = anchor.confidence > current_confidence or (
                    np.isclose(anchor.confidence, current_confidence) and anchor.z_depth_m < depth[v, u]
                )
        if replace:
            depth[v, u] = np.float32(anchor.z_depth_m)
            confidence[v, u] = np.float32(np.clip(anchor.confidence, 0.0, 1.0))
            provenance[v, u] = np.uint8(priority)
            mask[v, u] = True
    cells = occupied_grid_cells(mask, grid_size)
    return AlignmentAnchorRaster(
        depth, mask, confidence, provenance, len(anchor_list), int(np.count_nonzero(mask)),
        cells, float(cells) / float(grid_size * grid_size), tuple(anchor_list),
    )


def build_alignment_anchors(
    image_id: int,
    T_W_C: SE3,
    camera_model: Any,
    width: int,
    height: int,
    marker_size_m: float | None,
    detections: Iterable[Any],
    marker_poses: Iterable[Any],
    dense_points: Iterable[Any],
    observations: Iterable[Any],
    tracks: Iterable[Any],
    include_dense_tracks: bool,
    include_marker_surfaces: bool,
    marker_sample_grid_size: int = 6,
) -> AlignmentAnchorRaster:
    anchors: list[MetricAnchor] = []
    if include_dense_tracks:
        anchors.extend(dense_track_anchors(image_id, T_W_C, dense_points, observations, tracks))
    if include_marker_surfaces and marker_size_m is not None:
        anchors.extend(marker_surface_anchors(
            detections, marker_poses, T_W_C, camera_model, marker_size_m, width, height,
            marker_sample_grid_size,
        ))
    raster = rasterize_anchors(balance_anchor_weights(anchors), width, height)
    raster.image_id = int(image_id)
    return raster


def _value(record: Any, key: str, default: Any = None) -> Any:
    if isinstance(record, dict):
        return record.get(key, default)
    try:
        return record[key]
    except (KeyError, TypeError, IndexError):
        return getattr(record, key, default)


def _raw_confidence(anchor: MetricAnchor) -> float:
    value = anchor.raw_confidence
    return float(anchor.confidence if value is None or not np.isfinite(value) else value)
