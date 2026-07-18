from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from map_builder.geometry.se3 import SE3

from .alignment import occupied_grid_cells
from .geometry import FORWARD_RAY_EPS, intersect_marker_plane
from .models import PromptAnchor, PromptRaster


MARKER_SURFACE = "marker_surface"
DENSE_TRACK = "dense_track"


def dense_track_anchors(
    image_id: int,
    T_W_C: SE3,
    dense_points: Iterable[Any],
    observations: Iterable[Any],
    tracks: Iterable[Any] = (),
) -> list[PromptAnchor]:
    """Use only active points whose track is actually observed in this image."""
    point_by_track = {
        int(_value(p, "track_id")): p
        for p in dense_points
        if _value(p, "track_id") is not None and bool(_value(p, "is_active", 1))
    }
    track_by_id = {int(_value(t, "id")): t for t in tracks if _value(t, "id") is not None}
    anchors: list[PromptAnchor] = []
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
        anchors.append(PromptAnchor(u, v, float(X_C[2]), radial, confidence, DENSE_TRACK))
    return anchors


def marker_surface_anchors(
    detections: Iterable[Any],
    marker_poses: Iterable[Any],
    T_W_C: SE3,
    camera_model: Any,
    marker_size_m: float,
    width: int,
    height: int,
) -> list[PromptAnchor]:
    pose_by_marker = {int(_value(p, "marker_id")): SE3.from_json_dict(_value(p, "T_W_M")) for p in marker_poses}
    anchors: list[PromptAnchor] = []
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to rasterize visible marker surfaces.") from exc
    for detection in detections:
        pose = pose_by_marker.get(int(_value(detection, "marker_id")))
        corners = np.asarray(_value(detection, "corners"), dtype=float)
        if pose is None or corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
            continue
        polygon = np.round(corners).astype(np.int32)
        x, y, w, h = cv2.boundingRect(polygon)
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w, width), min(y + h, height)
        if x0 >= x1 or y0 >= y1:
            continue
        local_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        shifted = polygon - np.array([x0, y0], dtype=np.int32)
        cv2.fillConvexPoly(local_mask, shifted, 1)
        yy, xx = np.nonzero(local_mask)
        pixels = np.column_stack((xx + x0, yy + y0)).astype(float)
        valid, z, ranges = intersect_marker_plane(pixels, camera_model, T_W_C, pose, marker_size_m)
        for pixel, z_m, range_m in zip(pixels[valid], z[valid], ranges[valid]):
            anchors.append(PromptAnchor(float(pixel[0]), float(pixel[1]), float(z_m), float(range_m), 1.0, MARKER_SURFACE))
    return anchors


def rasterize_anchors(anchors: Iterable[PromptAnchor], width: int, height: int, grid_size: int = 4) -> PromptRaster:
    depth = np.zeros((height, width), dtype=np.float32)
    mask = np.zeros((height, width), dtype=bool)
    confidence = np.zeros((height, width), dtype=np.float32)
    provenance = np.zeros((height, width), dtype=np.uint8)
    anchor_list = list(anchors)
    for anchor in anchor_list:
        u, v = int(round(anchor.u)), int(round(anchor.v))
        if not (0 <= u < width and 0 <= v < height):
            continue
        if not np.isfinite(anchor.z_depth_m) or anchor.z_depth_m <= 0.0:
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
    return PromptRaster(
        depth_z_m=depth,
        mask=mask,
        confidence=confidence,
        provenance=provenance,
        anchor_count=len(anchor_list),
        pixel_count=int(np.count_nonzero(mask)),
        occupied_grid_cells=cells,
        spatial_coverage=float(cells) / float(grid_size * grid_size),
    )


def build_trusted_prompt(
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
) -> PromptRaster:
    anchors: list[PromptAnchor] = []
    if include_dense_tracks:
        anchors.extend(dense_track_anchors(image_id, T_W_C, dense_points, observations, tracks))
    if include_marker_surfaces and marker_size_m is not None:
        anchors.extend(marker_surface_anchors(detections, marker_poses, T_W_C, camera_model, marker_size_m, width, height))
    return rasterize_anchors(anchors, width, height)


def _value(record: Any, key: str, default: Any = None) -> Any:
    if isinstance(record, dict):
        return record.get(key, default)
    try:
        return record[key]
    except (KeyError, TypeError, IndexError):
        return getattr(record, key, default)
