from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
import threading
from typing import Any

import numpy as np

from map_builder.camera_models import load_camera_model_xml
from map_builder.dense_reconstruction.dense_store import DenseReconstructionStore
from map_builder.geometry.se3 import SE3
from map_builder.project import ProjectStore

from .geometry import FORWARD_RAY_EPS, deterministic_decimate
from .models import BACKEND_DAV2, MetricDepthArtifact
from .store import MetricDepthStore


@dataclass(frozen=True)
class DepthCloudLayer:
    image_id: int
    source_run_id: int
    points_world: np.ndarray
    rgb: np.ndarray
    range_m: np.ndarray
    confidence: np.ndarray
    T_W_C: SE3
    candidate_count: int
    sampled_candidate_count: int

    @property
    def point_count(self) -> int:
        return int(len(self.points_world))

    @property
    def nbytes(self) -> int:
        return int(self.points_world.nbytes + self.rgb.nbytes + self.range_m.nbytes + self.confidence.nbytes)

    def limited(self, maximum_points: int) -> "DepthCloudLayer":
        indices = deterministic_decimate(self.point_count, int(maximum_points))
        if len(indices) == self.point_count:
            return self
        return DepthCloudLayer(
            self.image_id,
            self.source_run_id,
            self.points_world[indices],
            self.rgb[indices],
            self.range_m[indices],
            self.confidence[indices],
            self.T_W_C,
            self.candidate_count,
            min(self.sampled_candidate_count, int(maximum_points)),
        )


@dataclass
class DepthSceneResult:
    selected_count: int
    layers: list[DepthCloudLayer] = field(default_factory=list)
    skipped: dict[int, str] = field(default_factory=dict)
    primary_artifact: MetricDepthArtifact | None = None
    primary_run_id: int | None = None
    camera_model: Any | None = None
    marker_poses: list[Any] = field(default_factory=list)
    marker_size_m: float | None = None
    dense_points: list[Any] = field(default_factory=list)

    @property
    def point_count(self) -> int:
        return sum(layer.point_count for layer in self.layers)


def fair_point_budgets(image_ids: list[int], maximum_points: int) -> dict[int, int]:
    ordered = list(dict.fromkeys(int(image_id) for image_id in image_ids))
    maximum = max(int(maximum_points), 0)
    if not ordered:
        return {}
    base, remainder = divmod(maximum, len(ordered))
    return {image_id: base + (1 if index < remainder else 0) for index, image_id in enumerate(ordered)}


def sample_depth_cloud_layer(
    artifact: MetricDepthArtifact,
    camera_model: Any,
    T_W_C: SE3,
    rgb: np.ndarray,
    confidence_threshold: float,
    maximum_points: int,
    source_run_id: int,
) -> DepthCloudLayer:
    ranges = np.asarray(artifact.range_m, dtype=np.float32)
    confidence = np.asarray(artifact.confidence, dtype=np.float32)
    valid = np.asarray(artifact.valid_mask, dtype=bool)
    valid &= np.isfinite(ranges) & (ranges > 0.0)
    valid &= np.isfinite(confidence) & (confidence >= float(confidence_threshold))
    candidates = np.flatnonzero(valid.ravel())
    candidate_count = int(len(candidates))
    selected = candidates[deterministic_decimate(candidate_count, int(maximum_points))]
    height, width = ranges.shape
    pixels = np.column_stack((selected % width, selected // width)).astype(np.float64, copy=False)
    rays = np.asarray(camera_model.unproject_many(pixels), dtype=float).reshape(-1, 3)
    selected_ranges = ranges.ravel()[selected]
    ray_valid = np.all(np.isfinite(rays), axis=1) & (rays[:, 2] > FORWARD_RAY_EPS)
    ray_valid &= np.isfinite(selected_ranges) & (selected_ranges > 0.0)
    pixels_i = pixels[ray_valid].astype(np.int32, copy=False)
    points_c = rays[ray_valid] * selected_ranges[ray_valid, None]
    points_w = T_W_C.transform_points(points_c).astype(np.float32, copy=False)
    source_rgb = np.asarray(rgb, dtype=np.uint8)
    if source_rgb.shape[:2] != (height, width) or source_rgb.ndim != 3 or source_rgb.shape[2] < 3:
        raise ValueError(
            f"RGB dimensions {source_rgb.shape} do not match depth dimensions {(height, width)}."
        )
    colors = source_rgb[pixels_i[:, 1], pixels_i[:, 0], :3].copy()
    return DepthCloudLayer(
        image_id=int(artifact.image_id),
        source_run_id=int(source_run_id),
        points_world=points_w,
        rgb=colors,
        range_m=selected_ranges[ray_valid].astype(np.float32, copy=False),
        confidence=confidence.ravel()[selected][ray_valid].astype(np.float32, copy=False),
        T_W_C=T_W_C,
        candidate_count=candidate_count,
        sampled_candidate_count=int(len(selected)),
    )


class DepthSceneBuilder:
    def __init__(self, project_folder: Path, maximum_cache_bytes: int = 256 * 1024 * 1024):
        self.project_folder = Path(project_folder).expanduser().resolve()
        self.maximum_cache_bytes = int(maximum_cache_bytes)
        self._cache: OrderedDict[tuple[Any, ...], DepthCloudLayer] = OrderedDict()
        self._cache_bytes = 0

    def build(
        self,
        selected_image_ids: list[int],
        primary_image_id: int | None,
        confidence_threshold: float,
        maximum_points: int,
        cancel_event: threading.Event | None = None,
        include_dense_points: bool = False,
    ) -> DepthSceneResult:
        selected_ids = list(dict.fromkeys(int(image_id) for image_id in selected_image_ids))
        result = DepthSceneResult(selected_count=len(selected_ids))
        if not selected_ids:
            return result
        with ProjectStore.open(self.project_folder) as project_store, MetricDepthStore.open(self.project_folder) as depth_store:
            ba_run_id = project_store.get_latest_successful_ba_run_id()
            camera_path = project_store.get_camera_config_path()
            if ba_run_id is None or camera_path is None or not camera_path.exists():
                reason = "current optimized poses or camera configuration are unavailable"
                result.skipped = {image_id: reason for image_id in selected_ids}
                return result
            camera_model = load_camera_model_xml(camera_path)
            result.camera_model = camera_model
            pose_by_image = {
                pose.image_id: SE3.from_json_dict(pose.T_W_C)
                for pose in project_store.get_optimized_camera_poses(ba_run_id)
            }
            usable: list[tuple[int, Any, Any, SE3]] = []
            for image_id in selected_ids:
                record = project_store.get_image(image_id)
                pose = pose_by_image.get(image_id)
                row = depth_store.latest_successful_record(image_id, ba_run_id, BACKEND_DAV2)
                if record is None or record.missing:
                    result.skipped[image_id] = "image is missing"
                elif pose is None:
                    result.skipped[image_id] = "optimized camera pose is unavailable"
                elif row is None:
                    result.skipped[image_id] = "current DAV2 depth artifact is unavailable"
                else:
                    usable.append((image_id, record, row, pose))
            budgets = fair_point_budgets([item[0] for item in usable], maximum_points)
            camera_mtime = camera_path.stat().st_mtime_ns
            loaded_primary: MetricDepthArtifact | None = None
            for image_id, record, row, pose in usable:
                _check_cancel(cancel_event)
                run_id = int(row["run_id"])
                budget = budgets[image_id]
                cache_key = (
                    image_id,
                    run_id,
                    int(ba_run_id),
                    camera_mtime,
                    int(record.size_bytes),
                    int(record.mtime_ns),
                    round(float(confidence_threshold), 6),
                )
                cached = self._cache_get(cache_key)
                needs_more = cached is None or cached.sampled_candidate_count < min(budget, cached.candidate_count)
                artifact: MetricDepthArtifact | None = None
                try:
                    if needs_more:
                        artifact = depth_store.load_artifact(run_id, image_id)
                        if artifact is None:
                            raise RuntimeError("artifact file is missing")
                        import cv2

                        bgr = cv2.imread(str(record.absolute_path(self.project_folder)), cv2.IMREAD_COLOR)
                        if bgr is None:
                            raise RuntimeError("image could not be decoded")
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        cached = sample_depth_cloud_layer(
                            artifact,
                            camera_model,
                            pose,
                            rgb,
                            confidence_threshold,
                            budget,
                            run_id,
                        )
                        self._cache_put(cache_key, cached)
                    assert cached is not None
                    if image_id == primary_image_id:
                        if artifact is None:
                            artifact = depth_store.load_artifact(run_id, image_id)
                        loaded_primary = artifact
                        result.primary_run_id = run_id
                    limited = cached.limited(budget)
                    if limited.point_count == 0:
                        result.skipped[image_id] = "no points available under the scene budget or confidence threshold"
                        continue
                    result.layers.append(limited)
                except Exception as exc:
                    result.skipped[image_id] = str(exc)
            if primary_image_id is not None and loaded_primary is None and primary_image_id not in result.skipped:
                primary_row = depth_store.latest_successful_record(primary_image_id, ba_run_id, BACKEND_DAV2)
                if primary_row is not None:
                    loaded_primary = depth_store.load_artifact(int(primary_row["run_id"]), primary_image_id)
                    result.primary_run_id = int(primary_row["run_id"])
            result.primary_artifact = loaded_primary
            result.marker_poses = project_store.get_optimized_marker_poses(ba_run_id)
            result.marker_size_m = project_store.get_marker_size_m()
        _check_cancel(cancel_event)
        if include_dense_points:
            try:
                with DenseReconstructionStore.open(self.project_folder) as dense_store:
                    result.dense_points = dense_store.list_active_dense_points()
            except Exception:
                result.dense_points = []
        return result

    def _cache_get(self, key: tuple[Any, ...]) -> DepthCloudLayer | None:
        layer = self._cache.pop(key, None)
        if layer is not None:
            self._cache[key] = layer
        return layer

    def _cache_put(self, key: tuple[Any, ...], layer: DepthCloudLayer) -> None:
        existing = self._cache.pop(key, None)
        if existing is not None:
            self._cache_bytes -= existing.nbytes
        if layer.nbytes > self.maximum_cache_bytes:
            return
        self._cache[key] = layer
        self._cache_bytes += layer.nbytes
        while self._cache and self._cache_bytes > self.maximum_cache_bytes:
            _old_key, old_layer = self._cache.popitem(last=False)
            self._cache_bytes -= old_layer.nbytes


def _check_cancel(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise InterruptedError("depth scene loading was superseded by a newer selection")
