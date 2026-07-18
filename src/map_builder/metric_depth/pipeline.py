from __future__ import annotations

from collections.abc import Callable
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np

from map_builder.camera_models import load_camera_model_xml
from map_builder.dense_reconstruction.dense_store import DenseReconstructionStore
from map_builder.geometry.se3 import SE3
from map_builder.project import ProjectStore

from .backends import BACKENDS
from .export import export_stored_artifact
from .models import MetricDepthConfig, MetricDepthProgress, MetricDepthRunSummary
from .prompt_builder import build_trusted_prompt
from .store import MetricDepthStore, dense_state_signature


LOG = logging.getLogger(__name__)
Progress = Callable[[MetricDepthProgress], None]


class CancelledError(RuntimeError):
    pass


class MetricDepthPipeline:
    def __init__(self, project_folder: Path, backend_factories: dict[str, Any] | None = None):
        self.project_folder = Path(project_folder).expanduser().resolve()
        self.project_store = ProjectStore.open(self.project_folder)
        self.dense_store = DenseReconstructionStore.open(self.project_folder)
        self.store = MetricDepthStore.open(self.project_folder)
        self.backend_factories = backend_factories or BACKENDS
        self._backend: Any | None = None

    def close(self) -> None:
        if self._backend is not None:
            self._backend.close()
            self._backend = None
        self.store.close()
        self.dense_store.close()
        self.project_store.close()

    def __enter__(self) -> "MetricDepthPipeline":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def run_image(self, image_id: int, config: MetricDepthConfig, progress: Progress | None = None, cancel_event: Any = None) -> MetricDepthRunSummary:
        return self._run_images([int(image_id)], config, progress, cancel_event)

    def run_all(self, config: MetricDepthConfig, progress: Progress | None = None, cancel_event: Any = None) -> MetricDepthRunSummary:
        ba_run_id = self.project_store.get_latest_successful_ba_run_id()
        eligible = [pose.image_id for pose in self.project_store.get_optimized_camera_poses(ba_run_id)]
        return self._run_images(eligible, config, progress, cancel_event)

    def _run_images(self, image_ids: list[int], config: MetricDepthConfig, progress: Progress | None, cancel_event: Any) -> MetricDepthRunSummary:
        config.validate()
        ba_run_id = self.project_store.get_latest_successful_ba_run_id()
        if ba_run_id is None:
            raise RuntimeError("No successful marker BA run; optimized T_W_C poses are required.")
        signature = dense_state_signature(self.dense_store)
        run_id = self.store.create_run(config.backend, config.model_id_or_path, config.to_dict(), ba_run_id, signature)
        summary = MetricDepthRunSummary(run_id=run_id, total=len(image_ids))
        if not image_ids:
            self.store.complete_run(run_id, False, "No optimized images are eligible.")
            summary.details = "No optimized images are eligible."
            return summary
        try:
            self._check_cancel(cancel_event)
            self._emit(progress, "loading_model", f"Loading {config.backend} model", 0, len(image_ids), None, 0.0)
            backend = self.backend_factories[config.backend]()
            self._backend = backend
            backend.load(config)
        except CancelledError:
            summary.cancelled = True
            self.store.complete_run(run_id, False, "cancelled while loading model")
            return summary
        except Exception as exc:
            message = f"{config.backend}; loading_model; missing model or optional dependency: {exc}"
            self.store.complete_run(run_id, False, message)
            raise RuntimeError(message) from exc
        durations: list[float] = []
        for index, image_id in enumerate(image_ids, 1):
            if _cancelled(cancel_event):
                summary.cancelled = True
                break
            if not config.recompute and self.store.latest_successful_record(image_id, ba_run_id, config.backend) is not None:
                summary.skipped += 1
                self._emit(progress, "loading_image", f"Image {index}/{len(image_ids)} already has a current successful map; skipped", index, len(image_ids), image_id, index / len(image_ids))
                continue
            started = time.perf_counter()
            try:
                artifact = self._process_one(image_id, config, ba_run_id, signature, index, len(image_ids), progress, cancel_event)
                artifact.metrics.processing_duration_s = time.perf_counter() - started
                if artifact.metrics.status != "success":
                    raise RuntimeError(artifact.metrics.error_message or "backend rejected output as non-metric")
                self._emit(progress, "saving_artifact", f"Image {index}/{len(image_ids)} — saving artifact", index, len(image_ids), image_id, index / len(image_ids))
                self.store.save_artifact_atomic(run_id, artifact)
                summary.completed += 1
                durations.append(artifact.metrics.processing_duration_s)
            except CancelledError:
                summary.cancelled = True
                break
            except Exception as exc:
                duration = time.perf_counter() - started
                stage = getattr(exc, "stage", "processing")
                self.store.record_failure(run_id, image_id, stage, f"{config.backend}; image {image_id}; {exc}", getattr(exc, "metrics", None))
                summary.failed += 1
                LOG.warning("Metric depth failed for backend=%s image=%s stage=%s: %s", config.backend, image_id, stage, exc)
                self._emit(progress, stage, f"Image {index}/{len(image_ids)} failed: {exc}", index, len(image_ids), image_id, index / len(image_ids))
            if durations:
                remaining = (len(image_ids) - index) * float(np.mean(durations[-10:]))
                self._emit(progress, "saving_artifact", f"Image {index}/{len(image_ids)} complete; estimated {remaining:.0f}s remaining", index, len(image_ids), image_id, index / len(image_ids))
        success = summary.completed > 0 and summary.failed == 0 and not summary.cancelled
        error = "cancelled between images" if summary.cancelled else (f"{summary.failed} image(s) failed" if summary.failed else None)
        self.store.complete_run(run_id, success, error)
        summary.details = f"Run {run_id}: {summary.completed} completed, {summary.failed} failed, {summary.skipped} skipped" + (", cancelled" if summary.cancelled else "")
        return summary

    def _process_one(self, image_id: int, config: MetricDepthConfig, ba_run_id: int, dense_signature: str, index: int, total: int, progress: Progress | None, cancel_event: Any):
        self._check_cancel(cancel_event)
        record = self.project_store.get_image(image_id)
        poses = {pose.image_id: pose for pose in self.project_store.get_optimized_camera_poses(ba_run_id)}
        if record is None or record.missing or image_id not in poses:
            raise StageError("loading_image", "project image or optimized camera pose is missing")
        camera_path = self.project_store.get_camera_config_path()
        if camera_path is None or not camera_path.exists():
            raise StageError("loading_image", "camera configuration is missing")
        self._emit(progress, "loading_image", f"Image {index}/{total} — loading {record.rel_path}", index, total, image_id, (index - 1) / total)
        try:
            import cv2
            bgr = cv2.imread(str(record.absolute_path(self.project_folder)), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError("image could not be decoded")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Exception as exc:
            raise StageError("loading_image", str(exc)) from exc
        camera = load_camera_model_xml(camera_path)
        h, w = rgb.shape[:2]
        if (camera.image_width, camera.image_height) != (w, h):
            raise StageError("loading_image", f"invalid camera geometry: camera is {camera.image_width}x{camera.image_height}, image is {w}x{h}")
        self._check_cancel(cancel_event)
        self._emit(progress, "building_prompts", f"Image {index}/{total} — building trusted metric prompts", index, total, image_id, (index - 0.7) / total)
        T_W_C = SE3.from_json_dict(poses[image_id].T_W_C)
        prompt = build_trusted_prompt(
            image_id, T_W_C, camera, w, h, self.project_store.get_marker_size_m(),
            self.project_store.get_detections_for_image(image_id), self.project_store.get_optimized_marker_poses(ba_run_id),
            self.dense_store.list_active_dense_points(), self.dense_store.list_track_observations(), self.dense_store.list_tracks(),
            config.include_dense_track_points, config.include_marker_surfaces,
        )
        self._check_cancel(cancel_event)
        stage = "aligning_depth" if config.backend == "depth_anything_v2_aligned" else "verifying_depth"
        self._emit(progress, "running_inference", f"Image {index}/{total} — running {config.backend} inference", index, total, image_id, (index - 0.4) / total)
        try:
            artifact = self._backend.infer(rgb, prompt, camera, config, progress)
        except Exception as exc:
            raise StageError("running_inference", str(exc)) from exc
        if artifact.metrics.status != "success":
            raise StageError(stage, artifact.metrics.error_message or "backend rejected output as non-metric", artifact.metrics)
        artifact.image_id = image_id
        artifact.metadata.update({
            "image_id": image_id, "image_relative_path": record.rel_path, "backend": config.backend,
            "model_reference": config.model_id_or_path, "image_dimensions": [w, h], "camera_model_name": camera.model_name,
            "camera_configuration_path": str(camera_path), "T_W_C": poses[image_id].T_W_C, "marker_ba_run_id": ba_run_id,
            "dense_state_signature": dense_signature, "depth_conventions": {"model_output": "camera_z_depth_m", "canonical": "radial_range_m", "formula": "range_m=z_depth_m/ray_C.z"},
            "run_configuration": config.to_dict(),
        })
        self._emit(progress, stage, f"Image {index}/{total} — {'aligned' if 'align' in stage else 'verified'} against anchors", index, total, image_id, (index - 0.25) / total)
        self._emit(progress, "converting_range", f"Image {index}/{total} — converted Z-depth to radial range", index, total, image_id, (index - 0.15) / total)
        return artifact

    def export_image(self, image_id: int, run_id: int, output_folder: Path) -> dict[str, Path]:
        return export_stored_artifact(self.store, run_id, image_id, output_folder)

    def export_all(self, run_id: int, output_folder: Path) -> list[dict[str, Path]]:
        rows = self.store.conn.execute("SELECT image_id FROM depth_map_records WHERE run_id=? AND status='success' ORDER BY image_id", (run_id,)).fetchall()
        return [self.export_image(int(row["image_id"]), run_id, output_folder) for row in rows]

    @staticmethod
    def _emit(progress: Progress | None, stage: str, message: str, index: int, total: int, image_id: int | None, fraction: float | None) -> None:
        LOG.info(message)
        if progress is not None:
            progress(MetricDepthProgress(stage, message, index, total, image_id, None if fraction is None else float(np.clip(fraction, 0.0, 1.0))))

    @staticmethod
    def _check_cancel(cancel_event: Any) -> None:
        if _cancelled(cancel_event):
            raise CancelledError("metric-depth operation cancelled")


class StageError(RuntimeError):
    def __init__(self, stage: str, message: str, metrics: Any = None):
        super().__init__(message)
        self.stage = stage
        self.metrics = metrics


def _cancelled(event: Any) -> bool:
    return bool(event is not None and event.is_set())
