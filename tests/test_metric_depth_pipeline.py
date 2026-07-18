import threading

import cv2
import numpy as np

from map_builder.camera_models import PinholeRadTanCameraModel
from map_builder.geometry import SE3
from map_builder.metric_depth.models import BACKEND_DAV2, MetricDepthArtifact, MetricDepthConfig, MetricDepthMetrics
from map_builder.metric_depth.pipeline import MetricDepthPipeline
from map_builder.project import BAConfig, OptimizedCameraPose, ProjectStore


def _project(tmp_path, values=(10, 20, 30)):
    camera = PinholeRadTanCameraModel(8, 6, 5, 5, 3.5, 2.5, 0, 0, 0, 0, 0)
    camera_path = tmp_path / "camera.xml"
    camera.save_xml(camera_path)
    with ProjectStore.open(tmp_path) as store:
        store.set_camera_config_path(camera_path)
        for index, value in enumerate(values):
            path = tmp_path / f"{index}.png"
            cv2.imwrite(str(path), np.full((6, 8, 3), value, dtype=np.uint8))
            stat = path.stat()
            store.upsert_image_index_entry(path.name, stat.st_size, stat.st_mtime_ns, 8, 6)
        images = store.list_images()
        run = store.create_ba_run(BAConfig())
        store.complete_ba_run(run, True)
        store.replace_optimized_camera_poses(run, [OptimizedCameraPose(image.id, run, SE3.identity().to_json_dict()) for image in images])
    return [image.id for image in images]


class _FakeBackend:
    loads = 0
    fail_value = None

    def load(self, _config):
        type(self).loads += 1

    def infer(self, image_rgb, prompt, _camera, _config, _progress=None):
        h, w = image_rgb.shape[:2]
        z = np.full((h, w), 2.0, dtype=np.float32)
        failed = self.fail_value is not None and int(image_rgb[0, 0, 0]) == self.fail_value
        metrics = MetricDepthMetrics(
            prompt_point_count=prompt.anchor_count,
            prompt_pixel_count=prompt.pixel_count,
            prompt_spatial_coverage=prompt.spatial_coverage,
            valid_output_fraction=0.0 if failed else 1.0,
            status="failed" if failed else "success",
            error_message="synthetic alignment failure" if failed else None,
        )
        valid = np.zeros((h, w), bool) if failed else np.ones((h, w), bool)
        return MetricDepthArtifact(0, BACKEND_DAV2, w, h, z, z.copy(), valid, valid.astype(np.float32), prompt.depth_z_m, prompt.mask, {}, metrics)

    def close(self):
        pass


def _config():
    return MetricDepthConfig(backend=BACKEND_DAV2, model_id_or_path="fake-local")


def test_selected_image_success_and_progress_order(tmp_path):
    image_ids = _project(tmp_path, (10,))
    events = []
    pipeline = MetricDepthPipeline(tmp_path, {BACKEND_DAV2: _FakeBackend})
    summary = pipeline.run_image(image_ids[0], _config(), events.append)
    artifact = pipeline.store.load_artifact(summary.run_id, image_ids[0])
    pipeline.close()
    assert summary.completed == 1 and artifact is not None
    stages = [event.stage for event in events]
    assert stages.index("loading_model") < stages.index("loading_image") < stages.index("building_prompts") < stages.index("running_inference") < stages.index("saving_artifact")


def test_batch_loads_model_once_and_continues_after_failure(tmp_path):
    image_ids = _project(tmp_path)
    _FakeBackend.loads = 0
    _FakeBackend.fail_value = 20
    pipeline = MetricDepthPipeline(tmp_path, {BACKEND_DAV2: _FakeBackend})
    summary = pipeline.run_all(_config())
    statuses = [pipeline.store.get_record(summary.run_id, image_id)["status"] for image_id in image_ids]
    pipeline.close()
    _FakeBackend.fail_value = None
    assert _FakeBackend.loads == 1
    assert summary.completed == 2 and summary.failed == 1
    assert statuses == ["success", "failed", "success"]


def test_cancellation_between_images_and_failed_alignment_not_success(tmp_path):
    image_ids = _project(tmp_path)
    cancel = threading.Event()
    pipeline = MetricDepthPipeline(tmp_path, {BACKEND_DAV2: _FakeBackend})

    def progress(event):
        if event.stage == "saving_artifact" and "complete" in event.message:
            cancel.set()

    summary = pipeline.run_all(_config(), progress, cancel)
    pipeline.close()
    assert summary.cancelled and summary.completed == 1

    _FakeBackend.fail_value = 10
    pipeline = MetricDepthPipeline(tmp_path, {BACKEND_DAV2: _FakeBackend})
    failed = pipeline.run_image(image_ids[0], MetricDepthConfig(backend=BACKEND_DAV2, model_id_or_path="fake-local", recompute=True))
    record = pipeline.store.get_record(failed.run_id, image_ids[0])
    pipeline.close()
    _FakeBackend.fail_value = None
    assert failed.completed == 0 and failed.failed == 1
    assert record["status"] == "failed" and record["artifact_rel_path"] is None
