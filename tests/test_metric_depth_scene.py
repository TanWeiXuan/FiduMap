import cv2
import numpy as np

from map_builder.camera_models import PinholeRadTanCameraModel
from map_builder.geometry import SE3
from map_builder.metric_depth.depth_scene import DepthSceneBuilder, fair_point_budgets, sample_depth_cloud_layer
from map_builder.metric_depth.models import BACKEND_DAV2, MetricDepthArtifact, MetricDepthMetrics
from map_builder.metric_depth.store import MetricDepthStore
from map_builder.project import BAConfig, OptimizedCameraPose, ProjectStore


class _CountingCamera:
    def __init__(self):
        self.calls: list[np.ndarray] = []

    def unproject_many(self, pixels):
        values = np.asarray(pixels, dtype=float)
        self.calls.append(values.copy())
        return np.tile(np.array([[0.0, 0.0, 1.0]]), (len(values), 1))


def _artifact(image_id: int, width: int = 5, height: int = 4) -> MetricDepthArtifact:
    ranges = np.full((height, width), 2.0, dtype=np.float32)
    valid = np.ones((height, width), dtype=bool)
    confidence = np.linspace(0.0, 1.0, width * height, dtype=np.float32).reshape(height, width)
    return MetricDepthArtifact(
        image_id,
        BACKEND_DAV2,
        width,
        height,
        ranges.copy(),
        ranges,
        valid,
        confidence,
        {},
        MetricDepthMetrics(valid_output_fraction=1.0, status="success"),
    )


def test_fair_point_budget_is_global_and_deterministic():
    assert fair_point_budgets([7, 3, 9], 10) == {7: 4, 3: 3, 9: 3}
    assert sum(fair_point_budgets([1, 2, 3], 2).values()) == 2
    assert fair_point_budgets([], 200_000) == {}


def test_sampling_happens_before_unprojection_and_applies_world_pose():
    artifact = _artifact(4)
    camera = _CountingCamera()
    pose = SE3(np.eye(3), np.array([1.0, 2.0, 3.0]))
    rgb = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    first = sample_depth_cloud_layer(artifact, camera, pose, rgb, 0.0, 5, 12)
    second = sample_depth_cloud_layer(artifact, camera, pose, rgb, 0.0, 5, 12)
    assert len(camera.calls[0]) == 5
    assert first.point_count == 5
    assert np.allclose(first.points_world, np.tile([1.0, 2.0, 5.0], (5, 1)))
    assert np.array_equal(first.points_world, second.points_world)
    assert np.array_equal(first.rgb, second.rgb)


def _scene_project(tmp_path):
    camera = PinholeRadTanCameraModel(5, 4, 4, 4, 2, 1.5, 0, 0, 0, 0, 0)
    camera_path = tmp_path / "camera.xml"
    camera.save_xml(camera_path)
    with ProjectStore.open(tmp_path) as store:
        store.set_camera_config_path(camera_path)
        for index in range(2):
            image_path = tmp_path / f"{index}.png"
            cv2.imwrite(str(image_path), np.full((4, 5, 3), 30 + index, dtype=np.uint8))
            stat = image_path.stat()
            store.upsert_image_index_entry(image_path.name, stat.st_size, stat.st_mtime_ns, 5, 4)
        images = store.list_images()
        ba_run = store.create_ba_run(BAConfig())
        store.complete_ba_run(ba_run, True)
        store.replace_optimized_camera_poses(
            ba_run,
            [
                OptimizedCameraPose(images[0].id, ba_run, SE3.identity().to_json_dict()),
                OptimizedCameraPose(images[1].id, ba_run, SE3(np.eye(3), np.array([1.0, 0.0, 0.0])).to_json_dict()),
            ],
        )
    with MetricDepthStore.open(tmp_path) as depth_store:
        run = depth_store.create_run(BACKEND_DAV2, "local", {}, ba_run, "points=0")
        depth_store.save_artifact_atomic(run, _artifact(images[0].id))
    return images


def test_scene_builder_skips_unavailable_maps_and_reuses_larger_cached_layer(tmp_path):
    images = _scene_project(tmp_path)
    builder = DepthSceneBuilder(tmp_path)
    first = builder.build([images[0].id, images[1].id], images[0].id, 0.0, 8)
    assert first.selected_count == 2
    assert len(first.layers) == 1
    assert first.layers[0].point_count == 8  # Unavailable selections do not consume the scene budget.
    assert images[1].id in first.skipped
    assert first.primary_artifact is not None
    cached = next(iter(builder._cache.values()))
    second = builder.build([images[0].id], images[0].id, 0.0, 2)
    assert second.layers[0].point_count == 2
    assert next(iter(builder._cache.values())) is cached
