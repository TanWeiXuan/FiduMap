import numpy as np
import pytest

from map_builder.camera_models import PinholeRadTanCameraModel
from map_builder.geometry import SE3
from map_builder.gui.depth_3d_viewer_panel import (
    confidence_scalars,
    depth_to_world_point_cloud,
    orbit_camera,
    pan_camera,
    vtk_image_to_rgb,
    zoom_camera,
)
from map_builder.metric_depth.backends.prompt_depth_anything import PromptDepthAnythingBackend, adapt_sparse_prompt
from map_builder.metric_depth.models import MetricDepthConfig
from map_builder.metric_depth.alignment import verify_metric_prediction
from map_builder.metric_depth.models import MetricDepthArtifact, MetricDepthMetrics


def test_sparse_prompt_adaptation_keeps_missing_zero():
    depth = np.zeros((4, 4), dtype=np.float32); depth[1, 1] = 2.0
    mask = depth > 0
    adapted, adapted_mask = adapt_sparse_prompt(depth, mask, (8, 8))
    assert np.all(adapted[~adapted_mask] == 0)
    assert set(np.unique(adapted)).issubset({0.0, 2.0})
    downsampled, downsampled_mask = adapt_sparse_prompt(depth, mask, (2, 2))
    assert downsampled_mask.sum() == 1
    assert downsampled[downsampled_mask][0] == 2.0


def test_prompt_metric_verification_rejects_excessive_anchor_error():
    prompt = np.zeros((3, 3), dtype=np.float32); prompt[0, 0] = 2; prompt[2, 2] = 3
    mask = prompt > 0
    prediction = np.where(mask, prompt + 2.0, 1.0)
    ok, absolute, relative, _inliers, message = verify_metric_prediction(prediction, prompt, mask, 0.5, 0.2)
    assert not ok
    assert absolute == 2.0 and relative > 0.2
    assert "excessive" in message


def test_promptda_processor_receives_rgb_and_sparse_prompt_shapes():
    torch = pytest.importorskip("torch")

    class Processor:
        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return {"pixel_values": torch.zeros((1, 3, 4, 4)), "prompt_depth": torch.zeros((1, 1, 4, 4))}

        def post_process_depth_estimation(self, _outputs, target_sizes):
            self.target_sizes = target_sizes
            return [{"predicted_depth": torch.full(target_sizes[0], 2.0)}]

    class Model:
        def __call__(self, **inputs):
            self.inputs = inputs
            return object()

    backend = PromptDepthAnythingBackend()
    backend.processor, backend.model, backend.device = Processor(), Model(), "cpu"
    depth = np.zeros((6, 8), dtype=np.float32); depth[2, 3] = 2.0
    prediction = backend.predict_metric(np.zeros((6, 8, 3), dtype=np.uint8), depth, depth > 0, MetricDepthConfig(backend="prompt_depth_anything", inference_size=4))
    assert backend.processor.kwargs["prompt_depth"].shape == (4, 4)
    assert backend.processor.kwargs["images"].size == (8, 6)
    assert backend.processor.target_sizes == [(6, 8)]
    assert prediction.shape == (6, 8)


def test_depth_to_world_points_and_deterministic_decimation():
    camera = PinholeRadTanCameraModel(4, 3, 2, 2, 1.5, 1, 0, 0, 0, 0, 0)
    ranges = np.full((3, 4), 2.0, dtype=np.float32)
    artifact = MetricDepthArtifact(1, "fake", 4, 3, ranges.copy(), ranges, np.ones_like(ranges, bool), np.ones_like(ranges), np.zeros_like(ranges), np.zeros_like(ranges, bool), {}, MetricDepthMetrics(status="success"))
    a = depth_to_world_point_cloud(artifact, camera, SE3.identity(), maximum_points=5)
    b = depth_to_world_point_cloud(artifact, camera, SE3.identity(), maximum_points=5)
    assert a[0].shape == (5, 3)
    assert np.array_equal(a[0], b[0])
    assert confidence_scalars(np.array([0.0, 1.0])).tolist() == [[255, 0, 0], [0, 255, 0]]


def test_vtk_image_conversion_flips_bottom_up_rgb_rows():
    class Scalars:
        def GetNumberOfComponents(self):
            return 3

    class PointData:
        def __init__(self, scalars):
            self.scalars = scalars

        def GetScalars(self):
            return self.scalars

    class Image:
        scalars = Scalars()

        def GetDimensions(self):
            return 2, 2, 1

        def GetPointData(self):
            return PointData(self.scalars)

    bottom_up = np.array(
        [
            [255, 0, 0], [0, 255, 0],
            [0, 0, 255], [255, 255, 255],
        ],
        dtype=np.uint8,
    )
    rgb = vtk_image_to_rgb(Image(), array_converter=lambda _scalars: bottom_up)
    assert rgb.tolist() == [
        [[0, 0, 255], [255, 255, 255]],
        [[255, 0, 0], [0, 255, 0]],
    ]


def test_canvas_camera_orbit_pan_and_zoom_helpers():
    class Camera:
        def __init__(self):
            self.position = np.array([0.0, 0.0, 10.0])
            self.focal = np.zeros(3)
            self.parallel = False
            self.parallel_scale = 2.0
            self.azimuth = self.elevation = self.dolly = None

        def Azimuth(self, value): self.azimuth = value
        def Elevation(self, value): self.elevation = value
        def OrthogonalizeViewUp(self): self.orthogonalized = True
        def GetPosition(self): return self.position
        def GetFocalPoint(self): return self.focal
        def GetViewUp(self): return (0.0, 1.0, 0.0)
        def SetPosition(self, *value): self.position = np.array(value)
        def SetFocalPoint(self, *value): self.focal = np.array(value)
        def GetParallelProjection(self): return self.parallel
        def GetParallelScale(self): return self.parallel_scale
        def SetParallelScale(self, value): self.parallel_scale = value
        def GetViewAngle(self): return 30.0
        def Dolly(self, value): self.dolly = value

    camera = Camera()
    orbit_camera(camera, 10, -5)
    assert camera.azimuth == -4.0 and camera.elevation == -2.0 and camera.orthogonalized
    old_delta = camera.focal - camera.position
    pan_camera(camera, 20, -10, 640, 480)
    assert not np.allclose(camera.position, [0, 0, 10])
    assert np.allclose(camera.focal - camera.position, old_delta)
    zoom_camera(camera, 1.2)
    assert camera.dolly == 1.2
    camera.parallel = True
    zoom_camera(camera, 2.0)
    assert camera.parallel_scale == 1.0
