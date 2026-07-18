import numpy as np
import pytest

from map_builder.gui.depth_3d_viewer_panel import (
    _vtk_point_actor,
    confidence_scalars,
    orbit_camera,
    pan_camera,
    vtk_image_to_rgb,
    zoom_camera,
)


def test_confidence_scalars_use_red_to_green_ramp():
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


def test_vtk_point_actor_uses_one_vertex_cell_per_point():
    pytest.importorskip("vtkmodules")
    points = np.arange(15, dtype=np.float32).reshape(5, 3)
    colors = np.full((5, 3), 127, dtype=np.uint8)
    actor = _vtk_point_actor(points, colors, 2.0)
    poly = actor.GetMapper().GetInput()
    assert poly.GetNumberOfPoints() == 5
    assert poly.GetNumberOfVerts() == 5
