from types import SimpleNamespace

import numpy as np

from map_builder.camera_models import OmniRadTanCameraModel, PinholeRadTanCameraModel
from map_builder.geometry import SE3
from map_builder.metric_depth.anchor_builder import dense_track_anchors, marker_surface_anchors, rasterize_anchors
from map_builder.metric_depth.geometry import z_depth_to_range
from map_builder.metric_depth.models import MetricAnchor
from map_builder.project import MarkerDetection, OptimizedMarkerPose


def _camera(width=5, height=5):
    return PinholeRadTanCameraModel(width, height, 2.0, 2.0, 2.0, 2.0, 0, 0, 0, 0, 0)


def test_z_depth_to_range():
    camera = _camera()
    z = np.full((5, 5), 2.0, dtype=np.float32)
    ranges, _valid = z_depth_to_range(z, camera)
    assert np.isclose(ranges[2, 2], 2.0)
    assert ranges[0, 0] > 2.0


def test_omni_forward_round_trip_and_nonforward_invalidation():
    camera = OmniRadTanCameraModel(5, 5, 2, 2, 2, 2, 1.0, 0, 0, 0, 0, 0)
    z = np.ones((5, 5), dtype=np.float32)
    ranges, valid = z_depth_to_range(z, camera)
    assert valid[2, 2]
    assert not valid[0, 0]
    assert ranges[0, 0] == 0


def test_dense_observation_becomes_metric_anchor_and_behind_camera_is_rejected():
    points = [
        {"track_id": 7, "x": 0.0, "y": 0.0, "z": 2.0, "is_active": 1, "num_observations": 3},
        {"track_id": 8, "x": 0.0, "y": 0.0, "z": -1.0, "is_active": 1},
    ]
    observations = [SimpleNamespace(track_id=7, image_id=3, x=12.0, y=8.0), SimpleNamespace(track_id=8, image_id=3, x=1.0, y=1.0)]
    tracks = [SimpleNamespace(id=7, status="active", num_observations=3), SimpleNamespace(id=8, status="active")]
    anchors = dense_track_anchors(3, SE3.identity(), points, observations, tracks)
    assert len(anchors) == 1
    assert anchors[0].z_depth_m == 2.0
    assert anchors[0].range_m == 2.0
    assert anchors[0].provenance == "dense_track"


def test_visible_marker_square_rasterizes_exact_plane_depth():
    camera = PinholeRadTanCameraModel(9, 9, 8, 8, 4, 4, 0, 0, 0, 0, 0)
    marker_pose = SE3(np.eye(3), np.array([0.0, 0.0, 2.0]))
    detection = MarkerDetection("aruco", "DICT_6X6_250", 4, [[2, 2], [6, 2], [6, 6], [2, 6]], "none")
    pose = OptimizedMarkerPose(4, 1, marker_pose.to_json_dict())
    anchors = marker_surface_anchors([detection], [pose], SE3.identity(), camera, 1.0, 9, 9)
    raster = rasterize_anchors(anchors, 9, 9)
    assert raster.mask[4, 4]
    assert np.isclose(raster.depth_z_m[4, 4], 2.0)
    assert np.all(raster.provenance[raster.mask] == 2)


def test_anchor_collision_priority_and_spatial_coverage():
    anchors = [
        MetricAnchor(1, 1, 4.0, 4.0, 0.9, "dense_track"),
        MetricAnchor(1, 1, 6.0, 6.0, 1.0, "marker_surface"),
        MetricAnchor(6, 6, 3.0, 3.0, 0.8, "dense_track"),
    ]
    raster = rasterize_anchors(anchors, 8, 8)
    assert raster.depth_z_m[1, 1] == 6.0
    assert raster.pixel_count == 2
    assert raster.occupied_grid_cells == 2
    assert raster.spatial_coverage == 2 / 16
