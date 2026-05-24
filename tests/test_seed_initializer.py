import numpy as np

from map_builder.geometry import SE3
from map_builder.initialization import build_observation_graph, initialize_seed_poses
from map_builder.project import PnPObservation


def test_seed_initializer_propagates_from_anchor_marker() -> None:
    T_C0_M0 = SE3(np.eye(3), np.array([0.0, 0.0, 1.0]))
    T_C0_M1 = SE3(np.eye(3), np.array([1.0, 0.0, 1.0]))
    observations = [
        PnPObservation(
            image_id=0,
            detection_id=0,
            marker_id=0,
            success=True,
            T_C_M=T_C0_M0.to_json_dict(),
            reprojection_error_px=0.1,
        ),
        PnPObservation(
            image_id=0,
            detection_id=1,
            marker_id=1,
            success=True,
            T_C_M=T_C0_M1.to_json_dict(),
            reprojection_error_px=0.2,
        ),
    ]
    graph = build_observation_graph(observations, anchor_marker_id=0)
    result = initialize_seed_poses(graph, anchor_marker_id=0)
    assert len(result.camera_poses) == 1
    assert len(result.marker_poses) == 2
    camera_pose = SE3.from_json_dict(result.camera_poses[0].T_W_C)
    marker_1_pose = SE3.from_json_dict([pose for pose in result.marker_poses if pose.marker_id == 1][0].T_W_M)
    np.testing.assert_allclose(camera_pose.t, [0.0, 0.0, -1.0])
    np.testing.assert_allclose(marker_1_pose.t, [1.0, 0.0, 0.0])
    assert result.disconnected_markers == []


def test_seed_initializer_reports_disconnected_markers() -> None:
    T = SE3.identity().to_json_dict()
    graph = build_observation_graph(
        [
            PnPObservation(image_id=0, detection_id=0, marker_id=0, success=True, T_C_M=T),
            PnPObservation(image_id=1, detection_id=1, marker_id=2, success=True, T_C_M=T),
        ],
        anchor_marker_id=0,
    )
    result = initialize_seed_poses(graph, anchor_marker_id=0)
    assert result.disconnected_markers == [2]
