from map_builder.geometry import SE3
from map_builder.gui.map_3d_viewer_panel import _camera_pose_key, _marker_pose_key
from map_builder.project import OptimizedCameraPose, OptimizedMarkerPose


def test_render_key_helpers_accept_optimized_pose_records() -> None:
    T = SE3.identity().to_json_dict()
    camera = OptimizedCameraPose(image_id=1, ba_run_id=2, T_W_C=T)
    marker = OptimizedMarkerPose(marker_id=3, ba_run_id=2, T_W_M=T, is_anchor=False)

    assert _camera_pose_key(camera)[0] == 1
    assert _camera_pose_key(camera)[2] is None
    assert _camera_pose_key(camera)[4] == 2
    assert _marker_pose_key(marker)[0] == 3
    assert _marker_pose_key(marker)[2] is None
    assert _marker_pose_key(marker)[5] == 2
