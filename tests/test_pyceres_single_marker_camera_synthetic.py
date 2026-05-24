from pathlib import Path

import numpy as np
import pytest

from map_builder.camera_models import PinholeRadTanCameraModel
from map_builder.geometry import SE3, marker_corners_for_detector_order
from map_builder.optimization import MapOptimizer
from map_builder.project import BAConfig, DetectorRunConfig, MarkerDetection, ProjectStore, SeedCameraPose, SeedMarkerPose


def test_pyceres_single_marker_camera_synthetic_reduces_error_and_keeps_anchor_fixed(tmp_path: Path) -> None:
    pytest.importorskip("pyceres")
    camera = PinholeRadTanCameraModel(800, 600, 600, 600, 400, 300, 0, 0, 0, 0, 0)
    marker_size = 0.2
    T_W_M0 = SE3.identity()
    T_W_C_gt = SE3(np.eye(3), np.array([0.05, -0.02, -1.4]))
    object_points = marker_corners_for_detector_order(marker_size)
    corners = camera.project_many(T_W_C_gt.inverse().transform_points(T_W_M0.transform_points(object_points)))

    store = ProjectStore.open(tmp_path)
    try:
        store.set_marker_size_m(marker_size)
        store.set_anchor_marker_id(0)
        store.upsert_image_index_entry("image.png", 10, 100)
        image = store.list_images()[0]
        run_id = store.create_detector_run(DetectorRunConfig(detector_type="ArUco", dictionary_name="DICT_4X4_50"))
        store.replace_image_detections(
            image.id,
            run_id,
            [
                MarkerDetection(
                    marker_family="aruco",
                    dictionary_name="DICT_4X4_50",
                    marker_id=0,
                    corners=corners.tolist(),
                    corner_refinement_method="CORNER_REFINE_SUBPIX",
                )
            ],
        )
        T_seed = SE3(np.eye(3), T_W_C_gt.t + np.array([0.08, -0.04, 0.07]))
        initial_error = np.linalg.norm(T_seed.t - T_W_C_gt.t)
        store.replace_seed_camera_poses([SeedCameraPose(image_id=image.id, T_W_C=T_seed.to_json_dict())])
        store.replace_seed_marker_poses([SeedMarkerPose(marker_id=0, T_W_M=SE3.identity().to_json_dict(), is_anchor=True)])
        result = MapOptimizer(store, camera).run_ba(BAConfig(max_num_iterations=80, run_outlier_second_pass=False))
        final_camera = SE3.from_json_dict(result.optimized_camera_poses[0].T_W_C)
        final_marker = SE3.from_json_dict(result.optimized_marker_poses[0].T_W_M)
        assert result.success
        assert result.summary.backend_name == "pyceres"
        assert np.linalg.norm(final_camera.t - T_W_C_gt.t) < initial_error
        np.testing.assert_allclose(final_marker.as_matrix(), SE3.identity().as_matrix(), atol=1e-12)
    finally:
        store.close()
