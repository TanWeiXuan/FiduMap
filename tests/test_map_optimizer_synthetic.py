from pathlib import Path

import numpy as np
import pytest

from map_builder.camera_models import PinholeRadTanCameraModel
from map_builder.geometry import SE3, marker_corners_for_detector_order
from map_builder.optimization import MapOptimizer
from map_builder.project import (
    BAConfig,
    DetectorRunConfig,
    MarkerDetection,
    ProjectStore,
    SeedCameraPose,
    SeedMarkerPose,
)


def test_map_optimizer_synthetic_reduces_reprojection_cost(tmp_path: Path) -> None:
    pytest.importorskip("pyceres")
    camera = PinholeRadTanCameraModel(800, 600, 600, 600, 400, 300, 0, 0, 0, 0, 0)
    marker_size = 0.2
    T_W_M0 = SE3.identity()
    T_W_M1 = SE3(np.eye(3), np.array([0.5, 0.0, 0.0]))
    camera_gt = {
        0: SE3(np.eye(3), np.array([0.0, 0.0, -1.3])),
        1: SE3(np.eye(3), np.array([0.2, 0.1, -1.4])),
        2: SE3(np.eye(3), np.array([-0.15, 0.05, -1.2])),
    }
    marker_gt = {0: T_W_M0, 1: T_W_M1}
    object_points = marker_corners_for_detector_order(marker_size)

    store = ProjectStore.open(tmp_path)
    try:
        store.set_marker_size_m(marker_size)
        store.set_anchor_marker_id(0)
        run_id = store.create_detector_run(DetectorRunConfig(detector_type="ArUco", dictionary_name="DICT_4X4_50"))
        for image_id, T_W_C in camera_gt.items():
            rel_path = f"image_{image_id}.png"
            store.upsert_image_index_entry(rel_path, 10 + image_id, 100 + image_id)
            db_image = [img for img in store.list_images() if img.rel_path == rel_path][0]
            detections = []
            for marker_id, T_W_M in marker_gt.items():
                corners = camera.project_many(T_W_C.inverse().transform_points(T_W_M.transform_points(object_points)))
                detections.append(
                    MarkerDetection(
                        marker_family="aruco",
                        dictionary_name="DICT_4X4_50",
                        marker_id=marker_id,
                        corners=corners.tolist(),
                        corner_refinement_method="CORNER_REFINE_SUBPIX",
                    )
                )
            store.replace_image_detections(db_image.id, run_id, detections)

        images = store.list_images()
        store.replace_seed_camera_poses(
            [
                SeedCameraPose(
                    image_id=img.id,
                    T_W_C=SE3(camera_gt[int(img.rel_path.split("_")[1].split(".")[0])].R, camera_gt[int(img.rel_path.split("_")[1].split(".")[0])].t + np.array([0.03, -0.02, 0.04])).to_json_dict(),
                )
                for img in images
            ]
        )
        store.replace_seed_marker_poses(
            [
                SeedMarkerPose(marker_id=0, T_W_M=SE3.identity().to_json_dict(), is_anchor=True),
                SeedMarkerPose(marker_id=1, T_W_M=SE3(np.eye(3), np.array([0.55, -0.03, 0.02])).to_json_dict()),
            ]
        )
        result = MapOptimizer(store, camera).run_ba(BAConfig(max_num_iterations=80))
        assert result.success
        assert result.summary.final_cost is not None
        assert result.summary.initial_cost is not None
        assert result.summary.final_cost < result.summary.initial_cost
        assert result.summary.mean_reprojection_error_px is not None
        assert result.summary.mean_reprojection_error_px < 1.0
        marker_1 = [pose for pose in result.optimized_marker_poses if pose.marker_id == 1][0]
        np.testing.assert_allclose(SE3.from_json_dict(marker_1.T_W_M).t, T_W_M1.t, atol=1e-2)
    finally:
        store.close()
