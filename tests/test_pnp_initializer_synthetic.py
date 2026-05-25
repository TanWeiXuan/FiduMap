from pathlib import Path

import numpy as np
import pytest

from map_builder.camera_models import PinholeRadTanCameraModel
from map_builder.geometry import SE3, marker_corners_for_detector_order
from map_builder.initialization import PnPInitializer
from map_builder.project import DetectorRunConfig, MarkerDetection, ProjectStore


def test_pnp_initializer_recovers_synthetic_marker_pose(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    camera = PinholeRadTanCameraModel(
        image_width=640,
        image_height=480,
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
        k1=0.0,
        k2=0.0,
        p1=0.0,
        p2=0.0,
        k3=0.0,
    )
    marker_size = 0.2
    object_points = marker_corners_for_detector_order(marker_size)
    T_C_M_gt = SE3(np.eye(3), np.array([0.02, -0.01, 1.5]))
    image_points = camera.project_many(T_C_M_gt.transform_points(object_points))

    store = ProjectStore.open(tmp_path)
    try:
        store.upsert_image_index_entry("synthetic.png", 10, 100)
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
                    corners=image_points.tolist(),
                    corner_refinement_method="CORNER_REFINE_SUBPIX",
                )
            ],
        )
        observations = PnPInitializer(tmp_path, store, camera, marker_size).run()
        assert len(observations) == 1
        assert observations[0].success
        assert observations[0].reprojection_error_px is not None
        assert observations[0].reprojection_error_px < 1e-5
        T_C_M_est = SE3.from_json_dict(observations[0].T_C_M)  # type: ignore[arg-type]
        np.testing.assert_allclose(T_C_M_est.t, T_C_M_gt.t, atol=1e-5)
        np.testing.assert_allclose(T_C_M_est.R, T_C_M_gt.R, atol=1e-5)
    finally:
        store.close()
