from pathlib import Path

from map_builder.geometry import SE3
from map_builder.project import DetectorRunConfig, MarkerDetection, PnPObservation, ProjectStore, SeedCameraPose, SeedMarkerPose


def test_initialization_settings_and_results_persist(tmp_path: Path) -> None:
    store = ProjectStore.open(tmp_path)
    try:
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
                    marker_id=7,
                    corners=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                    corner_refinement_method="CORNER_REFINE_SUBPIX",
                )
            ],
        )
        detection_id = store.list_detection_rows_for_initialization()[0]["detection_id"]
        store.set_marker_size_m(0.125)
        store.set_anchor_marker_id(7)
        T = SE3.identity().to_json_dict()
        store.replace_pnp_observations(
            [
                PnPObservation(
                    image_id=image.id,
                    detection_id=detection_id,
                    marker_id=7,
                    success=True,
                    T_C_M=T,
                    reprojection_error_px=0.5,
                )
            ]
        )
        store.replace_seed_camera_poses([SeedCameraPose(image_id=image.id, T_W_C=T, source_marker_id=7)])
        store.replace_seed_marker_poses([SeedMarkerPose(marker_id=7, T_W_M=T, is_anchor=True)])
        store.set_graph_diagnostics({"anchor_marker_exists": True, "disconnected_markers": []})
    finally:
        store.close()

    reopened = ProjectStore.open(tmp_path)
    try:
        assert reopened.get_marker_size_m() == 0.125
        assert reopened.get_anchor_marker_id() == 7
        assert len(reopened.list_pnp_observations(success_only=True)) == 1
        assert len(reopened.get_seed_camera_poses()) == 1
        assert reopened.get_seed_marker_poses()[0].is_anchor is True
        assert reopened.get_graph_diagnostics()["anchor_marker_exists"] is True
    finally:
        reopened.close()


def test_default_anchor_uses_smallest_observed_marker_id(tmp_path: Path) -> None:
    store = ProjectStore.open(tmp_path)
    try:
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
                    marker_id=9,
                    corners=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                    corner_refinement_method="CORNER_REFINE_SUBPIX",
                ),
                MarkerDetection(
                    marker_family="aruco",
                    dictionary_name="DICT_4X4_50",
                    marker_id=3,
                    corners=[[2.0, 2.0], [3.0, 2.0], [3.0, 3.0], [2.0, 3.0]],
                    corner_refinement_method="CORNER_REFINE_SUBPIX",
                ),
            ],
        )
        detection_rows = store.list_detection_rows_for_initialization()
        detection_by_marker = {row["marker_id"]: row["detection_id"] for row in detection_rows}
        T = SE3.identity().to_json_dict()
        store.replace_pnp_observations(
            [
                PnPObservation(
                    image_id=image.id,
                    detection_id=detection_by_marker[9],
                    marker_id=9,
                    success=True,
                    T_C_M=T,
                ),
                PnPObservation(
                    image_id=image.id,
                    detection_id=detection_by_marker[3],
                    marker_id=3,
                    success=True,
                    T_C_M=T,
                ),
            ]
        )
        assert store.get_configured_anchor_marker_id() is None
        assert store.get_anchor_marker_id() == 3
    finally:
        store.close()
