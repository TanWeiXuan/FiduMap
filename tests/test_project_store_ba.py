from pathlib import Path

from map_builder.geometry import SE3
from map_builder.project import (
    BAConfig,
    OptimizedCameraPose,
    OptimizedMarkerPose,
    ProjectStore,
    ReprojectionErrorRecord,
)


def test_project_store_ba_results_persist(tmp_path: Path) -> None:
    store = ProjectStore.open(tmp_path)
    try:
        store.set_ba_config(
            BAConfig(
                robust_loss_type="Cauchy",
                robust_loss_scale_px=2.0,
                corner_outlier_threshold_px=11.0,
                marker_observation_outlier_threshold_px=6.0,
                run_outlier_second_pass=False,
            )
        )
        store.upsert_image_index_entry("image.png", 10, 100)
        image = store.list_images()[0]
        ba_run_id = store.create_ba_run(BAConfig())
        T = SE3.identity().to_json_dict()
        store.replace_optimized_camera_poses(
            ba_run_id,
            [OptimizedCameraPose(image_id=image.id, ba_run_id=ba_run_id, T_W_C=T)],
        )
        store.replace_optimized_marker_poses(
            ba_run_id,
            [OptimizedMarkerPose(marker_id=0, ba_run_id=ba_run_id, T_W_M=T, is_anchor=True)],
        )
        store.replace_reprojection_errors(
            ba_run_id,
            [
                ReprojectionErrorRecord(
                    ba_run_id=ba_run_id,
                    image_id=image.id,
                    marker_id=0,
                    corner_index_detector_order=0,
                    error_px=0.25,
                    residual_x_px=0.1,
                    residual_y_px=0.2,
                )
            ],
        )
        store.complete_ba_run(
            ba_run_id,
            success=True,
            num_iterations=3,
            initial_cost=10.0,
            final_cost=1.0,
            mean_reprojection_error_px=0.25,
            median_reprojection_error_px=0.25,
            max_reprojection_error_px=0.25,
            num_observations=1,
            num_corners=1,
        )
    finally:
        store.close()

    reopened = ProjectStore.open(tmp_path)
    try:
        latest = reopened.get_latest_successful_ba_run_id()
        assert latest == ba_run_id
        assert reopened.get_ba_config().robust_loss_type == "Cauchy"
        assert reopened.get_ba_config().run_outlier_second_pass is False
        assert reopened.get_ba_summary().success is True  # type: ignore[union-attr]
        assert len(reopened.get_optimized_marker_poses()) == 1
        assert len(reopened.get_optimized_camera_poses()) == 1
        assert reopened.get_reprojection_errors()[0].error_px == 0.25
    finally:
        reopened.close()
