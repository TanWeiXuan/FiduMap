from pathlib import Path

from map_builder.project import DetectorRunConfig, MarkerDetection, ProjectStore


def test_project_store_persists_detections_and_marks_changed_image_stale(tmp_path: Path) -> None:
    store = ProjectStore.open(tmp_path)
    try:
        assert store.upsert_image_index_entry("a.jpg", 10, 100, 640, 480) == "new"
        images = store.list_images()
        assert len(images) == 1
        image = images[0]
        assert image.rel_path == "a.jpg"
        assert image.width == 640
        assert image.height == 480

        store.set_image_ignored(image.id, True)
        run_id = store.create_detector_run(
            DetectorRunConfig(detector_type="ArUco", dictionary_name="DICT_4X4_50"),
            opencv_version="test",
        )
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
    finally:
        store.close()

    reopened = ProjectStore.open(tmp_path)
    try:
        image = reopened.list_images()[0]
        assert image.ignored is True
        assert image.detection_status == "detected"
        assert image.marker_count == 1
        detections = reopened.get_detections_for_image(image.id)
        assert detections[0].marker_id == 7
        assert detections[0].corners[2] == [1.0, 1.0]

        assert reopened.upsert_image_index_entry("a.jpg", 11, 101) == "updated"
        changed = reopened.list_images()[0]
        assert changed.detection_status == "stale"
        assert changed.modified_since_detection is True
    finally:
        reopened.close()
