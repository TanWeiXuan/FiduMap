import csv

import numpy as np
import pytest

from absolute_pose_solver import (
    AbsolutePoseSolver,
    CameraConfig,
    DETECTOR_TO_MAP_CORNER,
    load_marker_map_csv,
)
from map_builder.camera_models import PinholeRadTanCameraModel
from map_builder.geometry import SE3
from map_builder.project.models import MarkerDetection


def _write_rows(path, rows, fieldnames=("id", "x", "y", "z")) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _marker_rows(marker_id: int = 7):
    return [
        {"id": (marker_id << 2) | corner, "x": corner, "y": corner + 10, "z": corner + 20}
        for corner in range(4)
    ]


def _camera() -> PinholeRadTanCameraModel:
    return PinholeRadTanCameraModel(
        640, 480, 400.0, 400.0, 320.0, 240.0, 0.0, 0.0, 0.0, 0.0, 0.0
    )


def test_load_marker_map_decodes_and_groups_corners(tmp_path) -> None:
    path = tmp_path / "map.csv"
    _write_rows(path, _marker_rows(7) + _marker_rows(2))

    marker_map = load_marker_map_csv(path)

    assert set(marker_map) == {2, 7}
    assert marker_map[7].shape == (4, 3)
    np.testing.assert_array_equal(marker_map[7][:, 0], np.arange(4))


def test_load_marker_map_rejects_duplicate_corner(tmp_path) -> None:
    path = tmp_path / "duplicate.csv"
    rows = _marker_rows()
    _write_rows(path, rows + [rows[0]])
    with pytest.raises(ValueError, match="Duplicate marker corner"):
        load_marker_map_csv(path)


def test_load_marker_map_rejects_missing_corner(tmp_path) -> None:
    path = tmp_path / "missing.csv"
    _write_rows(path, _marker_rows()[:-1])
    with pytest.raises(ValueError, match="missing corner indices"):
        load_marker_map_csv(path)


@pytest.mark.parametrize(
    ("rows", "fieldnames", "message"),
    [
        ([{"id": 0, "x": 0, "y": 0}], ("id", "x", "y"), "must contain"),
        ([{"id": "bad", "x": 0, "y": 0, "z": 0}], ("id", "x", "y", "z"), "Invalid point ID"),
        ([{"id": 0, "x": "nan", "y": 0, "z": 0}], ("id", "x", "y", "z"), "must be finite"),
    ],
)
def test_load_marker_map_rejects_malformed_rows(tmp_path, rows, fieldnames, message) -> None:
    path = tmp_path / "bad.csv"
    _write_rows(path, rows, fieldnames)
    with pytest.raises(ValueError, match=message):
        load_marker_map_csv(path)


def test_detector_corners_are_remapped_to_export_order(tmp_path) -> None:
    assert DETECTOR_TO_MAP_CORNER.tolist() == [1, 2, 3, 0]
    path = tmp_path / "map.csv"
    _write_rows(path, _marker_rows())
    solver = AbsolutePoseSolver(path, {"camera": CameraConfig(_camera(), SE3.identity())})
    detection = MarkerDetection(
        marker_family="aruco",
        dictionary_name="DICT_6X6_250",
        marker_id=7,
        corners=[[310, 230], [330, 230], [330, 250], [310, 250]],
        corner_refinement_method="CORNER_REFINE_SUBPIX",
    )

    _, points_W, _, _, _, _ = solver._build_correspondences({"camera": [detection]})

    np.testing.assert_array_equal(points_W[:, 0], np.array([1, 2, 3, 0]))
