from pathlib import Path

import pytest

from map_builder.export import write_marker_map_csv
from map_builder.geometry import SE3
from map_builder.project import OptimizedMarkerPose


def test_csv_exporter_writes_expected_ids_and_corner_order(tmp_path: Path) -> None:
    output = tmp_path / "map.csv"
    poses = [
        OptimizedMarkerPose(marker_id=0, ba_run_id=1, T_W_M=SE3.identity().to_json_dict(), is_anchor=True),
        OptimizedMarkerPose(marker_id=1, ba_run_id=1, T_W_M=SE3.identity().to_json_dict()),
    ]
    count = write_marker_map_csv(output, poses, marker_size_m=2.0)
    assert count == 8
    lines = output.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "id,x,y,z"
    assert lines[1] == "0,-1.000000,-1.000000,0.000000"
    assert lines[2] == "1,-1.000000,1.000000,0.000000"
    assert lines[3] == "2,1.000000,1.000000,0.000000"
    assert lines[4] == "3,1.000000,-1.000000,0.000000"
    assert [line.split(",")[0] for line in lines[5:9]] == ["4", "5", "6", "7"]


def test_csv_exporter_rejects_marker_id_too_large(tmp_path: Path) -> None:
    output = tmp_path / "map.csv"
    poses = [OptimizedMarkerPose(marker_id=1 << 30, ba_run_id=1, T_W_M=SE3.identity().to_json_dict())]
    with pytest.raises(ValueError):
        write_marker_map_csv(output, poses, marker_size_m=1.0)
