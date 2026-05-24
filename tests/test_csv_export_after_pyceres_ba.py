from pathlib import Path

import pytest

from map_builder.export import export_optimized_marker_map_csv
from map_builder.geometry import SE3
from map_builder.project import BAConfig, OptimizedMarkerPose, ProjectStore


def test_csv_export_after_successful_pyceres_ba_uses_export_corner_order(tmp_path: Path) -> None:
    pytest.importorskip("pyceres")
    store = ProjectStore.open(tmp_path)
    try:
        store.set_marker_size_m(2.0)
        run_id = store.create_ba_run(BAConfig())
        store.replace_optimized_marker_poses(
            run_id,
            [OptimizedMarkerPose(marker_id=0, ba_run_id=run_id, T_W_M=SE3.identity().to_json_dict(), is_anchor=True)],
        )
        store.complete_ba_run(run_id, success=True, num_observations=0, num_corners=0, solver_report="synthetic")
        output = tmp_path / "map.csv"
        assert export_optimized_marker_map_csv(store, output) == 4
        lines = output.read_text(encoding="utf-8").strip().splitlines()
        assert lines[1] == "0,-1.000000,-1.000000,0.000000"
        assert lines[2] == "1,-1.000000,1.000000,0.000000"
        assert lines[3] == "2,1.000000,1.000000,0.000000"
        assert lines[4] == "3,1.000000,-1.000000,0.000000"
    finally:
        store.close()
