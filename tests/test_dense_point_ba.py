import numpy as np
import pytest

from map_builder.camera_models import PinholeRadTanCameraModel
from map_builder.dense_reconstruction import point_ba
from map_builder.dense_reconstruction.dense_store import DenseReconstructionStore
from map_builder.dense_reconstruction.models import (
    DenseBAConfig,
    DensePointRecord,
    TrackObservationRecord,
    TrackRecord,
)


class _FakeCostFunction:
    def __init__(self):
        self.num_residuals = None
        self.parameter_block_sizes = None

    def set_num_residuals(self, value):
        self.num_residuals = value

    def set_parameter_block_sizes(self, value):
        self.parameter_block_sizes = value


class _FakeProblem:
    def __init__(self):
        self.blocks = []

    def add_residual_block(self, cost, loss, parameters):
        self.blocks.append((cost, loss, parameters))


class _FakeHuberLoss:
    def __init__(self, scale):
        self.scale = scale


class _FakeSolverOptions:
    def __init__(self):
        self.max_num_iterations = 0
        self.minimizer_progress_to_stdout = True


class _FakePyCeres:
    CostFunction = _FakeCostFunction
    Problem = _FakeProblem
    HuberLoss = _FakeHuberLoss
    SolverOptions = _FakeSolverOptions
    usable = True
    update = np.array([0.1, 0.0, 0.0])

    class SolverSummary:
        initial_cost = 10.0
        final_cost = 1.0

        def IsSolutionUsable(self):
            return _FakePyCeres.usable

    @staticmethod
    def solve(_options, problem, _summary):
        seen = set()
        for cost, _loss, parameters in problem.blocks:
            params = parameters[0]
            if id(params) not in seen:
                params[:] = params + _FakePyCeres.update
                seen.add(id(params))
            cost.compute_residual(params)


def _camera():
    return PinholeRadTanCameraModel(640, 480, 400, 400, 320, 240, 0, 0, 0, 0, 0)


def _seed_track_backed_point(store):
    camera = _camera()
    point = np.array([0.0, 0.0, 5.0])
    poses = {
        1: {"R": np.eye(3).tolist(), "t": [0.0, 0.0, 0.0]},
        2: {"R": np.eye(3).tolist(), "t": [1.0, 0.0, 0.0]},
    }
    observations = []
    for image_id, pose in poses.items():
        R = np.asarray(pose["R"])
        C = np.asarray(pose["t"])
        pixel = camera.project(R.T @ (point - C))
        observations.append(TrackObservationRecord(0, image_id, 0, float(pixel[0]), float(pixel[1])))
    store.replace_tracks_and_points(
        [
            (
                TrackRecord(status="active", num_observations=2, num_images=2, x=0, y=0, z=5),
                observations,
                DensePointRecord(x=0, y=0, z=5, num_observations=2),
            )
        ]
    )
    return poses, camera


@pytest.mark.parametrize("mode", ["points_and_cameras", "full", "unknown"])
def test_dense_ba_rejects_every_mode_except_points_only(tmp_path, mode):
    store = DenseReconstructionStore.open(tmp_path)

    with pytest.raises(NotImplementedError, match="supports only points_only"):
        point_ba.run_dense_point_ba(store, {}, _camera(), DenseBAConfig(mode=mode))


def test_dense_ba_refuses_active_merged_points(tmp_path, monkeypatch):
    store = DenseReconstructionStore.open(tmp_path)
    poses, camera = _seed_track_backed_point(store)
    active = store.list_active_dense_points()[0]
    store.replace_active_dense_points(
        [DensePointRecord(track_id=int(active["track_id"]), x=0, y=0, z=5)],
        source="merged",
    )
    monkeypatch.setattr(point_ba.importlib, "import_module", lambda _name: _FakePyCeres)

    with pytest.raises(RuntimeError, match="cannot run on an active merged point set"):
        point_ba.run_dense_point_ba(store, poses, camera, DenseBAConfig())
    run = store.conn.execute("SELECT * FROM dense_ba_runs ORDER BY id DESC LIMIT 1").fetchone()
    assert run["success"] == 0
    assert "retriangulation restores track-backed points" in run["error_message"]


def test_unusable_dense_ba_solution_does_not_update_coordinates(tmp_path, monkeypatch):
    store = DenseReconstructionStore.open(tmp_path)
    poses, camera = _seed_track_backed_point(store)
    before = store.list_active_dense_points()[0]
    before_xyz = np.array([before["x"], before["y"], before["z"]])
    _FakePyCeres.usable = False
    monkeypatch.setattr(point_ba.importlib, "import_module", lambda _name: _FakePyCeres)

    with pytest.raises(RuntimeError, match="unusable solver result"):
        point_ba.run_dense_point_ba(store, poses, camera, DenseBAConfig())

    after = store.list_active_dense_points()[0]
    assert np.allclose([after["x"], after["y"], after["z"]], before_xyz)
    assert after["source"] == "triangulated"
    run = store.conn.execute("SELECT * FROM dense_ba_runs ORDER BY id DESC LIMIT 1").fetchone()
    assert run["success"] == 0
    assert "stored point coordinates were not updated" in run["error_message"]


def test_usable_points_only_dense_ba_updates_track_backed_point(tmp_path, monkeypatch):
    store = DenseReconstructionStore.open(tmp_path)
    poses, camera = _seed_track_backed_point(store)
    _FakePyCeres.usable = True
    monkeypatch.setattr(point_ba.importlib, "import_module", lambda _name: _FakePyCeres)

    result = point_ba.run_dense_point_ba(
        store,
        poses,
        camera,
        DenseBAConfig(
            mode="points_only",
            max_mean_reprojection_error_px=10.0,
            max_reprojection_error_px=10.0,
        ),
    )

    after = store.list_active_dense_points()[0]
    assert np.allclose([after["x"], after["y"], after["z"]], [0.1, 0.0, 5.0])
    assert after["source"] == "dense_ba"
    assert after["mean_reprojection_error_px"] == pytest.approx(8.0)
    track = store.list_tracks()[0]
    assert np.allclose([track.x, track.y, track.z], [0.1, 0.0, 5.0])
    assert track.status == "active"
    assert track.mean_reprojection_error_px == pytest.approx(8.0)
    assert result.success == 1
    run = store.conn.execute("SELECT * FROM dense_ba_runs ORDER BY id DESC LIMIT 1").fetchone()
    assert run["success"] == 1


def test_dense_ba_deactivates_point_that_fails_post_solve_quality_checks(tmp_path, monkeypatch):
    store = DenseReconstructionStore.open(tmp_path)
    poses, camera = _seed_track_backed_point(store)
    _FakePyCeres.usable = True
    monkeypatch.setattr(point_ba.importlib, "import_module", lambda _name: _FakePyCeres)

    result = point_ba.run_dense_point_ba(store, poses, camera, DenseBAConfig())

    assert store.list_active_dense_points() == []
    row = store.conn.execute("SELECT * FROM dense_points ORDER BY id DESC LIMIT 1").fetchone()
    assert row["source"] == "dense_ba_rejected"
    assert row["is_active"] == 0
    assert row["max_reprojection_error_px"] == pytest.approx(8.0)
    track = store.list_tracks()[0]
    assert track.status == "rejected"
    assert track.max_reprojection_error_px == pytest.approx(8.0)
    assert result.success == 0
    assert result.failed == 1
