from types import SimpleNamespace

from map_builder.dense_reconstruction import dense_pipeline
from map_builder.dense_reconstruction.dense_pipeline import DensePipeline
from map_builder.dense_reconstruction.models import DenseReconstructionConfig, DenseStageSummary


class _CountsStore:
    def __init__(self):
        self.counts = {
            "feature_images": 0,
            "pairs": 0,
            "matches": 0,
            "inliers": 0,
            "points": 0,
        }

    def dense_counts(self):
        return dict(self.counts)


def _stage(store, order, name, updates, success=None):
    def run(_config, progress):
        order.append(name)
        progress(f"{name} progress")
        store.counts.update(updates)
        stage_success = max(updates.values(), default=1) if success is None else success
        return DenseStageSummary(stage=name, success=stage_success, details=f"{name} done")

    return run


def test_run_all_executes_stages_in_order_and_skips_unavailable_ba(monkeypatch):
    pipeline = DensePipeline.__new__(DensePipeline)
    store = _CountsStore()
    pipeline.store = store
    order = []
    pipeline.extract_features = _stage(store, order, "features", {"feature_images": 3})
    pipeline.build_frame_pairs = _stage(store, order, "pairs", {"pairs": 2})
    pipeline.match_frame_pairs = _stage(store, order, "matching", {"matches": 20})
    pipeline.filter_matches = _stage(store, order, "filter", {"inliers": 12})
    pipeline.build_tracks_and_triangulate = _stage(store, order, "tracks", {"points": 4})
    pipeline.run_dense_ba = _stage(store, order, "ba", {"points": 4})
    pipeline.merge_duplicates = _stage(store, order, "merge", {"points": 3}, success=1)
    monkeypatch.setattr(
        dense_pipeline,
        "check_dense_ba_availability",
        lambda: SimpleNamespace(available=False, details="pyceres unavailable"),
    )
    progress = []

    summary = pipeline.run_all(DenseReconstructionConfig(), progress.append)

    assert summary.success is True
    assert order == ["features", "pairs", "matching", "filter", "tracks", "merge"]
    assert summary.skipped_stages == ["dense_ba"]
    assert "3 active point(s)" in summary.details
    assert any("Dense BA skipped" in message for message in progress)


def test_run_all_stops_when_matching_produces_no_matches(monkeypatch):
    pipeline = DensePipeline.__new__(DensePipeline)
    store = _CountsStore()
    pipeline.store = store
    order = []
    pipeline.extract_features = _stage(store, order, "features", {"feature_images": 3})
    pipeline.build_frame_pairs = _stage(store, order, "pairs", {"pairs": 2})
    pipeline.match_frame_pairs = _stage(store, order, "matching", {"matches": 0}, success=0)
    pipeline.filter_matches = _stage(store, order, "filter", {"inliers": 10})
    pipeline.build_tracks_and_triangulate = _stage(store, order, "tracks", {"points": 4})
    pipeline.run_dense_ba = _stage(store, order, "ba", {"points": 4})
    pipeline.merge_duplicates = _stage(store, order, "merge", {"points": 4})
    monkeypatch.setattr(
        dense_pipeline,
        "check_dense_ba_availability",
        lambda: SimpleNamespace(available=True, details="available"),
    )

    summary = pipeline.run_all(DenseReconstructionConfig())

    assert summary.success is False
    assert order == ["features", "pairs", "matching"]
    assert "No refined feature matches" in summary.details
