import numpy as np

from map_builder.dense_reconstruction import dense_pipeline
from map_builder.dense_reconstruction.availability import AvailabilityResult
from map_builder.dense_reconstruction.dense_pipeline import DensePipeline
from map_builder.dense_reconstruction.models import DenseFeatureRecord, FramePairRecord, MatchingConfig, PairMatchRecord


class _ManyMatchMatcher:
    def __init__(self, _config):
        pass

    def match(self, _features_a, _features_b, pair_id):
        count = 12 if pair_id == 1 else 30
        return [
            PairMatchRecord(
                pair_id=pair_id,
                feature_idx_a=i,
                feature_idx_b=i,
                x_a=float(i),
                y_a=float(i),
                x_b=float(i + 1),
                y_b=float(i + 1),
            )
            for i in range(count)
        ]


def _feature(image_id):
    return DenseFeatureRecord(
        image_id=image_id,
        rel_path=f"{image_id}.jpg",
        keypoints=np.zeros((40, 2), dtype=np.float32),
        descriptors=np.ones((40, 64), dtype=np.float32),
        status="success",
        num_keypoints=40,
        extraction_mode="semi_dense_xfeat",
        descriptor_source="detectAndComputeDense",
    )


def test_dense_pipeline_matching_summary_distinguishes_pairs_from_raw_matches(tmp_path, monkeypatch):
    monkeypatch.setattr(
        dense_pipeline,
        "check_xfeat_matching_availability",
        lambda: AvailabilityResult(True, [], "available"),
    )
    monkeypatch.setattr(dense_pipeline, "XFeatSemiDenseMatcher", _ManyMatchMatcher)

    pipeline = DensePipeline(tmp_path)
    for image_id in [1, 2, 3]:
        pipeline.store.upsert_feature_record(_feature(image_id))
    pipeline.store.upsert_frame_pair(FramePairRecord(image_id_a=1, image_id_b=2))
    pipeline.store.upsert_frame_pair(FramePairRecord(image_id_a=2, image_id_b=3))

    summary = pipeline.match_frame_pairs(MatchingConfig())
    counts = pipeline.store.dense_counts()

    assert summary.success == 2
    assert "Matched 2 pair(s), 42 raw matches" in summary.details
    assert "per-pair matches min/median/max = 12/21/30" in summary.details
    assert counts["matched_pairs"] == 2
    assert counts["matches"] == 42
    pipeline.close()
