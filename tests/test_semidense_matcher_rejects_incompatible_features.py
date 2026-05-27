import numpy as np
import pytest

from map_builder.dense_reconstruction import dense_pipeline
from map_builder.dense_reconstruction.availability import AvailabilityResult
from map_builder.dense_reconstruction.dense_pipeline import DensePipeline
from map_builder.dense_reconstruction.models import DenseFeatureRecord, FramePairRecord, MatchingConfig
from map_builder.dense_reconstruction.xfeat_matching import (
    IncompatibleDenseFeatureError,
    _validate_compatible_feature,
)


def _feature(**overrides):
    values = {
        "image_id": 1,
        "rel_path": "a.jpg",
        "keypoints": np.array([[0.0, 0.0]], dtype=np.float32),
        "descriptors": np.ones((1, 64), dtype=np.float32),
        "scores": np.array([1.0], dtype=np.float32),
        "status": "success",
        "num_keypoints": 1,
        "extraction_mode": "semi_dense_xfeat",
        "descriptor_source": "detectAndComputeDense",
    }
    values.update(overrides)
    return DenseFeatureRecord(**values)


def test_semidense_matcher_rejects_incompatible_features():
    with pytest.raises(IncompatibleDenseFeatureError, match="older/incompatible dense feature format"):
        _validate_compatible_feature(_feature(extraction_mode=None), "a")

    with pytest.raises(IncompatibleDenseFeatureError, match="older/incompatible dense feature format"):
        _validate_compatible_feature(_feature(descriptor_source="match_lighterglue"), "a")


def test_dense_pipeline_reports_incompatible_pair_as_skipped(tmp_path, monkeypatch):
    class ValidatingMatcher:
        def __init__(self, _config):
            pass

        def match(self, features_a, features_b, pair_id):
            _validate_compatible_feature(features_a, "a")
            _validate_compatible_feature(features_b, "b")
            return []

    monkeypatch.setattr(
        dense_pipeline,
        "check_xfeat_matching_availability",
        lambda: AvailabilityResult(True, [], "available"),
    )
    monkeypatch.setattr(dense_pipeline, "XFeatSemiDenseMatcher", ValidatingMatcher)

    pipeline = DensePipeline(tmp_path)
    pipeline.store.upsert_feature_record(_feature(image_id=1, rel_path="a.jpg", extraction_mode=None))
    pipeline.store.upsert_feature_record(_feature(image_id=2, rel_path="b.jpg"))
    pipeline.store.upsert_frame_pair(FramePairRecord(image_id_a=1, image_id_b=2))

    summary = pipeline.match_frame_pairs(MatchingConfig())

    assert summary.success == 0
    assert "skipped 1 incompatible/missing pair(s)" in summary.details
    assert "0 raw matches" in summary.details
    pipeline.close()
