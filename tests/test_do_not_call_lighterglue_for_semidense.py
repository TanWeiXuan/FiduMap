import numpy as np

from map_builder.dense_reconstruction.models import DenseFeatureRecord, MatchingConfig
from map_builder.dense_reconstruction.xfeat_matching import XFeatSemiDenseMatcher


class _NoGrad:
    def __enter__(self):
        return None

    def __exit__(self, *_exc):
        return False


class _FakeTorch:
    def as_tensor(self, value, device=None):
        return np.asarray(value)

    def no_grad(self):
        return _NoGrad()

    def sum(self, value, dim=None):
        return np.sum(value, axis=dim)


class _FakeModel:
    dev = "cpu"

    def __init__(self):
        self.batch_match_called = False
        self.lighterglue_called = False

    def batch_match(self, descriptors_a, descriptors_b, min_cossim=-1):
        self.batch_match_called = True
        assert descriptors_a.shape == (1, 2, 2)
        assert descriptors_b.shape == (1, 2, 2)
        return [(np.array([0, 1]), np.array([1, 0]))]

    def match_lighterglue(self, *_args, **_kwargs):
        self.lighterglue_called = True
        raise AssertionError("semi-dense matching must not call match_lighterglue")


def _feature(image_id):
    return DenseFeatureRecord(
        image_id=image_id,
        keypoints=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        descriptors=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        scores=np.array([1.0, 1.0], dtype=np.float32),
        status="success",
        num_keypoints=2,
        extraction_mode="semi_dense_xfeat",
        descriptor_source="detectAndComputeDense",
    )


def test_do_not_call_lighterglue_for_semidense():
    matcher = XFeatSemiDenseMatcher.__new__(XFeatSemiDenseMatcher)
    matcher.config = MatchingConfig(min_match_score=0.1)
    matcher.torch = _FakeTorch()
    matcher.model = _FakeModel()

    matches = matcher.match(_feature(1), _feature(2), pair_id=10)

    assert matcher.model.batch_match_called
    assert not matcher.model.lighterglue_called
    assert [(m.feature_idx_a, m.feature_idx_b) for m in matches] == [(0, 1), (1, 0)]
