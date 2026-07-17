import numpy as np
import pytest
import torch

from map_builder.dense_reconstruction.models import DenseFeatureRecord, MatchingConfig
from map_builder.dense_reconstruction.xfeat_matching import XFeatSemiDenseMatcher


class _FakeXFeat:
    def __init__(self, retained_mask, refined_coordinates=None, refinement_error=None):
        self.dev = torch.device("cpu")
        self.retained_mask = torch.as_tensor(retained_mask, dtype=torch.bool)
        self.refined_coordinates = refined_coordinates
        self.refinement_error = refinement_error

    def batch_match(self, _descriptors_a, _descriptors_b, min_cossim):
        assert min_cossim == pytest.approx(0.1)
        return [
            (
                torch.tensor([0, 2, 3], dtype=torch.long),
                torch.tensor([1, 0, 2], dtype=torch.long),
            )
        ]

    def refine_matches(self, _d0, _d1, matches, batch_idx, return_mask):
        assert batch_idx == 0
        assert return_mask is True
        assert len(matches[0][0]) == 3
        if self.refinement_error is not None:
            raise self.refinement_error
        coordinates = self.refined_coordinates
        if coordinates is None:
            coordinates = torch.empty((int(self.retained_mask.sum()), 4), dtype=torch.float32)
        return torch.as_tensor(coordinates, dtype=torch.float32), self.retained_mask


def _feature(image_id, keypoints):
    descriptors = np.eye(4, dtype=np.float32)
    return DenseFeatureRecord(
        image_id=image_id,
        keypoints=np.asarray(keypoints, dtype=np.float32),
        descriptors=descriptors,
        scores=np.ones(4, dtype=np.float32),
        status="success",
        num_keypoints=4,
        extraction_mode="semi_dense_xfeat",
        descriptor_source="detectAndComputeDense",
    )


def _matcher(model):
    matcher = XFeatSemiDenseMatcher.__new__(XFeatSemiDenseMatcher)
    matcher.config = MatchingConfig(device="cpu")
    matcher.torch = torch
    matcher.model = model
    return matcher


def test_fine_refinement_emits_only_retained_coarse_matches():
    refined = [[10.25, 20.5, 110.0, 120.0], [30.75, 40.5, 130.0, 140.0]]
    matcher = _matcher(_FakeXFeat([True, False, True], refined))
    features_a = _feature(1, [[10, 20], [11, 21], [20, 30], [30, 40]])
    features_b = _feature(2, [[100, 110], [110, 120], [130, 140], [150, 160]])

    matches = matcher.match(features_a, features_b, pair_id=7)

    assert [(match.feature_idx_a, match.feature_idx_b) for match in matches] == [(0, 1), (3, 2)]
    assert np.allclose(
        [(match.x_a, match.y_a, match.x_b, match.y_b) for match in matches],
        refined,
    )
    assert all(match.feature_idx_a != 2 for match in matches)


def test_fine_refinement_with_zero_survivors_returns_no_matches():
    matcher = _matcher(_FakeXFeat([False, False, False]))
    features_a = _feature(1, [[10, 20], [11, 21], [20, 30], [30, 40]])
    features_b = _feature(2, [[100, 110], [110, 120], [130, 140], [150, 160]])

    assert matcher.match(features_a, features_b, pair_id=8) == []


def test_fine_refinement_failure_is_not_replaced_with_coarse_matches():
    matcher = _matcher(_FakeXFeat([True, True, True], refinement_error=ValueError("fine matcher failed")))
    features_a = _feature(1, [[10, 20], [11, 21], [20, 30], [30, 40]])
    features_b = _feature(2, [[100, 110], [110, 120], [130, 140], [150, 160]])

    with pytest.raises(RuntimeError, match="fine refinement failed for frame-pair 9"):
        matcher.match(features_a, features_b, pair_id=9)
