import numpy as np
import pytest

from map_builder.dense_reconstruction.epipolar_filter import (
    compute_ray_angular_errors,
    epipolar_inlier_mask,
    filter_pair_matches,
    relative_pose_21,
)
from map_builder.dense_reconstruction.models import EpipolarFilterConfig
from map_builder.dense_reconstruction.models import PairMatchRecord


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def test_epipolar_zero_for_consistent_rays_and_rejects_outlier():
    T1 = {"R": np.eye(3).tolist(), "t": [0, 0, 0]}
    T2 = {"R": np.eye(3).tolist(), "t": [1, 0, 0]}
    R21, t21 = relative_pose_21(T1, T2)
    points = [np.array([0.0, 0.0, 5.0]), np.array([0.2, 0.3, 4.0])]
    f1 = np.array([_unit(p) for p in points])
    f2 = np.array([_unit(p - np.array([1.0, 0.0, 0.0])) for p in points])
    errors = compute_ray_angular_errors(f1, f2, R21, t21)
    assert np.all(errors < 1e-10)

    f2_bad = f2.copy()
    f2_bad[1] = _unit([0.0, 1.0, 1.0])
    errors, inliers = epipolar_inlier_mask(f1, f2_bad, R21, t21, EpipolarFilterConfig(max_ray_angular_error=0.003))
    assert inliers.tolist() == [True, False]
    assert errors[1] > 0.003


def test_filter_pair_matches_requires_persisted_match_ids():
    class Camera:
        def unproject_many(self, pixels):
            rays = np.column_stack([pixels[:, 0] * 0.0, pixels[:, 1] * 0.0, np.ones(len(pixels))])
            return rays

    match = PairMatchRecord(pair_id=1, feature_idx_a=0, feature_idx_b=0, x_a=0, y_a=0, x_b=0, y_b=0)
    with pytest.raises(ValueError, match="persisted pair_matches"):
        filter_pair_matches(
            [match],
            {"R": np.eye(3).tolist(), "t": [0, 0, 0]},
            {"R": np.eye(3).tolist(), "t": [1, 0, 0]},
            Camera(),
            EpipolarFilterConfig(min_baseline_m=0.0),
        )
