import numpy as np

from map_builder.metric_depth.alignment import robust_affine_inverse_depth_alignment


def _synthetic():
    yy, xx = np.mgrid[:16, :16]
    relative = 0.2 + xx / 20.0 + yy / 50.0
    inverse = 0.4 * relative + 0.2
    z = 1.0 / inverse
    mask = np.zeros((16, 16), dtype=bool)
    mask[::3, ::3] = True
    confidence = mask.astype(np.float32)
    prompt = np.where(mask, z, 0.0)
    return relative, prompt, mask, confidence, z


def test_recovers_affine_inverse_depth_and_rejects_moderate_outliers():
    relative, prompt, mask, confidence, expected = _synthetic()
    prompt = prompt.copy()
    ys, xs = np.nonzero(mask)
    prompt[ys[:5], xs[:5]] *= 2.5
    result = robust_affine_inverse_depth_alignment(relative, prompt, mask, confidence, 12, 3, 0.15)
    assert result.success
    assert np.isclose(result.coefficient_a, 0.4, atol=0.03)
    assert np.isclose(result.coefficient_b, 0.2, atol=0.03)
    assert np.median(np.abs(result.z_depth_m[result.valid_mask] - expected[result.valid_mask])) < 0.05
    assert np.count_nonzero(result.inlier_mask) < np.count_nonzero(mask)


def test_alignment_rejects_insufficient_count_and_poor_coverage():
    relative, prompt, mask, confidence, _expected = _synthetic()
    insufficient = robust_affine_inverse_depth_alignment(relative, prompt, mask, confidence, 100, 3, 0.2)
    assert not insufficient.success and "insufficient" in insufficient.error_message
    clustered = np.zeros_like(mask); clustered[:3, :4] = True
    clustered_prompt = np.where(clustered, 1.0 / (0.4 * relative + 0.2), 0.0)
    poor = robust_affine_inverse_depth_alignment(relative, clustered_prompt, clustered, clustered.astype(float), 10, 3, 0.2)
    assert not poor.success and "coverage" in poor.error_message


def test_alignment_masks_invalid_fitted_depths_and_rejects_excessive_residual():
    relative, prompt, mask, confidence, _expected = _synthetic()
    noisy = prompt.copy()
    ys, xs = np.nonzero(mask)
    noisy[ys, xs] *= np.linspace(0.4, 2.0, len(xs))
    result = robust_affine_inverse_depth_alignment(relative, noisy, mask, confidence, 12, 3, 0.01)
    assert not result.success
    assert "error" in (result.error_message or "")
    assert np.all(result.z_depth_m[~result.valid_mask] == 0)

