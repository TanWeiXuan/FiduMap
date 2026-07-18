from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class AlignmentResult:
    success: bool
    z_depth_m: np.ndarray
    valid_mask: np.ndarray
    coefficient_a: float | None
    coefficient_b: float | None
    inlier_mask: np.ndarray
    median_absolute_error_m: float | None
    median_relative_error: float | None
    error_message: str | None = None


def validate_anchor_coverage(mask: np.ndarray, minimum_count: int, minimum_grid_cells: int, grid_size: int = 4) -> tuple[bool, int, str | None]:
    anchors = np.asarray(mask, dtype=bool)
    count = int(np.count_nonzero(anchors))
    cells = occupied_grid_cells(anchors, grid_size)
    if count < minimum_count:
        return False, cells, f"insufficient anchors: {count} < {minimum_count}"
    if cells < minimum_grid_cells:
        return False, cells, f"poor anchor coverage: {cells} grid cells < {minimum_grid_cells}"
    return True, cells, None


def occupied_grid_cells(mask: np.ndarray, grid_size: int = 4) -> int:
    ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
    if not len(xs):
        return 0
    h, w = mask.shape
    gx = np.minimum(xs * grid_size // max(w, 1), grid_size - 1)
    gy = np.minimum(ys * grid_size // max(h, 1), grid_size - 1)
    return len(set(zip(gx.tolist(), gy.tolist())))


def robust_affine_inverse_depth_alignment(
    relative_prediction: np.ndarray,
    anchor_depth_z_m: np.ndarray,
    anchor_mask: np.ndarray,
    anchor_confidence: np.ndarray,
    minimum_anchor_count: int,
    minimum_grid_cells: int,
    maximum_median_relative_error: float,
    max_iterations: int = 6,
) -> AlignmentResult:
    rel = np.asarray(relative_prediction, dtype=float)
    anchor_depth = np.asarray(anchor_depth_z_m, dtype=float)
    mask = np.asarray(anchor_mask, dtype=bool)
    weights = np.asarray(anchor_confidence, dtype=float)
    empty = np.zeros_like(rel, dtype=np.float32)
    coverage_ok, _cells, reason = validate_anchor_coverage(mask, minimum_anchor_count, minimum_grid_cells)
    if not coverage_ok:
        return AlignmentResult(False, empty, mask & False, None, None, mask & False, None, None, reason)
    sample_ok = mask & np.isfinite(rel) & np.isfinite(anchor_depth) & (anchor_depth > 0.0) & np.isfinite(weights) & (weights > 0.0)
    ys, xs = np.nonzero(sample_ok)
    if len(xs) < minimum_anchor_count:
        return AlignmentResult(False, empty, mask & False, None, None, mask & False, None, None, "too few finite model samples at anchors")
    x = rel[ys, xs]
    y = 1.0 / anchor_depth[ys, xs]
    w = np.clip(weights[ys, xs], 1e-3, 1.0)
    if np.ptp(anchor_depth[ys, xs]) <= max(1e-4, 0.01 * float(np.median(anchor_depth[ys, xs]))):
        return AlignmentResult(False, empty, mask & False, None, None, mask & False, None, None, "anchors do not span two meaningful depth ranges")
    inliers = np.ones(len(x), dtype=bool)
    coeff = np.array([np.nan, np.nan])
    for _ in range(max_iterations):
        if np.count_nonzero(inliers) < max(2, minimum_anchor_count // 2):
            break
        A = np.column_stack((x[inliers], np.ones(np.count_nonzero(inliers))))
        sw = np.sqrt(w[inliers])
        try:
            coeff = np.linalg.lstsq(A * sw[:, None], y[inliers] * sw, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        residual = y - (coeff[0] * x + coeff[1])
        center = float(np.median(residual[inliers]))
        mad = float(np.median(np.abs(residual[inliers] - center)))
        scale = max(1.4826 * mad, 1e-8)
        updated = np.abs(residual - center) <= 3.5 * scale
        if np.array_equal(updated, inliers):
            break
        inliers = updated
    if np.count_nonzero(inliers) < max(2, minimum_anchor_count // 2):
        return AlignmentResult(False, empty, mask & False, None, None, sample_ok & False, None, None, "too few robust alignment inliers")
    a, b = float(coeff[0]), float(coeff[1])
    if not np.isfinite(a) or not np.isfinite(b):
        return AlignmentResult(False, empty, mask & False, None, None, sample_ok & False, None, None, "non-finite alignment coefficients")
    inverse = a * rel + b
    finite_rel = np.isfinite(rel)
    non_positive_fraction = float(np.count_nonzero(finite_rel & (inverse <= 0.0))) / max(np.count_nonzero(finite_rel), 1)
    valid = finite_rel & np.isfinite(inverse) & (inverse > 0.0)
    z = np.zeros_like(rel, dtype=np.float32)
    z[valid] = (1.0 / inverse[valid]).astype(np.float32)
    pred_anchor = z[ys, xs]
    absolute = np.abs(pred_anchor - anchor_depth[ys, xs])
    relative = absolute / anchor_depth[ys, xs]
    median_abs = float(np.median(absolute[inliers]))
    median_rel = float(np.median(relative[inliers]))
    full_inliers = np.zeros_like(mask, dtype=bool)
    full_inliers[ys[inliers], xs[inliers]] = True
    if non_positive_fraction > 0.25:
        return AlignmentResult(False, z, valid, a, b, full_inliers, median_abs, median_rel, "too many non-positive fitted inverse depths")
    if median_rel > maximum_median_relative_error:
        return AlignmentResult(False, z, valid, a, b, full_inliers, median_abs, median_rel, f"alignment median relative error {median_rel:.3f} exceeds {maximum_median_relative_error:.3f}")
    return AlignmentResult(True, z, valid, a, b, full_inliers, median_abs, median_rel)
