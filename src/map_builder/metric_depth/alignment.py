from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
import hashlib
from typing import Any
import numpy as np

from .models import MetricAnchor


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


@dataclass(frozen=True)
class AnchorSplit:
    training_mask: np.ndarray
    holdout_mask: np.ndarray
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class SpatialGridFit:
    success: bool
    coefficients: np.ndarray
    rows: int
    columns: int
    inlier_mask: np.ndarray
    condition_number: float | None
    error_message: str | None = None


@dataclass
class SplineSpatialAlignmentResult:
    success: bool
    final_z_depth_m: np.ndarray
    valid_mask: np.ndarray
    confidence: np.ndarray
    global_spline_z_depth_m: np.ndarray
    spatial_log_correction: np.ndarray
    extrapolation_mask: np.ndarray
    training_inlier_mask: np.ndarray
    training_metrics: dict[str, float | int | None] = field(default_factory=dict)
    holdout_metrics: dict[str, float | int | None] = field(default_factory=dict)
    spline_knots_x: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    spline_knots_y: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    spatial_grid_coefficients: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=float))
    affine_holdout_median_relative_error: float | None = None
    anchor_mask: np.ndarray | None = None
    anchor_residual_m: np.ndarray | None = None
    anchor_split: np.ndarray | None = None
    prediction_normalization: dict[str, float] = field(default_factory=dict)
    correction_statistics: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error_message: str | None = None

    @property
    def z_depth_m(self) -> np.ndarray:
        return self.final_z_depth_m


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


def deterministic_anchor_split(
    anchors: Iterable[MetricAnchor],
    image_id: int,
    holdout_fraction: float = 0.2,
    minimum_training_count: int = 12,
) -> AnchorSplit:
    """Split with a stable digest, preferring complete source groups."""
    values = list(anchors)
    count = len(values)
    train = np.ones(count, dtype=bool)
    holdout = np.zeros(count, dtype=bool)
    warnings: list[str] = []
    maximum_holdout = max(0, count - int(minimum_training_count))
    if maximum_holdout == 0:
        return AnchorSplit(train, holdout, ("held-out selection unavailable: too few anchors beyond the training minimum",))

    for source in sorted({a.source or a.provenance for a in values}):
        indices = [i for i, anchor in enumerate(values) if (anchor.source or anchor.provenance) == source]
        if len(indices) < 5:
            warnings.append(f"{source} has too few anchors for a stratified holdout")
            continue
        target = max(1, int(round(len(indices) * float(holdout_fraction))))
        groups: dict[str, list[int]] = {}
        for index in indices:
            anchor = values[index]
            groups.setdefault(anchor.group_id or f"sample:{index}", []).append(index)
        ordered_groups = sorted(groups.values(), key=lambda group: _stable_anchor_digest(image_id, values[group[0]], group_only=True))
        chosen: list[int] = []
        if len(ordered_groups) >= 2:
            for group in ordered_groups:
                if chosen and abs(len(chosen) - target) <= abs(len(chosen) + len(group) - target):
                    break
                if len(chosen) + len(group) >= len(indices):
                    break
                chosen.extend(group)
                if len(chosen) >= target:
                    break
        if not chosen or count - (int(np.count_nonzero(holdout)) + len(chosen)) < minimum_training_count:
            ordered = sorted(indices, key=lambda i: _stable_anchor_digest(image_id, values[i], group_only=False))
            chosen = ordered[:target]
        for index in chosen:
            holdout[index] = True
            train[index] = False

    selected = np.flatnonzero(holdout)
    if len(selected) > maximum_holdout:
        keep = sorted(selected.tolist(), key=lambda i: _stable_anchor_digest(image_id, values[i], group_only=False))[:maximum_holdout]
        holdout[:] = False
        holdout[keep] = True
        train = ~holdout
    if not np.any(holdout):
        warnings.append("held-out selection unavailable")
    return AnchorSplit(train, holdout, tuple(warnings))


def _stable_anchor_digest(image_id: int, anchor: MetricAnchor, group_only: bool) -> bytes:
    parts = [str(int(image_id)), anchor.source or anchor.provenance, anchor.group_id]
    if not group_only:
        parts.extend((f"{float(anchor.u):.6f}", f"{float(anchor.v):.6f}"))
    return hashlib.sha256("|".join(parts).encode("utf-8")).digest()


def robust_prediction_normalization(prediction: np.ndarray) -> tuple[np.ndarray, float, float]:
    values = np.asarray(prediction, dtype=float)
    finite = np.isfinite(values)
    if np.count_nonzero(finite) < 2:
        raise ValueError("prediction map has too few finite values")
    low, high = (float(value) for value in np.percentile(values[finite], [1.0, 99.0]))
    if not np.isfinite(low) or not np.isfinite(high) or high - low <= max(1e-9, abs(low) * 1e-9):
        raise ValueError("prediction map has no meaningful finite range")
    normalized = np.full(values.shape, np.nan, dtype=float)
    normalized[finite] = np.clip((values[finite] - low) / (high - low), 0.0, 1.0)
    return normalized, low, high


def weighted_decreasing_isotonic_regression(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Weighted decreasing PAV fit returned in the original sample order."""
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    w = np.ones_like(y) if weights is None else np.asarray(weights, dtype=float).reshape(-1)
    if not (len(x) == len(y) == len(w)) or not len(x):
        raise ValueError("isotonic samples must be non-empty and equal length")
    if not np.all(np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0.0)):
        raise ValueError("isotonic samples and weights must be finite, with positive weights")
    order = np.argsort(x, kind="mergesort")
    sorted_x, sorted_y, sorted_w = x[order], y[order], w[order]
    unique_x, inverse = np.unique(sorted_x, return_inverse=True)
    unique_w = np.bincount(inverse, weights=sorted_w, minlength=len(unique_x))
    unique_y = np.bincount(inverse, weights=sorted_y * sorted_w, minlength=len(unique_x)) / unique_w
    levels: list[float] = []
    block_weights: list[float] = []
    starts: list[int] = []
    ends: list[int] = []
    for index, (value, weight) in enumerate(zip(unique_y, unique_w)):
        levels.append(float(value)); block_weights.append(float(weight)); starts.append(index); ends.append(index + 1)
        while len(levels) >= 2 and levels[-2] < levels[-1]:
            combined_weight = block_weights[-2] + block_weights[-1]
            combined_level = (levels[-2] * block_weights[-2] + levels[-1] * block_weights[-1]) / combined_weight
            levels[-2:] = [combined_level]
            block_weights[-2:] = [combined_weight]
            ends[-2:] = [ends[-1]]
            starts.pop()
    fitted_unique = np.empty(len(unique_x), dtype=float)
    for level, start, end in zip(levels, starts, ends):
        fitted_unique[start:end] = level
    fitted_sorted = fitted_unique[inverse]
    fitted = np.empty_like(fitted_sorted)
    fitted[order] = fitted_sorted
    return fitted


def _huber_weights(residual: np.ndarray, base_mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    residual = np.asarray(residual, dtype=float)
    active = np.ones(residual.shape, dtype=bool) if base_mask is None else np.asarray(base_mask, dtype=bool)
    center = float(np.median(residual[active])) if np.any(active) else 0.0
    mad = float(np.median(np.abs(residual[active] - center))) if np.any(active) else 0.0
    scale = max(1.4826 * mad, 1e-8)
    cutoff = 1.5 * scale
    magnitude = np.abs(residual - center)
    robust = np.ones_like(residual)
    outside = magnitude > cutoff
    robust[outside] = cutoff / np.maximum(magnitude[outside], 1e-12)
    return robust, magnitude <= 3.5 * scale


def fit_robust_decreasing_isotonic(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    max_iterations: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = np.asarray(weights, dtype=float)
    robust = np.ones_like(base)
    inliers = np.ones_like(base, dtype=bool)
    fitted = np.zeros_like(np.asarray(y, dtype=float))
    for _ in range(max_iterations):
        previous = robust.copy()
        fitted = weighted_decreasing_isotonic_regression(x, y, np.maximum(base * robust, 1e-12))
        robust, inliers = _huber_weights(np.asarray(y, dtype=float) - fitted)
        if np.allclose(previous, robust, atol=1e-3, rtol=1e-3):
            break
    fitted = weighted_decreasing_isotonic_regression(x, y, np.maximum(base * robust, 1e-12))
    return fitted, robust, inliers


def _weighted_quantiles(values: np.ndarray, weights: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    v, w = np.asarray(values)[order], np.asarray(weights)[order]
    cumulative = np.cumsum(w) - 0.5 * w
    cumulative /= max(float(np.sum(w)), 1e-12)
    return np.interp(quantiles, cumulative, v, left=v[0], right=v[-1])


def compress_decreasing_isotonic_to_knots(
    x: np.ndarray,
    fitted_y: np.ndarray,
    weights: np.ndarray,
    knot_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(fitted_y, dtype=float)
    order = np.argsort(x, kind="mergesort")
    sorted_x, sorted_y = x[order], y[order]
    unique_x, first = np.unique(sorted_x, return_index=True)
    if len(unique_x) < 2:
        raise ValueError("monotonic spline requires at least two distinct prediction values")
    unique_y = sorted_y[first]
    count = min(max(int(knot_count), 2), len(unique_x))
    knots_x = _weighted_quantiles(x, np.asarray(weights, dtype=float), np.linspace(0.0, 1.0, count))
    knots_x[0], knots_x[-1] = unique_x[0], unique_x[-1]
    knots_x = np.unique(knots_x)
    if len(knots_x) < 2:
        knots_x = np.array([unique_x[0], unique_x[-1]])
    knots_y = np.interp(knots_x, unique_x, unique_y)
    knots_y = np.minimum.accumulate(knots_y)
    return knots_x, knots_y


def evaluate_monotonic_spline(values: np.ndarray, knots_x: np.ndarray, knots_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=float)
    finite = np.isfinite(x)
    extrapolated = finite & ((x < knots_x[0]) | (x > knots_x[-1]))
    output = np.full(x.shape, np.nan, dtype=float)
    output[finite] = np.interp(x[finite], knots_x, knots_y, left=knots_y[0], right=knots_y[-1])
    return output, extrapolated


def bilinear_grid_basis(points_uv: np.ndarray, width: int, height: int, columns: int, rows: int) -> np.ndarray:
    points = np.asarray(points_uv, dtype=float).reshape(-1, 2)
    x = np.clip(points[:, 0] / max(width - 1, 1), 0.0, 1.0) * (columns - 1)
    y = np.clip(points[:, 1] / max(height - 1, 1), 0.0, 1.0) * (rows - 1)
    x0 = np.minimum(np.floor(x).astype(int), columns - 2); y0 = np.minimum(np.floor(y).astype(int), rows - 2)
    dx, dy = x - x0, y - y0
    basis = np.zeros((len(points), rows * columns), dtype=float)
    sample = np.arange(len(points))
    for node_x, node_y, value in (
        (x0, y0, (1.0 - dx) * (1.0 - dy)),
        (x0 + 1, y0, dx * (1.0 - dy)),
        (x0, y0 + 1, (1.0 - dx) * dy),
        (x0 + 1, y0 + 1, dx * dy),
    ):
        basis[sample, node_y * columns + node_x] += value
    return basis


def _grid_laplacian(columns: int, rows: int) -> np.ndarray:
    size = columns * rows
    laplacian = np.zeros((size, size), dtype=float)
    for y in range(rows):
        for x in range(columns):
            index = y * columns + x
            neighbors = []
            if x: neighbors.append(index - 1)
            if x + 1 < columns: neighbors.append(index + 1)
            if y: neighbors.append(index - columns)
            if y + 1 < rows: neighbors.append(index + columns)
            laplacian[index, index] = len(neighbors)
            laplacian[index, neighbors] = -1.0
    return laplacian


def fit_spatial_correction_grid(
    points_uv: np.ndarray,
    residual_log_depth: np.ndarray,
    weights: np.ndarray,
    width: int,
    height: int,
    columns: int = 8,
    rows: int = 6,
    lambda_smooth: float = 0.02,
    lambda_prior: float = 0.002,
    max_iterations: int = 6,
) -> SpatialGridFit:
    basis = bilinear_grid_basis(points_uv, width, height, columns, rows)
    residual = np.asarray(residual_log_depth, dtype=float).reshape(-1)
    base = np.asarray(weights, dtype=float).reshape(-1)
    size = rows * columns
    empty = np.zeros((rows, columns), dtype=float)
    if len(residual) < 2 or not np.all(np.isfinite(basis) & np.isfinite(residual[:, None])):
        return SpatialGridFit(False, empty, rows, columns, np.zeros(len(residual), bool), None, "spatial correction has too few finite samples")
    laplacian = _grid_laplacian(columns, rows)
    robust = np.ones(len(residual), dtype=float)
    coefficients = np.zeros(size, dtype=float)
    inliers = np.ones(len(residual), dtype=bool)
    condition = None
    for _ in range(max_iterations):
        combined = np.maximum(base * robust, 1e-12)
        normal = basis.T @ (combined[:, None] * basis) + float(lambda_smooth) * (laplacian.T @ laplacian) + float(lambda_prior) * np.eye(size)
        rhs = basis.T @ (combined * residual)
        try:
            condition = float(np.linalg.cond(normal))
            if not np.isfinite(condition) or condition > 1e12:
                return SpatialGridFit(False, empty, rows, columns, inliers, condition, "spatial correction solve is singular or ill-conditioned")
            coefficients = np.linalg.solve(normal, rhs)
        except np.linalg.LinAlgError:
            return SpatialGridFit(False, empty, rows, columns, inliers, condition, "spatial correction solve is singular")
        if not np.all(np.isfinite(coefficients)):
            return SpatialGridFit(False, empty, rows, columns, inliers, condition, "spatial correction solve produced non-finite coefficients")
        updated, inliers = _huber_weights(residual - basis @ coefficients)
        if np.allclose(updated, robust, atol=1e-3, rtol=1e-3):
            break
        robust = updated
    return SpatialGridFit(True, coefficients.reshape(rows, columns), rows, columns, inliers, condition)


def evaluate_spatial_grid(coefficients: np.ndarray, width: int, height: int) -> np.ndarray:
    grid = np.asarray(coefficients, dtype=float)
    rows, columns = grid.shape
    x = np.linspace(0.0, columns - 1, width)
    y = np.linspace(0.0, rows - 1, height)
    x0 = np.minimum(np.floor(x).astype(int), columns - 2); y0 = np.minimum(np.floor(y).astype(int), rows - 2)
    dx, dy = x - x0, y - y0
    top = grid[y0[:, None], x0[None, :]] * (1.0 - dx)[None, :] + grid[y0[:, None], (x0 + 1)[None, :]] * dx[None, :]
    bottom = grid[(y0 + 1)[:, None], x0[None, :]] * (1.0 - dx)[None, :] + grid[(y0 + 1)[:, None], (x0 + 1)[None, :]] * dx[None, :]
    return top * (1.0 - dy)[:, None] + bottom * dy[:, None]


def adapted_spatial_grid_rows(width: int, height: int, columns: int, requested_rows: int) -> int:
    aspect = float(width) / max(float(height), 1.0)
    if 0.4 <= aspect <= 2.5:
        return int(requested_rows)
    return int(np.clip(round(columns / max(aspect, 1e-6)), 2, 32))


def _error_metrics(predicted: np.ndarray, expected: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float | int | None]:
    use = np.ones(len(expected), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    use &= np.isfinite(predicted) & np.isfinite(expected) & (expected > 0.0)
    if not np.any(use):
        return {"count": 0, "median_absolute_error_m": None, "median_relative_error": None}
    absolute = np.abs(predicted[use] - expected[use])
    return {
        "count": int(np.count_nonzero(use)),
        "median_absolute_error_m": float(np.median(absolute)),
        "median_relative_error": float(np.median(absolute / expected[use])),
    }


def _fit_affine_samples(prediction: np.ndarray, depth: np.ndarray, weights: np.ndarray) -> tuple[float, float] | None:
    x, target, base = np.asarray(prediction, float), 1.0 / np.asarray(depth, float), np.asarray(weights, float)
    robust = np.ones_like(base)
    coefficients = np.array([np.nan, np.nan])
    for _ in range(6):
        design = np.column_stack((x, np.ones(len(x))))
        sw = np.sqrt(np.maximum(base * robust, 1e-12))
        try:
            coefficients = np.linalg.lstsq(design * sw[:, None], target * sw, rcond=None)[0]
        except np.linalg.LinAlgError:
            return None
        updated, _ = _huber_weights(target - design @ coefficients)
        if np.allclose(updated, robust, atol=1e-3, rtol=1e-3):
            break
        robust = updated
    return None if not np.all(np.isfinite(coefficients)) else (float(coefficients[0]), float(coefficients[1]))


def monotonic_spline_spatial_alignment(
    relative_prediction: np.ndarray,
    anchors: Iterable[MetricAnchor],
    image_id: int,
    config: Any,
    stage_callback: Callable[[str], None] | None = None,
) -> SplineSpatialAlignmentResult:
    """Fit one image independently; held-out anchors are never used by either fitted component."""
    prediction = np.asarray(relative_prediction, dtype=float)
    height, width = prediction.shape
    empty = np.zeros((height, width), dtype=np.float32)
    invalid = np.zeros((height, width), dtype=bool)

    def failure(message: str, warnings: list[str] | None = None) -> SplineSpatialAlignmentResult:
        return SplineSpatialAlignmentResult(False, empty.copy(), invalid.copy(), empty.copy(), empty.copy(), empty.copy(), invalid.copy(), invalid.copy(), warnings=warnings or [], error_message=message)

    values = [a for a in anchors if np.isfinite(a.u) and np.isfinite(a.v) and np.isfinite(a.z_depth_m) and a.z_depth_m > 0.0 and 0 <= round(a.u) < width and 0 <= round(a.v) < height and np.isfinite(a.fit_weight) and a.fit_weight > 0.0]
    minimum = int(getattr(config, "minimum_anchor_count", 12))
    if len(values) < minimum:
        return failure(f"insufficient anchors: {len(values)} < {minimum}")
    try:
        normalized, prediction_low, prediction_high = robust_prediction_normalization(prediction)
    except ValueError as exc:
        return failure(str(exc))
    pixels = np.array([[a.u, a.v] for a in values], dtype=float)
    pixel_indices = np.rint(pixels).astype(int)
    sample_prediction = normalized[pixel_indices[:, 1], pixel_indices[:, 0]]
    depth = np.array([a.z_depth_m for a in values], dtype=float)
    weights = np.array([a.fit_weight for a in values], dtype=float)
    finite = np.isfinite(sample_prediction) & np.isfinite(depth) & (depth > 0.0) & np.isfinite(weights) & (weights > 0.0)
    values = [anchor for anchor, keep in zip(values, finite) if keep]
    pixels, pixel_indices, sample_prediction, depth, weights = pixels[finite], pixel_indices[finite], sample_prediction[finite], depth[finite], weights[finite]
    if len(values) < minimum:
        return failure("too few finite model samples at anchors")
    split = deterministic_anchor_split(values, image_id, float(getattr(config, "holdout_fraction", 0.2)), minimum)
    train, holdout = split.training_mask, split.holdout_mask
    split_warnings = list(split.warnings)
    train_support = np.zeros((height, width), dtype=bool)
    train_support[pixel_indices[train, 1], pixel_indices[train, 0]] = True
    coverage_ok, _cells, reason = validate_anchor_coverage(train_support, minimum, int(getattr(config, "minimum_anchor_grid_cells", 3)))
    if not coverage_ok:
        return failure(reason or "insufficient training coverage", split_warnings)
    if np.ptp(depth[train]) <= max(1e-4, 0.01 * float(np.median(depth[train]))):
        return failure("anchors do not span two meaningful depth ranges", split_warnings)
    target = np.log(depth)
    try:
        fitted, robust, spline_inliers = fit_robust_decreasing_isotonic(
            sample_prediction[train], target[train], weights[train]
        )
        knots_x, knots_y = compress_decreasing_isotonic_to_knots(
            sample_prediction[train], fitted, weights[train] * robust,
            int(getattr(config, "spline_knot_count", 12)),
        )
    except ValueError:
        return failure("monotonic spline could not be fitted", split_warnings)
    global_log, extrapolation = evaluate_monotonic_spline(normalized, knots_x, knots_y)
    global_depth = np.zeros((height, width), dtype=np.float32)
    finite_global = np.isfinite(global_log)
    global_depth[finite_global] = np.exp(global_log[finite_global]).astype(np.float32)
    train_global_log, _ = evaluate_monotonic_spline(sample_prediction[train], knots_x, knots_y)
    grid_columns = int(getattr(config, "spatial_grid_columns", 8))
    grid_rows = adapted_spatial_grid_rows(width, height, grid_columns, int(getattr(config, "spatial_grid_rows", 6)))
    if stage_callback is not None:
        stage_callback("fitting_spatial_correction")
    spatial_fit = fit_spatial_correction_grid(
        pixels[train], target[train] - train_global_log, weights[train], width, height,
        grid_columns, grid_rows,
        float(getattr(config, "spatial_smoothness", 0.02)), float(getattr(config, "spatial_prior", 0.002)),
    )
    if not spatial_fit.success:
        return failure(spatial_fit.error_message or "spatial correction field could not be fitted", split_warnings)
    correction_unclamped = evaluate_spatial_grid(spatial_fit.coefficients, width, height)
    if not np.all(np.isfinite(correction_unclamped)):
        return failure("spatial correction field contains non-finite values", split_warnings)
    bound = float(getattr(config, "maximum_log_depth_correction", 0.4))
    correction = np.clip(correction_unclamped, -bound, bound)
    saturated = np.abs(correction_unclamped) >= bound
    saturation_fraction = float(np.mean(saturated))
    final_log = global_log + correction
    valid = np.isfinite(final_log)
    final_depth = np.zeros((height, width), dtype=np.float32)
    final_depth[valid] = np.exp(final_log[valid]).astype(np.float32)
    valid &= np.isfinite(final_depth) & (final_depth > 0.0)
    invalid_fraction = 1.0 - float(np.mean(valid))

    if stage_callback is not None:
        stage_callback("evaluating_holdout")
    predicted_samples = final_depth[pixel_indices[:, 1], pixel_indices[:, 0]].astype(float)
    training_metrics = _error_metrics(predicted_samples, depth, train)
    holdout_metrics = _error_metrics(predicted_samples, depth, holdout)
    marker_holdout = holdout & np.array([(a.source or a.provenance) == "marker_surface" for a in values])
    dense_holdout = holdout & np.array([(a.source or a.provenance) == "dense_track" for a in values])
    holdout_metrics["marker_median_relative_error"] = _error_metrics(predicted_samples, depth, marker_holdout)["median_relative_error"]
    holdout_metrics["dense_track_median_relative_error"] = _error_metrics(predicted_samples, depth, dense_holdout)["median_relative_error"]
    affine_error = None
    affine = _fit_affine_samples(prediction[pixel_indices[train, 1], pixel_indices[train, 0]], depth[train], weights[train])
    if affine is not None and np.any(holdout):
        inverse = affine[0] * prediction[pixel_indices[holdout, 1], pixel_indices[holdout, 0]] + affine[1]
        affine_pred = np.where(inverse > 0.0, 1.0 / inverse, np.nan)
        affine_error = _error_metrics(affine_pred, depth[holdout])["median_relative_error"]

    warnings = split_warnings
    if len({a.source or a.provenance for a in values}) == 1:
        warnings.append("only one metric-anchor source is available")
    if float(np.mean(extrapolation)) > 0.5:
        warnings.append("spline extrapolation covers much of the image")
    if saturation_fraction > 0.1:
        warnings.append("spatial correction saturation is high")
    spline_holdout_error = holdout_metrics["median_relative_error"]
    if affine_error is not None and spline_holdout_error is not None and float(spline_holdout_error) > float(affine_error):
        warnings.append("spline-spatial alignment is worse than affine on held-out anchors")
    reject_reason = None
    if invalid_fraction > 0.05:
        reject_reason = "too many invalid or non-positive output depths"
    elif saturation_fraction > float(getattr(config, "maximum_correction_saturation_fraction", 0.25)):
        reject_reason = f"correction saturation fraction {saturation_fraction:.3f} exceeds the configured limit"
    elif spline_holdout_error is not None and float(spline_holdout_error) > float(getattr(config, "maximum_alignment_median_relative_error", 0.25)):
        reject_reason = f"holdout median relative error {float(spline_holdout_error):.3f} exceeds {float(getattr(config, 'maximum_alignment_median_relative_error', 0.25)):.3f}"

    anchor_mask = np.zeros((height, width), dtype=bool)
    anchor_residual = np.full((height, width), np.nan, dtype=np.float32)
    split_map = np.zeros((height, width), dtype=np.uint8)
    inlier_map = np.zeros((height, width), dtype=bool)
    train_indices = np.flatnonzero(train)
    combined_inliers = np.asarray(spline_inliers, bool) & np.asarray(spatial_fit.inlier_mask, bool)
    training_metrics["robust_inlier_count"] = int(np.count_nonzero(combined_inliers))
    for local, index in enumerate(train_indices):
        u, v = pixel_indices[index]
        anchor_mask[v, u] = True; split_map[v, u] = 1
        anchor_residual[v, u] = np.float32(predicted_samples[index] - depth[index])
        if combined_inliers[local]: inlier_map[v, u] = True
    for index in np.flatnonzero(holdout):
        u, v = pixel_indices[index]
        anchor_mask[v, u] = True; split_map[v, u] = 2
        anchor_residual[v, u] = np.float32(predicted_samples[index] - depth[index])

    if stage_callback is not None:
        stage_callback("generating_confidence")
    confidence = _spline_spatial_confidence(
        valid, train_support, extrapolation, correction, saturated,
        None if spline_holdout_error is None else float(spline_holdout_error),
        int(np.count_nonzero(combined_inliers)), len(train_indices), bound,
    )
    success = reject_reason is None
    if success:
        _overwrite_exact_anchor_pixels(final_depth, confidence, values, pixel_indices)
    else:
        valid[:] = False
        confidence[:] = 0.0
    return SplineSpatialAlignmentResult(
        success, final_depth, valid, confidence, global_depth, correction.astype(np.float32), extrapolation,
        inlier_map, training_metrics, holdout_metrics, knots_x, knots_y,
        spatial_fit.coefficients, None if affine_error is None else float(affine_error), anchor_mask,
        anchor_residual, split_map, {"percentile_1": prediction_low, "percentile_99": prediction_high},
        {
            "unclamped_rms": float(np.sqrt(np.mean(np.square(correction_unclamped)))),
            "unclamped_maximum_absolute": float(np.max(np.abs(correction_unclamped))),
            "saturation_fraction": saturation_fraction,
            "extrapolation_fraction": float(np.mean(extrapolation)),
        }, warnings, reject_reason,
    )


def _spline_spatial_confidence(
    valid: np.ndarray,
    training_support: np.ndarray,
    extrapolation: np.ndarray,
    correction: np.ndarray,
    saturated: np.ndarray,
    holdout_relative_error: float | None,
    inliers: int,
    training_count: int,
    correction_bound: float,
) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for metric-depth confidence maps.") from exc
    distance = cv2.distanceTransform(1 - np.asarray(training_support, np.uint8), cv2.DIST_L2, 3)
    distance_decay = np.exp(-distance / max(min(valid.shape) * 0.25, 1.0))
    quality = max(0.0, 1.0 - float(holdout_relative_error or 0.0))
    quality *= min(float(inliers) / max(training_count, 1), 1.0)
    correction_decay = np.exp(-np.abs(correction) / max(correction_bound, 1e-6))
    confidence = quality * distance_decay * correction_decay
    confidence[extrapolation] *= 0.5
    confidence[saturated] *= 0.25
    confidence[~valid] = 0.0
    return np.clip(confidence, 0.0, 1.0).astype(np.float32)


def _overwrite_exact_anchor_pixels(
    final_depth: np.ndarray,
    confidence: np.ndarray,
    anchors: list[MetricAnchor],
    pixel_indices: np.ndarray,
) -> None:
    selected: dict[tuple[int, int], MetricAnchor] = {}
    for anchor, (u, v) in zip(anchors, pixel_indices):
        key = (int(u), int(v))
        current = selected.get(key)
        priority = 2 if (anchor.source or anchor.provenance) == "marker_surface" else 1
        current_priority = -1 if current is None else (2 if (current.source or current.provenance) == "marker_surface" else 1)
        raw = float(anchor.confidence if anchor.raw_confidence is None else anchor.raw_confidence)
        current_raw = -1.0 if current is None else float(current.confidence if current.raw_confidence is None else current.raw_confidence)
        if current is None or priority > current_priority or (priority == current_priority and raw > current_raw):
            selected[key] = anchor
    for (u, v), anchor in selected.items():
        final_depth[v, u] = np.float32(anchor.z_depth_m)
        if (anchor.source or anchor.provenance) == "marker_surface":
            confidence[v, u] = 1.0
        else:
            raw = anchor.confidence if anchor.raw_confidence is None else anchor.raw_confidence
            confidence[v, u] = max(float(confidence[v, u]), float(np.clip(raw, 0.0, 1.0)))
