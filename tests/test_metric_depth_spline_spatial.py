import numpy as np

from map_builder.camera_models import PinholeRadTanCameraModel
from map_builder.geometry import SE3
from map_builder.metric_depth.alignment import (
    compress_decreasing_isotonic_to_knots,
    deterministic_anchor_split,
    evaluate_monotonic_spline,
    evaluate_spatial_grid,
    fit_robust_decreasing_isotonic,
    fit_spatial_correction_grid,
    monotonic_spline_spatial_alignment,
    weighted_decreasing_isotonic_regression,
)
from map_builder.metric_depth.anchor_builder import (
    balance_anchor_weights,
    marker_surface_anchors,
    rasterize_anchors,
)
from map_builder.metric_depth.backends.depth_anything_v2 import DepthAnythingV2AlignedBackend
from map_builder.metric_depth.models import MetricAnchor, MetricDepthConfig
from map_builder.project import MarkerDetection, OptimizedMarkerPose


def _anchor(u, v, z, source, group, confidence=1.0):
    return MetricAnchor(u, v, z, z, confidence, source, group, source, confidence, confidence)


def test_marker_grid_count_does_not_depend_on_projected_area():
    pose = SE3(np.eye(3), np.array([0.0, 0.0, 2.0]))
    optimized = [OptimizedMarkerPose(17, 1, pose.to_json_dict())]
    counts = []
    for focal, half_pixels in ((20.0, 5.0), (80.0, 20.0)):
        camera = PinholeRadTanCameraModel(101, 101, focal, focal, 50, 50, 0, 0, 0, 0, 0)
        corners = [[50-half_pixels, 50-half_pixels], [50+half_pixels, 50-half_pixels], [50+half_pixels, 50+half_pixels], [50-half_pixels, 50+half_pixels]]
        detection = MarkerDetection("aruco", "DICT_6X6_250", 17, corners, "none")
        anchors = marker_surface_anchors([detection], optimized, SE3.identity(), camera, 1.0, 101, 101)
        counts.append(len(anchors))
        assert {anchor.group_id for anchor in anchors} == {"marker:17"}
    assert counts == [36, 36]


def test_source_and_marker_group_weights_are_balanced():
    anchors = [
        *[_anchor(i, 0, 2, "marker_surface", "marker:1") for i in range(3)],
        *[_anchor(i, 1, 2, "marker_surface", "marker:2") for i in range(9)],
        _anchor(0, 2, 2, "dense_track", "track:1", 0.2),
        _anchor(1, 2, 2, "dense_track", "track:2", 0.8),
    ]
    balanced = balance_anchor_weights(anchors)
    totals = {}
    groups = {}
    for anchor in balanced:
        totals[anchor.source] = totals.get(anchor.source, 0.0) + anchor.fit_weight
        groups[anchor.group_id] = groups.get(anchor.group_id, 0.0) + anchor.fit_weight
    assert np.isclose(totals["marker_surface"], 0.5)
    assert np.isclose(totals["dense_track"], 0.5)
    assert np.isclose(groups["marker:1"], groups["marker:2"])
    dense = [anchor.fit_weight for anchor in balanced if anchor.source == "dense_track"]
    assert np.isclose(dense[1] / dense[0], 4.0)


def test_deterministic_stratified_split_never_overlaps_training():
    anchors = balance_anchor_weights([
        *[_anchor(i, 2, 2 + i / 100, "marker_surface", f"marker:{i // 6}") for i in range(24)],
        *[_anchor(i, 8, 2 + i / 100, "dense_track", f"track:{i}") for i in range(24)],
    ])
    first = deterministic_anchor_split(anchors, 41, 0.2, 12)
    second = deterministic_anchor_split(anchors, 41, 0.2, 12)
    assert np.array_equal(first.training_mask, second.training_mask)
    assert np.array_equal(first.holdout_mask, second.holdout_mask)
    assert not np.any(first.training_mask & first.holdout_mask)
    assert np.count_nonzero(first.training_mask) >= 12
    assert any(a.source == "marker_surface" for a, use in zip(anchors, first.holdout_mask) if use)
    assert any(a.source == "dense_track" for a, use in zip(anchors, first.holdout_mask) if use)


def test_decreasing_pava_preserves_monotonic_samples():
    fitted = weighted_decreasing_isotonic_regression(
        np.array([0.0, 1.0, 2.0]), np.array([3.0, 2.0, 1.0])
    )
    assert np.allclose(fitted, [3.0, 2.0, 1.0])


def test_decreasing_pava_pools_violations():
    fitted = weighted_decreasing_isotonic_regression(
        np.array([0.0, 1.0, 2.0]), np.array([3.0, 1.0, 2.0])
    )
    assert np.allclose(fitted, [3.0, 1.5, 1.5])


def test_decreasing_pava_uses_weighted_pooling():
    fitted = weighted_decreasing_isotonic_regression(
        np.array([0.0, 1.0, 2.0]), np.array([3.0, 1.0, 2.0]), np.array([1.0, 3.0, 1.0])
    )
    assert np.allclose(fitted, [3.0, 1.25, 1.25])


def test_decreasing_pava_combines_duplicate_predictions():
    x = np.array([1.0, 0.0, 0.0, 2.0])
    fitted = weighted_decreasing_isotonic_regression(
        x, np.array([2.0, 4.0, 2.0, 1.0]), np.array([1.0, 1.0, 3.0, 1.0])
    )
    duplicate = fitted[x == 0.0]
    assert np.allclose(duplicate, 2.5)
    assert np.all(np.diff(fitted[np.argsort(x, kind="mergesort")]) <= 1e-12)


def test_robust_decreasing_isotonic_downweights_outlier():
    x = np.arange(7, dtype=float)
    y = np.array([7.0, 6.0, 5.0, 12.0, 3.0, 2.0, 1.0])
    fitted, robust, _ = fit_robust_decreasing_isotonic(x, y, np.ones(7))
    assert robust[3] < 1.0
    assert np.all(np.diff(fitted) <= 1e-12)


def test_decreasing_knot_compression_evaluation_and_extrapolation():
    x = np.linspace(0.0, 1.0, 20)
    fitted = 2.0 - x**2
    knots_x, knots_y = compress_decreasing_isotonic_to_knots(x, fitted, np.ones_like(x), 7)
    assert np.all(np.diff(knots_y) <= 1e-12)
    samples = np.linspace(knots_x[0], knots_x[-1], 101)
    evaluated, extrapolated = evaluate_monotonic_spline(samples, knots_x, knots_y)
    assert np.all(np.diff(evaluated) <= 1e-12)
    assert not np.any(extrapolated)
    clamped, outside = evaluate_monotonic_spline(
        np.array([-1.0, knots_x[0], knots_x[-1], 2.0]), knots_x, knots_y
    )
    assert outside.tolist() == [True, False, False, True]
    assert clamped[0] == knots_y[0]
    assert clamped[-1] == knots_y[-1]


def test_spatial_grid_recovers_smooth_field_and_bounds_are_measurable():
    height, width = 72, 96
    yy, xx = np.mgrid[:height, :width]
    x, y = xx / (width - 1), yy / (height - 1)
    expected = 0.12 * x - 0.08 * y + 0.05 * np.sin(np.pi*x) * np.sin(np.pi*y)
    points = np.column_stack((xx[4::10, 4::12].ravel(), yy[4::10, 4::12].ravel()))
    residual = expected[np.rint(points[:, 1]).astype(int), np.rint(points[:, 0]).astype(int)]
    residual[3] += 1.0
    fit = fit_spatial_correction_grid(points, residual, np.ones(len(points)), width, height, 8, 6)
    assert fit.success
    recovered = evaluate_spatial_grid(fit.coefficients, width, height)
    assert np.median(np.abs(recovered - expected)) < 0.04
    bounded = np.clip(recovered * 20.0, -0.4, 0.4)
    assert np.max(np.abs(bounded)) <= 0.4
    assert np.mean(np.abs(recovered * 20.0) >= 0.4) > 0.0


def _combined_problem():
    height, width = 72, 96
    yy, xx = np.mgrid[:height, :width]
    prediction = 0.7 * xx / (width - 1) + 0.3 * yy / (height - 1)
    x, y = xx / (width - 1), yy / (height - 1)
    log_z = 2.0 - 0.8 * prediction - 0.8 * prediction**2 + 0.20*x - 0.14*y + 0.08*np.sin(np.pi*x)*np.sin(np.pi*y)
    depth = np.exp(log_z)
    anchors = []
    for index in range(48):
        u = (3 + index * 17) % width
        v = (4 + index * 11) % height
        source = "marker_surface" if index < 24 else "dense_track"
        group = f"marker:{index // 6}" if source == "marker_surface" else f"track:{index}"
        value = float(depth[v, u]) * (1.8 if index in {7, 31} else 1.0)
        anchors.append(_anchor(u, v, value, source, group, 0.8 if source == "dense_track" else 1.0))
    return prediction, depth, balance_anchor_weights(anchors)


def test_decreasing_spline_spatial_alignment_is_bounded_and_deterministic():
    prediction, expected, anchors = _combined_problem()
    config = MetricDepthConfig(minimum_anchor_count=20, maximum_alignment_median_relative_error=0.5, spatial_grid_columns=6, spatial_grid_rows=5)
    first = monotonic_spline_spatial_alignment(prediction, anchors, 9, config)
    second = monotonic_spline_spatial_alignment(prediction, anchors, 9, config)
    assert first.success and second.success
    assert np.array_equal(first.final_z_depth_m, second.final_z_depth_m)
    assert np.all(np.diff(first.spline_knots_y) <= 1e-12)
    assert first.holdout_metrics["median_relative_error"] < first.affine_holdout_median_relative_error
    assert first.holdout_metrics["median_relative_error"] < 0.5
    assert np.all(np.isfinite(first.final_z_depth_m[first.valid_mask]))
    assert np.all(first.final_z_depth_m[first.valid_mask] > 0.0)
    assert first.correction_statistics["unclamped_maximum_absolute"] < 0.4
    for anchor in anchors:
        assert np.isclose(first.final_z_depth_m[round(anchor.v), round(anchor.u)], anchor.z_depth_m)
    assert first.holdout_metrics["median_relative_error"] > 0.0
    assert np.mean(first.confidence[first.extrapolation_mask]) <= np.mean(first.confidence[~first.extrapolation_mask])


def test_combined_method_fails_cleanly_for_insufficient_or_constant_prediction():
    prediction, _expected, anchors = _combined_problem()
    config = MetricDepthConfig(minimum_anchor_count=20)
    insufficient = monotonic_spline_spatial_alignment(prediction, anchors[:5], 1, config)
    constant = monotonic_spline_spatial_alignment(np.ones_like(prediction), anchors, 1, config)
    assert not insufficient.success and "insufficient" in insufficient.error_message
    assert not constant.success and "range" in constant.error_message


def test_backend_dispatch_progress_metadata_and_diagnostics_without_model_weights():
    prediction, _expected, anchors = _combined_problem()
    raster = rasterize_anchors(anchors, prediction.shape[1], prediction.shape[0])
    raster.image_id = 23

    class FakePredictionBackend(DepthAnythingV2AlignedBackend):
        def predict_relative(self, _image, _config):
            return prediction.astype(np.float32)

    backend = FakePredictionBackend()
    camera = PinholeRadTanCameraModel(prediction.shape[1], prediction.shape[0], 80, 80, 48, 36, 0, 0, 0, 0, 0)
    events = []
    config = MetricDepthConfig(minimum_anchor_count=20, maximum_alignment_median_relative_error=0.5, spatial_grid_columns=6, spatial_grid_rows=5)
    artifact = backend.infer(np.zeros((*prediction.shape, 3), np.uint8), raster, camera, config, events.append)
    stages = [event.stage for event in events]
    assert stages == [
        "building_balanced_anchors", "splitting_anchors", "fitting_spline",
        "fitting_spatial_correction", "evaluating_holdout", "generating_confidence",
    ]
    assert artifact.metrics.status == "success"
    assert not hasattr(artifact.metrics, "spline_direction")
    assert artifact.metadata["alignment_mode"] == "monotonic_spline_spatial"
    assert "spline_direction" not in artifact.metadata
    assert artifact.metadata["dav2_inference_duration_s"] >= 0.0
    assert artifact.global_spline_z_depth_m is not None
    assert artifact.anchor_split is not None
