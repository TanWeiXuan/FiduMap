# Metric depth maps

Metric Depth Maps is an optional offline workflow that creates a depth artifact for each optimized reference image. It does not perform runtime matching, online localization, or PnP.

## Depth Anything V2 Small alignment

The workflow uses `depth-anything/Depth-Anything-V2-Small-hf` and offers two independently fitted per-image alignment methods.

- **Monotonic spline + spatial correction** is the GUI default. It models `log(z) = g(prediction) + h(u,v)`, where `g` is a robust monotonically decreasing piecewise-linear spline and `h` is a small smooth bilinear control grid.
- **Robust affine inverse depth** preserves the original baseline, `1/z = a*prediction + b`. Saved configurations created before the alignment selector existed load with this compatibility mode.

Depth Anything V2 output is treated as disparity-like: a larger prediction represents a nearer surface and therefore a smaller metric depth. The global mapping from normalized DAV2 prediction to metric log-depth is constrained to be monotonically decreasing. It uses weighted pool-adjacent-violators isotonic regression, compresses the result to 12 weighted-quantile knots by default, and clamps values beyond the fitted training range to the nearest endpoint. The 8-by-6 spatial grid fits training-anchor log-depth residuals with robust reweighting, Laplacian smoothness, and a zero prior. Its applied correction is bounded to `+/-0.4` log-depth by default. No SciPy, MLP, per-pixel optimizer, or neural training is used.

Both methods remain per-image alignments. The spline-spatial method can model nonlinear and gradual image-location bias, but it does **not** guarantee consistency between images.

## Installation and offline behavior

Install the optional inference and viewer stack with:

```sh
python -m pip install -r src/map_builder/requirements-depth.txt
```

**Allow model download** is off by default. In that mode Transformers uses `local_files_only=True`, so an uncached ID or invalid directory fails clearly without a network request. The model is loaded once per batch and reused across sequential images. Fitting time and DAV2 inference time are recorded separately. Work runs on the existing background worker; Tk's main thread only handles progress and display updates.

## Balanced metric anchors

Active dense points contribute one anchor only when the track has a stored observation in the image. The metric point is transformed using:

```text
X_C = R_W_C.T @ (X_W - t_W_C)
```

Only finite forward points are accepted. Reprojection diagnostics and observation count produce bounded raw confidence, which distributes weight within the dense-track source.

Each visible optimized marker contributes a deterministic 6-by-6 grid in the finite marker-local square, independent of its projected pixel area. Samples are transformed through `T_W_M` and `T_W_C`, projected with the authoritative camera model, and retained only when finite, in bounds, and inside the detected marker polygon. The marker plane is never extended.

When both sources exist they receive equal total fit weight. Visible markers receive equal shares of the marker total, divided over their valid samples; dense tracks divide their source total according to raw confidence. Stored diagnostics retain raw confidence and final fit weight. Pixel collisions still prefer markers, then confidence.

## Deterministic validation

Anchors are split reproducibly using SHA-256 over image ID, source, group ID, and pixel coordinates. The default is 80% training and 20% holdout, stratified by source. Complete marker or track groups are held out when this leaves enough training support; otherwise a deterministic sample split is used. The minimum configured training count is always preserved.

Held-out anchors do not affect spline fitting, spatial fitting, robust inlier selection, or regularization. They estimate generalization error, determine whether the fitted map is accepted, provide marker-only and dense-track-only validation metrics, and support comparison with the affine baseline. Training, holdout, marker-only holdout, dense-track-only holdout, and same-split affine-baseline errors are reported separately. A result is rejected for insufficient training coverage, an unfittable spline, singular/non-finite spatial solve, excessive invalid depth, excessive holdout relative error, or excessive correction saturation. A validated spline-spatial result is never silently replaced by affine output.

Metrics are evaluated before any anchor overwrite. After successful validation, trusted anchor pixels receive exact metric Z-depth. Marker/track collision priority remains unchanged.

## Confidence and extrapolation

Spline-spatial confidence combines held-out alignment quality, robust training-inlier fraction, distance to the nearest training anchor, spline-range support, spatial-correction magnitude, and correction saturation. OpenCV's distance transform supplies the support distance. Confidence is zero for invalid depth and reduced outside the fitted spline range, far from training anchors, for large corrections, and where the correction bound is reached. Exact marker anchors receive high confidence; dense-track anchor confidence follows their quality.

Confidence is a heuristic, **not a calibrated probability**. Sparse anchors do not make unsupported pixels metrically observed. Smoothness, the zero prior, and the DAV2 prediction govern unsupported regions.

## Z-depth and radial range

Alignment uses camera Z-depth. Canonical storage also contains radial range along the unit camera bearing:

```text
ray_C = camera_model.unproject(pixel)
range_m = z_depth_m / ray_C.z
X_C = range_m * ray_C
```

Only finite forward rays and finite positive depths are valid.

## Artifacts and compatibility

Metadata is stored in `.map_builder/metric_depth.sqlite`; arrays are atomic NPZ files under `.map_builder/metric_depth_maps/<run_id>/<image_id>.npz`. Full arrays are never stored in SQLite.

Every artifact retains the standard final-output arrays:

```text
z_depth_m
range_m
valid_mask
confidence
```

Schema-v2 spline-spatial artifacts can additionally contain global spline depth, clamped spatial log correction, extrapolation mask, anchor mask, sparse pre-overwrite residual, train/holdout split, and DAV2 prediction. Metadata records normalization percentiles, spline knots, grid dimensions/coefficients, regularization, bounds, training and holdout metrics, affine baseline metrics, warnings, and timings. Old artifacts with only standard arrays remain readable; obsolete direction metadata is ignored, and diagnostic display modes report unavailable data instead of failing.

## GUI and diagnostic views

The Metric Depth Maps tab provides the two-method selector and a separated advanced section for knot count, grid columns/rows, maximum correction, and holdout fraction. Per-artifact summaries include method, split counts, source counts, training/holdout errors, affine comparison, spatial statistics, saturation, and extrapolation.

The Image Viewer preserves RGB, radial range, camera Z, confidence, overlays, markers, XFeat, pan, zoom, and pixel readout. Spline-spatial artifacts add Global spline depth, Spatial correction, Final aligned depth, Anchor residual, Anchor split, and Spline extrapolation. Spatial correction and residual use zero-centered diverging colors; residuals remain sparse rather than being interpolated into unsupported areas.

## Limitations

- Alignment is independently fitted for each image and does not guarantee cross-image consistency.
- Sparse anchors do not uniquely determine unsupported regions.
- Held-out anchor accuracy is more meaningful than training residual, but it only measures sampled support.
- Multi-view verification is still required for localization-grade dense geometry.
- There is no depth fusion, TSDF, surfels, meshing, stereo, camera-pose optimization, runtime matching, or PnP in this workflow.
