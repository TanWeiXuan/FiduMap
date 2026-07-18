# Metric depth maps

Metric Depth Maps is an optional offline workflow that creates a depth artifact for each optimized reference image. It does not perform runtime matching, online localization, or PnP.

## Backends

- **Depth Anything V2 Small — aligned** loads `depth-anything/Depth-Anything-V2-Small-hf`, predicts relative depth, and robustly fits a global affine relationship in inverse depth: `1/z = a * prediction + b`. This produces globally aligned metric depth, but local errors can remain, especially far from trusted anchors.
- **Prompt Depth Anything Small** loads `depth-anything/prompt-depth-anything-vits-hf`, passes sparse metric camera-Z prompts through the official Transformers processor, and verifies the metric output at the trusted anchors. Passing verification does not guarantee correctness in unprompted regions.

Both default Hugging Face checkpoints identify their license as Apache-2.0. The implementations and papers are attributed to the [Depth Anything V2 project](https://github.com/DepthAnything/Depth-Anything-V2) and [Prompt Depth Anything project](https://github.com/DepthAnything/PromptDA). Review each model card and project license before redistribution or commercial use.

## Installation and offline behavior

The base application does not require the inference or VTK packages. Install the optional stack with:

```sh
python -m pip install -r src/map_builder/requirements-depth.txt
```

The compatibility target pins Transformers 4.56.2, which contains the model and image-processor APIs for both backends. PyTorch is constrained to `>=2.2,<3`; the remaining optional ranges are recorded in the requirements file.

The model reference can be a Hugging Face ID or a local Transformers model directory. **Allow model download** is off by default. In that mode all `from_pretrained` calls use `local_files_only=True`, so an uncached ID or invalid directory fails clearly and no network download is attempted. Enabling the checkbox lets Transformers use its normal Hugging Face cache and download behavior; FiduMap has no custom downloader or checkpoint manager.

Models are loaded once per batch, reused across images, and images are processed sequentially. CPU inference can be slow, but memory remains bounded to one image/model forward pass at a time. Cancellation is checked between stages and images, not during a model forward pass.

## Trusted sparse prompts

One common builder supplies both backends:

1. Active dense points are accepted only when their track has an actual observation in the selected image. The world point is transformed with `X_C = R_W_C.T @ (X_W - t_W_C)`. Behind-camera and non-finite points are rejected. Confidence is a bounded function of observation count and available reprojection diagnostics.
2. Markers detected in the image and present in the current optimized marker map are rasterized only inside their detected polygon. Camera rays are intersected with the finite known marker square; the plane is never extended beyond its boundary. These exact surface prompts have highest priority and confidence.

Missing prompt pixels remain zero and are represented separately by a mask. Collisions prefer marker surfaces, then confidence, then the nearer equal-confidence depth. Coverage is measured on a fixed 4×4 grid. No wall-plane propagation, occlusion guessing, or projection of every global dense point is performed.

## Z-depth and radial range

Models consume and produce camera Z-depth. Stored/exported canonical depth is radial range along the unit camera bearing:

```text
ray_C = camera_model.unproject(pixel)
range_m = z_depth_m / ray_C.z
X_C = range_m * ray_C
```

Conversion requires a finite forward ray (`ray_C.z > 0`) and finite positive Z-depth. Rear-hemisphere pixels are invalid. This workflow does not implement virtual-pinhole tiling, so learned inference over very wide-angle or fisheye imagery has no full-field correction.

## Alignment, verification, and confidence

DAV2 alignment uses confidence-weighted NumPy least squares with iterative MAD rejection. It rejects insufficient anchor count or grid coverage, anchors without meaningful depth variation, too few robust inliers, non-finite coefficients, more than 25% non-positive fitted inverse depths, and excessive median relative anchor error.

PromptDA rejects insufficient prompt support, too few positive finite outputs, or median anchor absolute/relative error above the configured limits. Sparse prompts are adapted with valid-aware nearest sampling so missing zeros are never averaged with valid depths.

Confidence is deliberately simple: a global residual/inlier quality score multiplied by OpenCV distance-to-prompt decay and prompt confidence. It is **not a calibrated probability**. Only finite, positive, confidence-qualified depth should be considered by any future localization code.

## Storage, staleness, and export

Metadata is stored in `.map_builder/metric_depth.sqlite`; arrays are atomic NPZ files under `.map_builder/metric_depth_maps/<run_id>/<image_id>.npz`. Arrays never enter the main project database. A product is stale when its marker BA run ID differs from the current successful BA run. The dense signature records active point count, track count, and maximum active point ID for diagnostics.

Canonical exports contain an NPZ with `z_depth_m`, `range_m`, `valid_mask`, `confidence`, `prompt_depth_z_m`, and `prompt_mask`, plus JSON metadata containing image/camera/pose/run conventions and quality metrics. Portable exports add uint16 millimetre range PNG (zero means invalid), uint8 confidence PNG, and prompt-mask PNG. Values above 65.535 m are written as invalid zero and explicitly flagged/count-recorded in metadata; they never wrap.

## GUI workflow

After successful marker BA, open the third left tab, **Metric Depth Maps**, select backend/model/device and prompt sources, then generate the selected optimized image or all optimized images. Batch failures are logged per image and do not abort later images. Export buttons require successful products.

The Image Viewer adds RGB, radial range, camera Z, confidence, prompt depth, and RGB-overlay modes with robust display percentiles and per-pixel metric readout. It does not normalize exported arrays. The third right tab, **Depth 3D View**, lazily back-projects a deterministically decimated point cloud through the project camera and `T_W_C`. VTK is optional; without it the tab shows the install command and the rest of the GUI remains usable.

The viewer first attempts VTK's native `vtkTkRenderWindowInteractor`. Standard binary VTK wheels can omit the native RenderingTk library, particularly on Windows. If that bridge cannot load, FiduMap automatically uses an embedded Tk canvas backed by VTK off-screen rendering; orbit, pan, zoom, reset, coloring, and scene overlays remain available. If both rendering paths fail, only the Depth 3D View is disabled and the tab reports the precise error.

## Known limitations

- No rear-hemisphere or virtual-pinhole learned inference support.
- DAV2 alignment is one global affine inverse-depth fit and may be locally inaccurate.
- PromptDA is verified only at trusted anchors; unprompted regions can be wrong.
- Confidence is heuristic, not probabilistic.
- No cross-view depth verification/fusion, meshing, training, online feature matching, PnP, or runtime localization is included.
