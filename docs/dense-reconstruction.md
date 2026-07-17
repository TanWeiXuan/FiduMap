# Dense reconstruction

`src/map_builder/dense_reconstruction` contains an experimental semi-dense feature matching and 3D reconstruction pipeline built on top of XFeat features and `pyceres`.

## Intended purpose

After marker-based bundle adjustment has estimated camera poses, dense reconstruction produces an additional world-frame point cloud from overlapping images. Marker-map BA remains the metric backbone: dense reconstruction keeps those camera poses fixed and refines only the reconstructed points.

## Pipeline stages

The package includes code for:

1. semi-dense feature extraction using the vendored XFeat model in `vendor/xfeat`;
2. candidate frame-pair selection using marker overlap, camera orientation, and useful baseline;
3. refined feature matching across selected pairs;
4. epipolar filtering using the optimized marker-map camera poses;
5. confidence-ordered track construction and robust multiview triangulation;
6. optional structure-only point bundle adjustment;
7. duplicate-point merging; and
8. CSV point-cloud export.

The GUI exposes each stage separately and also provides **Run Full Dense Reconstruction**. The automatic workflow stops at the first stage that produces no usable output and reports the reason. Dense point BA is skipped when `pyceres` is unavailable; export remains an explicit user action.

## Quality controls

The dense reconstruction panel exposes the main quality/compute controls, including maximum keypoints, pair count, required common markers, baseline and orientation limits, descriptor score, epipolar error, triangulation angle, reprojection error, and duplicate radius.

The status area reports the reconstruction funnel:

```text
feature images → selected pairs → refined matches → epipolar inliers → tracks → active points
```

Multiview triangulation may remove a small number of inconsistent observations before rejecting a track. Structure-only BA applies post-solve reprojection checks and deactivates points that no longer meet the configured quality thresholds.

## Important files

| File | Purpose |
|---|---|
| `availability.py` | Reports optional dependency availability. |
| `dense_pipeline.py` | High-level orchestration, stage diagnostics, and full-pipeline execution. |
| `dense_store.py` | Persistence helpers for dense reconstruction artifacts. |
| `pair_selection.py` | Chooses candidate image pairs. |
| `xfeat_extractor.py` | Feature extraction via XFeat. |
| `xfeat_matching.py` | Feature matching support. |
| `epipolar_filter.py` | Geometry-based match filtering. |
| `track_builder.py` | Builds confidence-ordered multiframe tracks. |
| `triangulation.py` | Robustly converts matched observations into world-frame 3D points. |
| `point_ba.py` | Structure-only point-cloud bundle adjustment and post-solve rejection. |
| `duplicate_merge.py` | Duplicate point consolidation. |
| `point_cloud_export.py` | CSV export for active dense points. |

## Limitations

The dense reconstruction module remains experimental. It requires optional dependencies, most notably PyTorch, that are not listed in the default requirements file. Camera poses are intentionally fixed to the marker-map BA result; joint camera/point optimization is not implemented. Duplicate merging is intentionally performed after point BA, and merged points do not currently retain the full union of all source-track observations.
