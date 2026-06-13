# Dense reconstruction

`src/map_builder/dense_reconstruction` contains an experimental semi-dense feature matching and 3D reconstruction pipeline built on top of XFeat features and `pyceres`.

## Intended purpose

After marker-based bundle adjustment has estimated camera poses, dense reconstruction is intended to produce an additional point cloud from overlapping images.

## Pipeline stages

The package includes code for:

1. semi-dense feature extraction using the vendored XFeat model in `vendor/xfeat`;
2. candidate frame-pair selection based on camera poses and marker overlap;
3. feature matching across selected pairs;
4. epipolar filtering of matches;
5. triangulation of 3D points into the world frame;
6. optional point-cloud bundle adjustment;
7. duplicate-point merging; and
8. CSV point-cloud export.

## Important files

| File | Purpose |
|---|---|
| `availability.py` | Reports optional dependency availability. |
| `dense_pipeline.py` | High-level orchestration for dense reconstruction stages. |
| `dense_store.py` | Persistence helpers for dense reconstruction artifacts. |
| `pair_selection.py` | Chooses candidate image pairs. |
| `xfeat_extractor.py` | Feature extraction via XFeat. |
| `xfeat_matching.py` | Feature matching support. |
| `epipolar_filter.py` | Geometry-based match filtering. |
| `triangulation.py` | Converts matched observations into 3D points. |
| `point_ba.py` | Point-cloud bundle adjustment. |
| `duplicate_merge.py` | Duplicate point consolidation. |
| `point_cloud_export.py` | CSV export for dense points. |

## Limitations

The dense reconstruction module is incomplete and should be considered experimental. It requires optional dependencies, most notably PyTorch, that are not listed in the default requirements file. A partially working version is integrated into the GUI, but additional development is needed before this should be treated as production-ready.
