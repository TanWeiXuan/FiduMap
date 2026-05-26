# XFeat Vendored Attribution

- **Upstream project name:** accelerated_features (XFeat)
- **Upstream repository URL:** https://github.com/verlab/accelerated_features
- **Exact upstream commit SHA:** e92685f57f8318b18725c5c8c0bd28c7fe188d9a
- **Original license name:** Apache License 2.0
- **Location of copied license file:** `vendor/xfeat/LICENSE`

## Vendored files

- `vendor/xfeat/xfeat.py` (from `modules/xfeat.py`)
- `vendor/xfeat/model.py` (from `modules/model.py`)
- `vendor/xfeat/interpolator.py` (from `modules/interpolator.py`)
- `vendor/xfeat/lighterglue.py` (from `modules/lighterglue.py`)
- `vendor/xfeat/LICENSE` (from `LICENSE`)

## Local modifications

- Added vendoring notice header to all vendored Python files.
- Converted upstream imports in `vendor/xfeat/xfeat.py` from `modules.*` to package-relative imports (`.model`, `.interpolator`, `.lighterglue`).
- Changed default weight loading path in `vendor/xfeat/xfeat.py` to resolve `vendor/xfeat/weights/xfeat.pt` relative to the vendored file location.
- Changed default weight loading path in `vendor/xfeat/lighterglue.py` to resolve `vendor/xfeat/weights/xfeat-lighterglue.pt` relative to the vendored file location.

## Notes

- This is a minimal vendored subset and **not** the full upstream repository.
- If this code is used for research, please cite the XFeat CVPR 2024 paper as requested by upstream.
