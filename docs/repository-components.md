# Repository components

## Tests

The `tests` directory covers the current map builder implementation. Coverage includes:

- camera projection and unprojection;
- camera XML read/write;
- marker geometry;
- SE(3) utilities;
- image indexing;
- project store persistence;
- marker detection smoke tests;
- PnP initialisation;
- observation graph construction;
- residual math;
- pyceres backend import;
- synthetic bundle adjustment;
- CSV export;
- GUI imports and map viewer logic;
- dense reconstruction availability, metadata, matching, pair selection, triangulation, export, and duplicate merging checks.

Run the suite with:

```sh
pytest -q
```

## Example data

`src/map_builder/example_data` contains sample dataset folders used for tests and manual experimentation. The current marker mapper dataset includes sample marker images, `camera_params.xml`, `marker_info.txt`, and attribution metadata.

## Camera calibration helper

`src/camera_calibration` contains a ChArUco calibration helper and a printable ChArUco target PDF. This is separate from the map builder package but supports the calibrated-camera input expected by the map builder workflow.

## Vendored assets

| Folder | Purpose |
|---|---|
| `vendor/azure_ttk_theme` | Tkinter theme files used by the GUI. |
| `vendor/xfeat` | Vendored semi-dense feature extraction/matching code and model weights used by dense reconstruction. |

## Miscellaneous folder

`misc` is a placeholder for local/manual artifacts. Its contents are ignored by git except for `.gitkeep`.

## Windows launcher

`run_map_builder.bat` starts the GUI from the repository root:

```bat
cd /d "%~dp0"
python src\map_builder\gui\app.py
```
