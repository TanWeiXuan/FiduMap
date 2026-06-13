# FiduMap

FiduMap is a fiducial-marker based localisation project. The repository currently focuses on the **Map Builder**, which builds an optimised 3D map of ArUco/AprilTag markers from calibrated images. The exported map is intended to support a future runtime localisation system that estimates camera pose from known markers.

## Documentation

Detailed documentation now lives in the [`docs`](docs/index.md) folder.

| Document | Description |
|---|---|
| [Documentation index](docs/index.md) | Master overview of the repository and links to detailed docs. |
| [Getting started](docs/getting-started.md) | Installation, GUI launch, test command, and dependency notes. |
| [Map builder architecture](docs/map-builder.md) | Workflow, coordinate conventions, output format, and package responsibilities. |
| [Dense reconstruction](docs/dense-reconstruction.md) | Experimental semi-dense matching and point-cloud reconstruction notes. |
| [Repository components](docs/repository-components.md) | Tests, example data, vendored assets, helper scripts, and support folders. |
| [Runtime localisation direction](docs/runtime-localisation.md) | Planned runtime localisation workflow that consumes exported marker maps. |
| [Future plans and notes](docs/future-plans.md) | Incomplete features, critical bug watchlist, and contribution notes. |

## Quick start

Create and activate a virtual environment, then install the map builder requirements:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r src/map_builder/requirements.txt
pip install pytest
```

On Windows, activate with `.venv\Scripts\activate` and use `src\map_builder\requirements.txt`.

Run the GUI:

```sh
python src/map_builder/gui/app.py
```

Or on Windows:

```bat
run_map_builder.bat
```

Run tests:

```sh
pytest -q
```

## Current workflow summary

```text
calibrated camera + marker images
        ↓
image indexing
        ↓
ArUco / AprilTag detection
        ↓
per-marker PnP initialisation
        ↓
camera-marker observation graph
        ↓
seed pose propagation from an anchor marker
        ↓
pyceres bundle adjustment
        ↓
outlier rejection and optional second BA pass
        ↓
CSV export of world-frame marker corners
```

See [Map builder architecture](docs/map-builder.md) for detailed coordinate conventions, package notes, and output format details.
