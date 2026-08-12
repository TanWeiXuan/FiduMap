# FiduMap

FiduMap is a fiducial-marker based localisation project. The **Map Builder** builds an optimised 3D map of ArUco/AprilTag markers from calibrated images, and the OpenGV-backed **absolute pose solver** estimates a camera or rigid camera body's pose from known markers in that map.

## Documentation

Detailed documentation now lives in the [`docs`](docs/index.md) folder.

| Document | Description |
|---|---|
| [Documentation index](docs/index.md) | Master overview of the repository and links to detailed docs. |
| [Getting started](docs/getting-started.md) | Installation, GUI launch, test command, and dependency notes. |
| [Map builder architecture](docs/map-builder.md) | Workflow, coordinate conventions, output format, and package responsibilities. |
| [Dense reconstruction](docs/dense-reconstruction.md) | Experimental semi-dense matching and point-cloud reconstruction notes. |
| [Metric depth maps](docs/metric-depth.md) | Optional offline metric-depth generation, verification, viewing, and export. |
| [Repository components](docs/repository-components.md) | Tests, example data, vendored assets, helper scripts, and support folders. |
| [Absolute pose solver](docs/runtime-localisation.md) | Quick start, single- and multi-camera usage, frames, tuning, diagnostics, and troubleshooting. |
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

Build and install the absolute pose solver from a source checkout:

```sh
git submodule update --init --recursive
python -m pip install .
```

See the [absolute pose solver quick start](docs/runtime-localisation.md#quick-start) for map export, camera setup, marker detection, and pose estimation.

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
