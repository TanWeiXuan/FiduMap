# FiduMap

FiduMap is a fiducial-marker based localisation project. The intended system has two phases:

1. **Map building**: collect calibrated images of an environment containing ArUco/AprilTag markers, optimise the 3D marker map, and export world-frame marker corner coordinates.
2. **Runtime localisation**: detect known markers in a live camera image, match them to the exported map, and estimate the camera pose in the saved map frame.

The repository currently has the **Map Builder** as the main fleshed‑out implementation.  An experimental semi‑dense matching and 3D reconstruction pipeline also lives under `src/map_builder/dense_reconstruction` (see [the dense reconstruction section](#dense_reconstruction-experimental)); this pipeline is incomplete and contributions are welcome.  The runtime localisation side is still not a separate fully implemented module, but the map builder output is designed to support it directly.

## Current implemented workflow

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

## Repository layout

```text
.
├── src/
│   └── map_builder/          # Main implemented package
├── tests/                    # Pytest suite for the map builder
├── vendor/
│   ├── PyRansacLib/          # Git submodule for RANSAC-related utilities
│   ├── azure_ttk_theme/      # Tkinter theme used by the GUI
│   └── xfeat/               # XFeat model and code for dense reconstruction (vendored)
├── misc/                     # Placeholder for local/manual artifacts
├── run_map_builder.bat       # Windows launcher for the map builder GUI
├── .gitmodules               # Submodule definition
└── .gitignore
```

## Installation

From the repository root:

```sh
python -m venv .venv
```

On Windows:

```bat
.venv\Scripts\activate
pip install -r src\map_builder\requirements.txt
pip install pytest
```

On Linux/macOS:

```sh
source .venv/bin/activate
pip install -r src/map_builder/requirements.txt
pip install pytest
```

The main package requirements are:

```text
numpy
opencv-contrib-python
matplotlib
pyceres
```

`opencv-contrib-python` is required because marker detection uses `cv2.aruco`.

### Optional dependencies

The dense reconstruction pipeline (see the [`dense_reconstruction` section](#dense_reconstruction-experimental)) has additional optional dependencies, including **PyTorch** and the vendored **XFeat** model. These are **not** installed by default.  If you plan to experiment with dense reconstruction, install PyTorch for your platform and ensure that `vendor/xfeat` is present (it is provided as part of this repository).  Without these optional components the dense reconstruction stages will remain unavailable.

## Running the GUI

On Windows, run:

```bat
run_map_builder.bat
```

Or directly:

```sh
python src/map_builder/gui/app.py
```

The GUI is intended to let you select an image folder, index images, run marker detection, initialise poses, run bundle adjustment, inspect results, visualise the marker map, and export the final CSV.

## Running tests

From the repository root:

```sh
pytest -q
```

The test suite exercises camera projection/unprojection, marker geometry, project persistence, and synthetic bundle adjustment.  The exact number of passing and skipped tests may vary over time and depends on optional GUI/display availability and whether optional dependencies (such as PyTorch) are installed.  Use the command above to execute the tests on your system and review the results.

---

# Map Builder

The `map_builder` package is the main implemented component in this repository. It builds a persistent 3D fiducial marker map from multiple images with overlapping marker observations.

The most important design decision is that the optimiser estimates **one SE(3) pose per marker**, not four independent 3D points per marker. Each marker is a rigid square with known side length, so its four corners are derived from the marker pose. This preserves the marker geometry during optimisation.

## Map builder goal

The final output is a CSV file of 3D marker corner points in the world/map frame:

```text
id,x,y,z
```

The exported point ID is encoded as:

```text
point_id = (marker_id << 2) | corner_index
```

The intended corner order is:

| Corner index | Meaning |
|---:|---|
| 0 | bottom-left |
| 1 | top-left |
| 2 | top-right |
| 3 | bottom-right |

Marker `0` is typically used as the world anchor:

```text
T_W_M0 = Identity
```

## Coordinate convention

The map builder uses:

```text
T_A_B maps points from frame B into frame A.
```

Important transforms:

| Transform | Meaning |
|---|---|
| `T_W_Mi` | pose of marker `i` in the world/map frame |
| `T_W_Ck` | pose of camera keyframe/image `k` in the world/map frame |
| `T_Ck_W` | inverse of `T_W_Ck` |
| `T_Ck_Mi` | pose of marker `i` relative to camera keyframe `k` |

OpenCV-style PnP returns `T_Ck_Mi`, the transform that maps marker-local 3D points into the camera frame.

During reprojection, the predicted camera-frame point for marker corner `c` is:

```text
X_Ck = T_Ck_W * T_W_Mi * X_Mi,c
```

The camera model then projects `X_Ck` into the image.

## Map builder package layout

```text
src/map_builder/
├── camera_models/     # Camera models and XML I/O
├── detection/         # Marker detection abstractions and OpenCV implementation
├── example_data/      # Sample dataset folders for tests and manual runs
├── export/            # CSV exporter for optimised maps
├── geometry/          # SE3 and marker-corner geometry helpers
├── gui/               # Tkinter GUI
├── initialization/    # PnP, observation graph and seed pose initialisation
├── optimization/      # pyceres bundle adjustment and residuals
├── dense_reconstruction/ # experimental semi-dense matching and 3D reconstruction
├── project/           # SQLite-backed project store and shared dataclasses
├── __init__.py
└── requirements.txt
```

## `camera_models`

This package defines the projection and unprojection layer used throughout the map builder.

Responsibilities:

- define the abstract `CameraModel` API,
- project camera-frame rays to image pixels,
- unproject image pixels to unit camera-frame rays,
- apply or invert Brown-Conrady radial-tangential distortion,
- load and save camera calibration files as XML.

Implemented models:

| Model | Purpose |
|---|---|
| `PinholeRadTanCameraModel` | pinhole camera with radial-tangential distortion |
| `OmniRadTanCameraModel` | omnidirectional camera model with radial-tangential distortion |

Example XML files are in:

```text
src/map_builder/camera_models/example_configs/
```

## `detection`

This package handles marker detection.

Responsibilities:

- define the `MarkerDetector` protocol,
- list supported ArUco and AprilTag dictionaries,
- wrap OpenCV's ArUco/AprilTag detector,
- run detection across indexed images,
- store marker IDs, corner pixels, dictionary metadata and corner-refinement information.

Important files:

| File | Purpose |
|---|---|
| `marker_detector.py` | detector protocol and supported dictionary names |
| `opencv_aruco_detector.py` | OpenCV ArUco/AprilTag detector implementation |
| `detection_runner.py` | runs detection over project images and stores results |

## `example_data`

This folder contains sample dataset folders used for tests and manual experimentation.

The current dataset is:

- `marker_mapper_dataset/`

It includes:

- sample marker images,
- `camera_params.xml`,
- `marker_info.txt`,
- attribution metadata.

## `export`

This package writes the final optimised map.

Responsibilities:

- load optimised marker poses from the project store,
- derive each marker's world-frame corner coordinates,
- encode corner IDs with `(marker_id << 2) | corner_index`,
- write `id,x,y,z` rows to CSV.

Main file:

```text
src/map_builder/export/csv_exporter.py
```

The exported CSV is the bridge between map building and future runtime localisation.

## `geometry`

This package contains the low-level geometric utilities.

Responsibilities:

- represent SE(3) transforms,
- compose and invert transforms,
- transform 3D points between frames,
- convert between rotation matrices, OpenCV rotation vectors and quaternions,
- define marker-local corner coordinates,
- provide detector-order and export-order marker corners.

Important files:

| File | Purpose |
|---|---|
| `se3.py` | rigid transform helper using the `T_A_B` convention |
| `marker_geometry.py` | marker size validation and corner coordinate helpers |

## `gui`

This package contains the Tkinter GUI for interactive map building.

Responsibilities:

- choose an image/project folder,
- display indexed images,
- run detection, initialisation, optimisation and export steps,
- overlay detected markers on images,
- show diagnostics and tables,
- visualise the map in 3D.

Important files:

| File | Purpose |
|---|---|
| `app.py` | GUI entry point |
| `main_window.py` | main application window |
| `control_panel.py` | pipeline controls and settings |
| `image_list_panel.py` | indexed image list |
| `image_viewer_panel.py` | image display and detection overlay |
| `map_3d_viewer_panel.py` | 3D map visualisation |
| `right_panel_tabs.py` | side-panel tabs |
| `menu_bar.py` | menu actions |
| `theme.py` | Azure Tkinter theme loading |

The GUI uses `vendor/azure_ttk_theme`.

## `initialization`

This package computes the initial guesses required by nonlinear optimisation.

Responsibilities:

1. Solve PnP for each detected marker observation.
2. Store `T_Ck_Mi` estimates from PnP.
3. Build the camera-marker observation graph.
4. Build marker-overlap graph diagnostics.
5. Traverse the graph from the anchor marker to initialise camera and marker poses.

Important files:

| File | Purpose |
|---|---|
| `pnp_initializer.py` | per-detection PnP using OpenCV |
| `observation_graph.py` | camera-marker and marker-overlap graph construction |
| `seed_initializer.py` | breadth-first seed pose propagation from the anchor marker |
| `diagnostics.py` | graph and initialisation diagnostics |

The core propagation equations are:

```text
T_W_Ck = T_W_Mi * inverse(T_Ck_Mi)
T_W_Mi = T_W_Ck * T_Ck_Mi
```

If multiple candidate initialisations exist, the implementation prefers the candidate with the lower reprojection error.

## `optimization`

This package performs bundle adjustment.

Responsibilities:

- assemble BA variables and observations,
- optimise camera poses and marker poses,
- keep the anchor marker fixed to remove gauge freedom,
- use image-space marker-corner reprojection residuals,
- apply robust losses,
- compute reprojection error diagnostics,
- reject outlier observations and optionally run a second pass,
- store optimised poses and error records in the project database.

Important files:

| File | Purpose |
|---|---|
| `ba_problem.py` | BA problem container |
| `ba_costs.py` | cost/residual definitions for pyceres |
| `pyceres_backend.py` | pyceres solver backend |
| `optimize_map.py` | high-level BA orchestration |
| `residuals.py` | reprojection error records and statistics |
| `residual_math.py` | lower-level residual math |
| `outlier_rejection.py` | outlier detection rules |
| `diagnostics.py` | optimisation diagnostics |

Conceptually, the optimiser minimises:

```text
sum over observations:
  robust_loss(
    ||u_obs - project(T_Ck_W * T_W_Mi * X_Mi,c)||^2
  )
```

Detector/PnP poses are used only for initialisation. The final optimisation uses the original pixel corner measurements.

## `project`

This package owns persistent project state.

Responsibilities:

- create and manage the `.map_builder/project.sqlite` database inside an image folder,
- store image metadata,
- store detector runs and marker detections,
- store PnP observations,
- store seed camera/marker poses,
- store BA runs, optimised poses and reprojection errors,
- define dataclasses shared across the package.

Important files:

| File | Purpose |
|---|---|
| `project_store.py` | SQLite-backed project database API |
| `models.py` | shared dataclasses |
| `image_indexer.py` | image-folder indexing and metadata refresh |

The side-car database is ignored by git:

```text
.map_builder/
```

## `dense_reconstruction` (experimental)

This package contains an initial implementation of a semi‑dense feature matching and 3D reconstruction pipeline built on top of [XFeat](https://github.com/MobileRoboticsSkoltech/xfeat) features and `pyceres`. It is intended to produce a point cloud from overlapping images after marker‑based bundle adjustment. The pipeline includes:

- semi‑dense feature extraction using the XFeat model (vendored in `vendor/xfeat`);
- candidate frame‑pair selection based on camera poses and marker overlap;
- matching features across selected pairs;
- filtering matches via epipolar geometry;
- triangulating 3D points into the world frame;
- optional point‑cloud bundle adjustment and duplicate‑point merging;
- exporting the resulting point cloud to CSV.

### Limitations and contributions

The dense reconstruction module is **not yet complete**. Parts of the pipeline are under active development. A partially working version has been integrated into the GUI, but more development effort is required to finish it. It also requires additional optional dependencies (most notably PyTorch) that are **not** listed in `requirements.txt`. If these dependencies are missing, the module will report the missing packages and prevent the user from running the corresponding stages.

We welcome community contributions to finish and improve the dense reconstruction pipeline, including algorithmic enhancements, better documentation, integration with the rest of the project, and additional tests.

---

# Other repository components

## `tests`

The `tests` directory covers the current map builder implementation.

Test coverage includes:

- camera projection/unprojection,
- camera XML read/write,
- marker geometry,
- SE(3) utilities,
- image indexing,
- project store persistence,
- marker detection smoke tests,
- PnP initialisation,
- observation graph construction,
- residual math,
- pyceres backend import,
- synthetic bundle adjustment,
- CSV export,
- GUI imports and map viewer logic.

## `vendor`

The `vendor` folder contains third-party code/assets.

| Folder | Purpose |
|---|---|
| `PyRansacLib` | git submodule for RANSAC-related utilities, likely useful for future runtime localisation |
| `azure_ttk_theme` | Tkinter theme files used by the GUI |
| `xfeat` | vendored semi‑dense feature extraction/matching code (XFeat) used by the dense reconstruction pipeline |

When cloning the repository, initialise submodules if needed:

```sh
git submodule update --init --recursive
```

## `misc`

The `misc` folder is a placeholder for local/manual artifacts. Its contents are ignored by git except for `.gitkeep`.

## `run_map_builder.bat`

This Windows convenience launcher starts the GUI from the repository root:

```bat
cd /d "%~dp0"
python src\map_builder\gui\app.py
```

# Runtime localisation direction

The current map builder output is designed for a future runtime localiser. The intended runtime workflow is:

1. Load the exported marker-corner CSV.
2. Detect markers in the current camera image.
3. For each detected marker ID, retrieve the corresponding world-frame marker corners.
4. Build 2D-3D correspondences:

   ```text
   detected image corner pixel <-> saved world-frame marker corner
   ```

5. Run PnP with RANSAC.
6. Refine the camera pose with reprojection residuals.
7. Output the camera pose in the marker-map/world frame.

Because of this intended use, the map builder exports explicit world-frame marker corners even though it internally optimises over rigid marker poses.
