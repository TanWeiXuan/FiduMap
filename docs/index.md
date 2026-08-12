# FiduMap documentation

FiduMap is a fiducial-marker based localisation project. The **Map Builder** turns calibrated images containing ArUco/AprilTag markers into an optimised 3D marker map. The `absolute_pose_solver` then detects known markers in runtime images and estimates a camera or rigid camera body's pose in the exported map frame.

## System overview

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

The map builder estimates one rigid `SE(3)` pose per marker rather than four unrelated corner points. This keeps each marker square rigid while still exporting explicit world-frame corner coordinates for consumers that need direct 2D-3D correspondences.

## Documentation map

| Document | Scope |
|---|---|
| [Getting started](getting-started.md) | Installation, GUI launch, test command, and common dependency notes. |
| [Map builder architecture](map-builder.md) | Main package design, workflow, coordinate conventions, output format, and subpackage responsibilities. |
| [Dense reconstruction](dense-reconstruction.md) | Experimental semi-dense matching and point-cloud reconstruction pipeline. |
| [Metric depth maps](metric-depth.md) | Optional offline Depth Anything V2 alignment workflow, storage, export, and viewers. |
| [Repository components](repository-components.md) | Tests, vendored assets, example data, helper scripts, and repository-level folders. |
| [Absolute pose solver](runtime-localisation.md) | Installation and quick start, frames, single- and multi-camera use, tuning, diagnostics, and troubleshooting. |
| [Future plans and notes](future-plans.md) | Incomplete features, known risks, critical bug watchlist, and contribution notes. |

## Repository layout

```text
.
├── docs/                     # Markdown documentation
├── src/
│   ├── camera_calibration/   # ChArUco calibration helper and calibration target
│   ├── map_builder/          # Map construction and shared camera/geometry code
│   └── absolute_pose_solver/ # OpenGV-backed online pose solver
├── tests/                    # Pytest suite
├── vendor/
│   ├── azure_ttk_theme/      # Tkinter theme used by the GUI
│   ├── eigen/                # Eigen submodule used by the native solver
│   ├── opengv/               # OpenGV submodule used by the native solver
│   └── xfeat/                # XFeat model/code for dense reconstruction
├── misc/                     # Placeholder for local/manual artifacts
├── run_map_builder.bat       # Windows launcher for the map builder GUI
└── README.md                 # Short repository introduction
```
