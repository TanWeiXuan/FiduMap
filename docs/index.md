# FiduMap documentation

FiduMap is a fiducial-marker based localisation project. The repository currently focuses on a **Map Builder** that turns calibrated images containing ArUco/AprilTag markers into an optimised 3D marker map. The intended downstream phase is runtime localisation: detecting known markers in live camera images and estimating the camera pose in the exported map frame.

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
| [Repository components](repository-components.md) | Tests, vendored assets, example data, helper scripts, and repository-level folders. |
| [Runtime localisation direction](runtime-localisation.md) | Intended future localiser workflow that consumes exported marker maps. |
| [Future plans and notes](future-plans.md) | Incomplete features, known risks, critical bug watchlist, and contribution notes. |

## Repository layout

```text
.
├── docs/                     # Markdown documentation
├── src/
│   ├── camera_calibration/   # ChArUco calibration helper and calibration target
│   └── map_builder/          # Main implemented package
├── tests/                    # Pytest suite for the map builder
├── vendor/
│   ├── azure_ttk_theme/      # Tkinter theme used by the GUI
│   └── xfeat/                # XFeat model/code for dense reconstruction
├── misc/                     # Placeholder for local/manual artifacts
├── run_map_builder.bat       # Windows launcher for the map builder GUI
└── README.md                 # Short repository introduction
```
