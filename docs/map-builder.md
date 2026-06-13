# Map builder architecture

The `map_builder` package is the main implemented component in this repository. It builds a persistent 3D fiducial marker map from multiple images with overlapping marker observations.

## Goal and output format

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

## Package layout

```text
src/map_builder/
├── camera_models/          # Camera models and XML I/O
├── detection/              # Marker detection abstractions and OpenCV implementation
├── dense_reconstruction/   # Experimental semi-dense matching and 3D reconstruction
├── example_data/           # Sample dataset folders for tests and manual runs
├── export/                 # CSV exporter for optimised maps
├── geometry/               # SE3 and marker-corner geometry helpers
├── gui/                    # Tkinter GUI
├── initialization/         # PnP, observation graph and seed pose initialisation
├── optimization/           # pyceres bundle adjustment and residuals
├── project/                # SQLite-backed project store and shared dataclasses
├── __init__.py
└── requirements.txt
```

## Subpackages

| Subpackage | Responsibilities | Important files |
|---|---|---|
| `camera_models` | Camera model API, projection, unprojection, distortion, calibration XML I/O. | `base.py`, `pinhole_radtan.py`, `omni_radtan.py`, `xml_io.py` |
| `detection` | Marker detector protocol, supported dictionaries, OpenCV ArUco/AprilTag wrapper, detection runner. | `marker_detector.py`, `opencv_aruco_detector.py`, `detection_runner.py` |
| `export` | Load optimised marker poses, derive marker corners, encode point IDs, write CSV. | `csv_exporter.py` |
| `geometry` | Rigid transforms, composition/inversion, marker-local corner helpers, rotation conversions. | `se3.py`, `marker_geometry.py` |
| `gui` | Tkinter interface for indexing, detection, initialisation, optimisation, visualisation, and export. | `app.py`, `main_window.py`, `control_panel.py`, `image_viewer_panel.py`, `map_3d_viewer_panel.py` |
| `initialization` | Per-detection PnP, observation graphs, marker-overlap diagnostics, seed pose propagation. | `pnp_initializer.py`, `observation_graph.py`, `seed_initializer.py`, `diagnostics.py` |
| `optimization` | Bundle adjustment problem construction, pyceres backend, robust residuals, diagnostics, outlier rejection. | `ba_problem.py`, `ba_costs.py`, `pyceres_backend.py`, `optimize_map.py`, `residuals.py`, `outlier_rejection.py` |
| `project` | `.map_builder/project.sqlite` creation, image metadata, detections, PnP observations, seed poses, BA results. | `project_store.py`, `models.py`, `image_indexer.py` |

## Initialisation equations

Seed pose propagation uses observations to alternate between known marker and camera poses:

```text
T_W_Ck = T_W_Mi * inverse(T_Ck_Mi)
T_W_Mi = T_W_Ck * T_Ck_Mi
```

If multiple candidate initialisations exist, the implementation prefers the candidate with the lower reprojection error.

## Optimisation objective

Detector/PnP poses are used only for initialisation. Final optimisation uses original pixel corner measurements and conceptually minimises:

```text
sum over observations:
  robust_loss(
    ||u_obs - project(T_Ck_W * T_W_Mi * X_Mi,c)||^2
  )
```

The anchor marker is fixed to remove gauge freedom.
