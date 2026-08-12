# Absolute pose solver

The `absolute_pose_solver` package estimates the world pose of a camera, or of
a rigid multi-camera body, from detected markers in a map exported by the
FiduMap Map Builder. It uses the existing FiduMap camera models for
distortion-aware pixel unprojection and OpenGV for robust absolute-pose
estimation.

For a single active camera, the solver uses KNEIP RANSAC. For multiple active
cameras on one rigid body, it uses generalized GP3P RANSAC. It then runs OpenGV
UPnP on the RANSAC inliers and returns the valid candidate with the lowest mean
pixel reprojection error.

## Quick start

### 1. Install the package

When installing from a source checkout, initialise the Eigen and OpenGV
submodules before building:

```sh
git submodule update --init --recursive
python -m pip install .
```

Installing from source requires CMake and a working C++ compiler. Marker
detection also requires an OpenCV build containing `cv2.aruco`:

```sh
python -m pip install opencv-contrib-python
```

If a FiduMap GitHub Release provides a wheel, it can be installed without a
local compiler. Choose the wheel matching the operating system, CPU
architecture, and CPython version.

### 2. Export a marker map

In the Map Builder, complete marker detection, initialisation, and bundle
adjustment, then select **Export Optimized CSV**. The exported file contains
the world coordinates of all four corners of every optimized marker.

Keep the camera calibration XML used for the images. Runtime images must use
the same camera model, resolution, and image orientation as the calibration.

### 3. Detect markers and solve a single-camera pose

For a single camera, define the body frame to be the camera frame by setting
`T_B_C` to the identity transform. The returned `T_W_B` is then `T_W_C`, the
camera pose in the exported map frame.

```python
import cv2

from absolute_pose_solver import AbsolutePoseSolver, CameraConfig, FiducialDetector
from map_builder.camera_models import load_camera_model_xml
from map_builder.geometry import SE3

camera = load_camera_model_xml("camera_params.xml")
solver = AbsolutePoseSolver(
    "marker_map.csv",
    {"camera": CameraConfig(model=camera, T_B_C=SE3.identity())},
)

# This dictionary must match the one used to build the map.
detector = FiducialDetector("DICT_6X6_250")
image = cv2.imread("frame.jpg", cv2.IMREAD_COLOR)
if image is None:
    raise RuntimeError("Could not read frame.jpg")

detections = detector.detect(image)
result = solver.solve({"camera": detections})

if not result.success or result.T_W_B is None:
    print(
        "Pose solve failed:",
        result.num_correspondences,
        "correspondences and",
        result.num_inliers,
        "inliers",
    )
else:
    T_W_C = result.T_W_B
    print("T_W_C =\n", T_W_C.as_matrix())
    print("inliers:", result.num_inliers, "/", result.num_correspondences)
    print("mean reprojection error (px):", result.mean_reprojection_error_px)
```

Create the detector and solver once, outside the frame-processing loop. Only
`detect()` and `solve()` need to run for each new image.

## Input map format

The solver accepts the CSV written by **Export Optimized CSV**. Its required
fields are `id,x,y,z`, where each point ID encodes a marker and corner:

```text
marker_id = id >> 2
corner_index = id & 0b11
```

Each marker must have exactly four finite corners. The CSV corner order is
bottom-left, top-left, top-right, bottom-right. OpenCV detections use top-left,
top-right, bottom-right, bottom-left; the solver performs the required corner
remapping internally. Do not reorder detector corners in application code.

Detected marker IDs absent from the map are ignored.

## Coordinate frames

FiduMap names a transform `T_A_B` when it maps points from frame B into frame
A. Consequently:

- `T_W_B` is the returned body pose in the map/world frame.
- `T_B_C` is the fixed extrinsic transform from a camera frame into the rigid
  body frame.
- For a single camera with `T_B_C = SE3.identity()`, `T_W_B` is the camera pose
  `T_W_C`.
- To transform a body-frame point into the world frame, call
  `result.T_W_B.transform_points(points_B)`.
- The inverse pose is `T_B_W = result.T_W_B.inverse()`.

Camera bearing vectors use `+z` forward. Translation values have the same unit
as the exported map, normally metres.

## Multi-camera rigs

Use one `CameraConfig` per rigidly mounted camera. All `T_B_C` extrinsics must
refer to the same body frame, and the input detections should correspond to the
same pose in time.

```python
cameras = {
    "front": CameraConfig(front_model, T_B_C_front),
    "down": CameraConfig(down_model, T_B_C_down),
}
solver = AbsolutePoseSolver("marker_map.csv", cameras)

result = solver.solve(
    {
        "front": front_detector.detect(front_image),
        "down": down_detector.detect(down_image),
    }
)
```

The keys passed to `solve()` must match the camera IDs configured in the
constructor. A camera may be omitted for a frame when it has no image or no
detections. Generalized pose estimation is used when at least two configured
cameras contribute mapped observations.

## Solver configuration

```python
solver = AbsolutePoseSolver(
    "marker_map.csv",
    cameras,
    ransac_threshold_deg=1.0,
    ransac_max_iterations=1000,
    ransac_probability=0.999,
)
```

| Option | Meaning |
|---|---|
| `ransac_threshold_deg` | Maximum angular bearing error for a RANSAC inlier. Increase cautiously for noisier detections. |
| `ransac_max_iterations` | Maximum robust-estimation iterations. More iterations can help with many outliers but increase latency. |
| `ransac_probability` | Requested probability that RANSAC samples an outlier-free minimal set. |

The threshold is angular rather than pixel-based because detections are
unprojected through the configured camera model before OpenGV is called.

## Result diagnostics

`solve()` returns an `AbsolutePoseResult` for both success and expected solve
failures.

| Field | Meaning |
|---|---|
| `success` | Whether a valid pose was found. |
| `T_W_B` | Solved `SE3` pose, or `None` on failure. |
| `num_correspondences` | Number of usable corner correspondences; a mapped marker normally contributes four. |
| `num_inliers` | Number of corner correspondences accepted by RANSAC. |
| `inlier_indices` | Indices into the flattened corner-correspondence arrays, not marker IDs. |
| `mean_reprojection_error_px` | Mean camera-model pixel error over the RANSAC inliers, or `None` on failure. |

Use the inlier ratio and reprojection error as runtime quality signals. The
appropriate acceptance limits depend on camera resolution, calibration
quality, marker size, viewing distance, and application accuracy requirements.

## Troubleshooting

- **The native extension cannot be imported:** install a matching release
  wheel or build from a checkout after initialising the submodules.
- **`cv2.aruco` is missing:** replace `opencv-python` with
  `opencv-contrib-python` in the active environment.
- **There are zero correspondences despite detections:** check that the marker
  dictionary and IDs match the exported map.
- **The solve fails with few observations:** keep several well-spread markers
  visible. A single planar marker supplies four corners but generally has
  weaker geometry than multiple markers across different positions or planes.
- **Reprojection error is high:** verify the calibration XML, image resolution,
  cropping/rotation, marker dictionary, and `T_B_C` extrinsics.
- **The pose appears inverted:** remember that the result maps body coordinates
  into the world frame. Use `result.T_W_B.inverse()` when a world-to-body
  transform is required.
- **A camera ID is rejected:** pass only IDs configured when the solver was
  constructed.

Invalid maps, malformed corner arrays, or invalid camera configurations raise
an exception. Normal cases such as insufficient mapped correspondences or no
valid pose return `success=False` with diagnostics instead.
