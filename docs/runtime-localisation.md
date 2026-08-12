# Runtime localisation

`absolute_pose_solver` localises a camera or rigid multi-camera body directly against a marker-corner CSV exported by the FiduMap Map Builder. Pixels are unprojected by each existing FiduMap `CameraModel`; OpenGV then runs KNEIP RANSAC for one active camera or generalized GP3P RANSAC for multiple active cameras. OpenGV UPnP is run only on the RANSAC inliers, and the candidate with the lowest camera-model pixel reprojection error is returned.

## Map and corner order

The input CSV has fields `id,x,y,z`. A point ID encodes `marker_id = id >> 2` and `corner_index = id & 0b11`. Its corner order is bottom-left, top-left, top-right, bottom-right. OpenCV detections use top-left, top-right, bottom-right, bottom-left, so runtime correspondences deliberately remap detector indices with `[1, 2, 3, 0]`. Unknown detected marker IDs are ignored.

## Frames and cameras

FiduMap uses `T_A_B` for a transform mapping frame B into frame A. Each `CameraConfig` therefore contains a camera model and `T_B_C`, which maps a camera ray and optical centre into the body frame. The result is `T_W_B`: the pose that maps the body/rig frame into the map/world frame.

Camera models can also be constructed with the existing `load_camera_model_xml()` function. No OpenCV intrinsic matrix is passed to the solver.

```python
from absolute_pose_solver import AbsolutePoseSolver, CameraConfig, FiducialDetector
from map_builder.camera_models import load_camera_model_xml

front_model = load_camera_model_xml("front_camera.xml")
down_model = load_camera_model_xml("down_camera.xml")

cameras = {
    "front": CameraConfig(model=front_model, T_B_C=T_B_C_front),
    "down": CameraConfig(model=down_model, T_B_C=T_B_C_down),
}
solver = AbsolutePoseSolver("marker_map.csv", cameras)
detector = FiducialDetector("DICT_6X6_250")

result = solver.solve({
    "front": detector.detect(front_image),
    "down": detector.detect(down_image),
})
if result.success:
    T_W_B = result.T_W_B
```

For one camera, configure one entry and pass `{"front": detections}` to the same API.

## Building

Clone submodules and install the project so the small pybind11 module is compiled against vendored OpenGV and Eigen:

```text
git submodule update --init --recursive
python -m pip install .
```
