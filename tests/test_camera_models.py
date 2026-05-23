import numpy as np

from map_builder.camera_models import OmniRadTanCameraModel, PinholeRadTanCameraModel


def _ang_err(a: np.ndarray, b: np.ndarray) -> float:
    d = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(np.arccos(d))


def test_pinhole_no_distortion_center_projection_unprojection() -> None:
    cam = PinholeRadTanCameraModel(640, 480, 100.0, 100.0, 320.0, 240.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    pixel = cam.project(np.array([0.0, 0.0, 1.0]))
    np.testing.assert_allclose(pixel, np.array([320.0, 240.0]), atol=1e-12)
    ray = cam.unproject(np.array([320.0, 240.0]))
    np.testing.assert_allclose(ray, np.array([0.0, 0.0, 1.0]), atol=1e-12)


def test_pinhole_roundtrip_zero_distortion() -> None:
    cam = PinholeRadTanCameraModel(640, 480, 200.0, 200.0, 320.0, 240.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    rays = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, -0.2, 1.2], [0.2, 0.1, 1.4]])
    rays = rays / np.linalg.norm(rays, axis=1, keepdims=True)
    for ray in rays:
        back = cam.unproject(cam.project(ray))
        assert _ang_err(ray, back) < 2e-8


def test_pinhole_distorted_projection_and_local_roundtrip() -> None:
    cam = PinholeRadTanCameraModel(640, 480, 250.0, 250.0, 320.0, 240.0, 0.01, -0.001, 0.0005, -0.0003, 0.0001)
    ray = np.array([0.2, -0.15, 1.0])
    ray /= np.linalg.norm(ray)
    pix = cam.project(ray)
    assert np.isfinite(pix).all()

    pixel = np.array([330.0, 245.0])
    pixel_rt = cam.project(cam.unproject(pixel))
    np.testing.assert_allclose(pixel_rt, pixel, atol=1e-7)


def test_omni_xi_zero_matches_pinhole() -> None:
    pin = PinholeRadTanCameraModel(640, 480, 200.0, 210.0, 320.0, 240.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    omni = OmniRadTanCameraModel(640, 480, 200.0, 210.0, 320.0, 240.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    rays = np.array([[0.0, 0.0, 1.0], [0.2, 0.1, 1.4], [-0.1, 0.05, 1.1]])
    rays = rays / np.linalg.norm(rays, axis=1, keepdims=True)
    for ray in rays:
        np.testing.assert_allclose(pin.project(ray), omni.project(ray), atol=1e-12)

    pixel = np.array([350.0, 220.0])
    np.testing.assert_allclose(pin.unproject(pixel), omni.unproject(pixel), atol=1e-10)


def test_omni_roundtrip() -> None:
    cam = OmniRadTanCameraModel(1280, 720, 500.0, 500.0, 640.0, 360.0, 1.0, 0.002, -0.0001, 0.0002, -0.0002, 0.0)
    rays = np.array([[0.0, 0.0, 1.0], [0.2, -0.1, 0.97], [-0.15, 0.22, 0.96]])
    rays = rays / np.linalg.norm(rays, axis=1, keepdims=True)
    for ray in rays:
        back = cam.unproject(cam.project(ray))
        assert _ang_err(ray, back) < 1e-7
