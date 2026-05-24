import numpy as np

from map_builder.geometry import SE3


def test_se3_identity_transform() -> None:
    points = np.array([[1.0, 2.0, 3.0]])
    np.testing.assert_allclose(SE3.identity().transform_points(points), points)


def test_se3_inverse_composition_gives_identity() -> None:
    theta = 0.3
    R = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    T = SE3(R, np.array([1.0, -2.0, 0.5]))
    identity = T @ T.inverse()
    np.testing.assert_allclose(identity.R, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(identity.t, np.zeros(3), atol=1e-12)


def test_se3_json_roundtrip_and_transform_points() -> None:
    T = SE3(np.eye(3), np.array([1.0, 2.0, 3.0]))
    restored = SE3.from_json_dict(T.to_json_dict())
    np.testing.assert_allclose(restored.transform_points([[1.0, 1.0, 1.0]]), [[2.0, 3.0, 4.0]])
