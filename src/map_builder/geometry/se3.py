"""Small SE(3) helper with explicit frame convention.

``T_A_B`` maps points from frame B into frame A.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SE3:
    R: np.ndarray
    t: np.ndarray

    def __post_init__(self) -> None:
        R = np.asarray(self.R, dtype=float)
        t = np.asarray(self.t, dtype=float).reshape(3)
        if R.shape != (3, 3):
            raise ValueError(f"Expected R shape (3,3), got {R.shape}.")
        object.__setattr__(self, "R", R)
        object.__setattr__(self, "t", t)

    @classmethod
    def identity(cls) -> "SE3":
        return cls(np.eye(3, dtype=float), np.zeros(3, dtype=float))

    @classmethod
    def from_rvec_tvec(cls, rvec: Any, tvec: Any) -> "SE3":
        try:
            import cv2  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("OpenCV is required to convert rvec/tvec to SE3.") from exc
        R, _jacobian = cv2.Rodrigues(np.asarray(rvec, dtype=float).reshape(3, 1))
        return cls(R, np.asarray(tvec, dtype=float).reshape(3))

    @classmethod
    def exp(cls, xi: Any) -> "SE3":
        """Create a pose from a simple translation + rotation-vector parameter.

        This is a pragmatic optimizer parameterization, not a coupled Lie
        exponential: ``xi[:3]`` is translation rho and ``xi[3:]`` is an OpenCV
        Rodrigues rotation vector phi.
        """

        xi_arr = np.asarray(xi, dtype=float).reshape(6)
        return cls.from_rvec_tvec(xi_arr[3:], xi_arr[:3])

    def log(self) -> np.ndarray:
        """Return ``[tx, ty, tz, rvec_x, rvec_y, rvec_z]``."""

        try:
            import cv2  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("OpenCV is required to convert SE3 to rvec/tvec.") from exc
        rvec, _jacobian = cv2.Rodrigues(self.R)
        return np.concatenate([self.t, rvec.reshape(3)])

    def as_matrix(self) -> np.ndarray:
        matrix = np.eye(4, dtype=float)
        matrix[:3, :3] = self.R
        matrix[:3, 3] = self.t
        return matrix

    def inverse(self) -> "SE3":
        R_inv = self.R.T
        return SE3(R_inv, -R_inv @ self.t)

    def compose(self, other: "SE3") -> "SE3":
        return SE3(self.R @ other.R, self.R @ other.t + self.t)

    def __matmul__(self, other: "SE3") -> "SE3":
        return self.compose(other)

    def transform_points(self, points: Any) -> np.ndarray:
        p = np.asarray(points, dtype=float)
        if p.ndim == 1:
            if p.shape != (3,):
                raise ValueError(f"Expected point shape (3,), got {p.shape}.")
            return self.R @ p + self.t
        if p.ndim == 2 and p.shape[1] == 3:
            return (self.R @ p.T).T + self.t
        raise ValueError(f"Expected points shape (3,) or (N,3), got {p.shape}.")

    def rotation_quaternion_wxyz(self) -> tuple[float, float, float, float]:
        R = self.R
        trace = float(np.trace(R))
        if trace > 0.0:
            s = np.sqrt(trace + 1.0) * 2.0
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        else:
            idx = int(np.argmax(np.diag(R)))
            if idx == 0:
                s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif idx == 1:
                s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s
        q = np.array([w, x, y, z], dtype=float)
        q /= np.linalg.norm(q)
        return tuple(float(v) for v in q)

    def to_json_dict(self) -> dict[str, Any]:
        return {"R": self.R.tolist(), "t": self.t.tolist()}

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "SE3":
        return cls(np.asarray(data["R"], dtype=float), np.asarray(data["t"], dtype=float))
