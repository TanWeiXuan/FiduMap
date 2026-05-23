from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .base import CameraModel, distort_radtan, normalize_vectors, undistort_radtan_iterative


@dataclass
class PinholeRadTanCameraModel(CameraModel):
    """Pinhole camera with Brown-Conrady radial-tangential distortion."""

    image_width: int
    image_height: int
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float
    k2: float
    p1: float
    p2: float
    k3: float

    model_name: str = "pinhole_radtan"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("image_width and image_height must be positive.")
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError("fx and fy must be positive.")

    def project(self, ray: np.ndarray) -> np.ndarray:
        rays = np.asarray(ray, dtype=float)
        if rays.shape != (3,):
            raise ValueError(f"Expected ray shape (3,), got {rays.shape}.")
        r = normalize_vectors(rays)
        if r[2] <= 1e-12:
            return np.array([np.nan, np.nan], dtype=float)
        x = r[0] / r[2]
        y = r[1] / r[2]
        xd, yd = distort_radtan(x, y, self.k1, self.k2, self.p1, self.p2, self.k3)
        return np.array([self.fx * xd + self.cx, self.fy * yd + self.cy], dtype=float)

    def project_many(self, rays: np.ndarray) -> np.ndarray:
        r = normalize_vectors(np.asarray(rays, dtype=float))
        if r.ndim != 2 or r.shape[1] != 3:
            raise ValueError(f"Expected rays shape (N,3), got {r.shape}.")
        out = np.full((r.shape[0], 2), np.nan, dtype=float)
        mask = r[:, 2] > 1e-12
        if not np.any(mask):
            return out
        x = r[mask, 0] / r[mask, 2]
        y = r[mask, 1] / r[mask, 2]
        xd, yd = distort_radtan(x, y, self.k1, self.k2, self.p1, self.p2, self.k3)
        out[mask, 0] = self.fx * xd + self.cx
        out[mask, 1] = self.fy * yd + self.cy
        return out

    def unproject(self, pixel: np.ndarray) -> np.ndarray:
        p = np.asarray(pixel, dtype=float)
        if p.shape != (2,):
            raise ValueError(f"Expected pixel shape (2,), got {p.shape}.")
        xd = (p[0] - self.cx) / self.fx
        yd = (p[1] - self.cy) / self.fy
        x, y = undistort_radtan_iterative(xd, yd, self.k1, self.k2, self.p1, self.p2, self.k3)
        return normalize_vectors(np.array([x, y, 1.0], dtype=float))

    def unproject_many(self, pixels: np.ndarray) -> np.ndarray:
        p = np.asarray(pixels, dtype=float)
        if p.ndim != 2 or p.shape[1] != 2:
            raise ValueError(f"Expected pixels shape (N,2), got {p.shape}.")
        xd = (p[:, 0] - self.cx) / self.fx
        yd = (p[:, 1] - self.cy) / self.fy
        x, y = undistort_radtan_iterative(xd, yd, self.k1, self.k2, self.p1, self.p2, self.k3)
        rays = np.column_stack((x, y, np.ones_like(x)))
        return normalize_vectors(rays)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PinholeRadTanCameraModel":
        return cls(**data)

    def save_xml(self, path: str | Path) -> None:
        from .xml_io import save_camera_model_xml

        save_camera_model_xml(self, path)

    @classmethod
    def load_xml(cls, path: str | Path) -> "PinholeRadTanCameraModel":
        from .xml_io import load_camera_model_xml

        cam = load_camera_model_xml(path)
        if not isinstance(cam, cls):
            raise ValueError(f"XML at {path} is not a {cls.__name__} model.")
        return cam

    @classmethod
    def write_example_xml(cls, path: str | Path) -> None:
        cls(
            image_width=1920,
            image_height=1080,
            fx=1000.0,
            fy=1000.0,
            cx=960.0,
            cy=540.0,
            k1=0.0,
            k2=0.0,
            p1=0.0,
            p2=0.0,
            k3=0.0,
        ).save_xml(path)
