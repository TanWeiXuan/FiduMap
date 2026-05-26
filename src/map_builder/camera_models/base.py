"""Base camera model interfaces and shared geometric helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class CameraModel(ABC):
    """Abstract camera model API.

    Conventions:
    - Bearing vectors are camera-frame rays [x, y, z] with +z forward.
    - Pixels are image coordinates [u, v].
    - Unprojection always returns unit-length rays.
    """

    model_name: str
    image_width: int
    image_height: int

    @abstractmethod
    def project(self, ray: np.ndarray) -> np.ndarray:
        """Project a single bearing ray to a pixel [u, v]."""

    @abstractmethod
    def project_many(self, rays: np.ndarray) -> np.ndarray:
        """Project an array of rays with shape (N, 3) to pixels (N, 2)."""

    @abstractmethod
    def unproject(self, pixel: np.ndarray) -> np.ndarray:
        """Unproject a single pixel [u, v] to a unit ray [x, y, z]."""

    @abstractmethod
    def unproject_many(self, pixels: np.ndarray) -> np.ndarray:
        """Unproject pixels with shape (N, 2) to rays (N, 3)."""

    def is_pixel_in_bounds(self, pixel: np.ndarray) -> bool:
        """Return True when pixel lies inside image bounds [0,w) x [0,h)."""
        p = np.asarray(pixel, dtype=float)
        return 0.0 <= p[0] < float(self.image_width) and 0.0 <= p[1] < float(self.image_height)

    @abstractmethod
    def validate(self) -> None:
        """Validate camera parameters; raise ValueError on invalid values."""

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize model fields to a plain dictionary."""

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> "CameraModel":
        """Deserialize from dictionary."""

    @abstractmethod
    def save_xml(self, path: str | Path) -> None:
        """Save the camera model to XML."""

    @classmethod
    @abstractmethod
    def load_xml(cls, path: str | Path) -> "CameraModel":
        """Load a camera model of this class from XML."""

    @classmethod
    @abstractmethod
    def write_example_xml(cls, path: str | Path) -> None:
        """Write example XML file for this camera model."""


def normalize_vectors(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize one vector of shape (3,) or array of vectors shape (N, 3)."""
    v = np.asarray(vectors, dtype=float)
    if not ((v.ndim == 1 and v.shape[0] == 3) or (v.ndim == 2 and v.shape[1] == 3)):
        raise ValueError(f"Expected shape (3,) or (N,3), got {v.shape}.")

    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    if np.any(norms <= eps):
        raise ValueError("Cannot normalize near-zero vector(s).")

    return v / norms


def distort_radtan(
    x: np.ndarray,
    y: np.ndarray,
    k1: float,
    k2: float,
    p1: float,
    p2: float,
    k3: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Brown-Conrady radial-tangential distortion for normalized image points."""
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    x_d = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    y_d = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return x_d, y_d


def undistort_radtan_iterative(
    xd: np.ndarray,
    yd: np.ndarray,
    k1: float,
    k2: float,
    p1: float,
    p2: float,
    k3: float,
    max_iterations: int = 10,
    tolerance: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Iteratively invert radial-tangential distortion."""
    x = np.asarray(xd, dtype=float).copy()
    y = np.asarray(yd, dtype=float).copy()
    xd = np.asarray(xd, dtype=float)
    yd = np.asarray(yd, dtype=float)

    for _ in range(max_iterations):
        x_proj, y_proj = distort_radtan(x, y, k1, k2, p1, p2, k3)
        dx = xd - x_proj
        dy = yd - y_proj
        x += dx
        y += dy
        if float(np.max(np.abs(dx)) + np.max(np.abs(dy))) < tolerance:
            break

    return x, y
