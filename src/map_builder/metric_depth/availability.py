from __future__ import annotations

from dataclasses import dataclass
import importlib.util


@dataclass(frozen=True)
class DependencyAvailability:
    available: bool
    missing: tuple[str, ...]
    details: str


def check_inference_availability() -> DependencyAvailability:
    required = ("torch", "transformers", "safetensors", "PIL")
    missing = tuple(name for name in required if importlib.util.find_spec(name) is None)
    install = "pip install -r src/map_builder/requirements-depth.txt"
    details = "Metric-depth inference dependencies are available."
    if missing:
        details = f"Missing optional dependencies: {', '.join(missing)}. Install with: {install}"
    return DependencyAvailability(not missing, missing, details)


def check_vtk_availability() -> DependencyAvailability:
    missing = () if importlib.util.find_spec("vtkmodules") is not None else ("vtk",)
    details = (
        "VTK depth viewer is available."
        if not missing
        else "Missing optional dependency: vtk. Install with: pip install -r src/map_builder/requirements-depth.txt"
    )
    return DependencyAvailability(not missing, missing, details)

