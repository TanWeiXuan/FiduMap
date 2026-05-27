from __future__ import annotations
import importlib
from dataclasses import dataclass

from .vendor_xfeat import (
    ensure_vendor_parent_on_path,
    vendored_lighterglue_file_exists,
    vendored_xfeat_file_exists,
)

@dataclass
class AvailabilityResult:
    available: bool
    missing_dependencies: list[str]
    details: str

def _try_import_any(paths: list[str]) -> bool:
    for p in paths:
        try:
            importlib.import_module(p)
            return True
        except Exception:
            pass
    return False


def _try_import(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def check_dense_reconstruction_availability() -> AvailabilityResult:
    missing = []
    for dep in ["numpy", "cv2", "torch"]:
        if not _try_import(dep):
            missing.append(dep)
    if not vendored_xfeat_file_exists():
        missing.append("vendored XFeat")
    elif "torch" not in missing:
        ensure_vendor_parent_on_path()
        if not _try_import_any(["vendor.xfeat.xfeat", "xfeat", "map_builder.vendor.xfeat.xfeat"]):
            missing.append("vendored XFeat")
    if not _try_import("kornia"):
        missing.append("kornia")
    if not vendored_lighterglue_file_exists():
        missing.append("vendored XFeat-LighterGlue")
    elif "kornia" not in missing and "torch" not in missing:
        ensure_vendor_parent_on_path()
        if not _try_import_any(["vendor.xfeat.lighterglue", "lighterglue"]):
            missing.append("vendored XFeat-LighterGlue")
    ok = not missing
    details = "Dense reconstruction available." if ok else "Dense reconstruction unavailable: missing " + " / ".join(missing)
    return AvailabilityResult(ok, missing, details)


def check_dense_ba_availability() -> AvailabilityResult:
    if _try_import("pyceres"):
        return AvailabilityResult(True, [], "Dense point BA available.")
    return AvailabilityResult(False, ["pyceres"], "Dense point BA unavailable: pyceres not installed.")
