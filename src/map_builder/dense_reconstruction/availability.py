from __future__ import annotations
from dataclasses import dataclass
import importlib

@dataclass
class AvailabilityResult:
    available: bool
    missing_dependencies: list[str]
    details: str

def _try_import_any(paths:list[str])->bool:
    for p in paths:
        try:
            importlib.import_module(p)
            return True
        except Exception:
            pass
    return False

def check_dense_reconstruction_availability()->AvailabilityResult:
    missing=[]
    for dep in ['numpy','cv2','torch']:
        try: importlib.import_module(dep)
        except Exception: missing.append(dep)
    if not _try_import_any(['vendor.xfeat.xfeat','xfeat','map_builder.vendor.xfeat.xfeat']):
        missing.append('vendored XFeat')
    if not _try_import_any(['vendor.xfeat.lighterglue','lighterglue']):
        missing.append('vendored XFeat-LighterGlue')
    ok=not missing
    return AvailabilityResult(ok, missing, 'Dense reconstruction available.' if ok else 'Dense reconstruction unavailable: missing ' + ' / '.join(missing))
