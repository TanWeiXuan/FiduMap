from __future__ import annotations

from pathlib import Path
import sys


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def vendored_xfeat_dir() -> Path:
    return repository_root() / "vendor" / "xfeat"


def ensure_vendor_parent_on_path() -> None:
    root = str(repository_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def vendored_xfeat_file_exists() -> bool:
    base = vendored_xfeat_dir()
    return (base / "xfeat.py").is_file() and (base / "model.py").is_file()


def vendored_lighterglue_file_exists() -> bool:
    base = vendored_xfeat_dir()
    return (base / "lighterglue.py").is_file() and (base / "weights" / "xfeat-lighterglue.pt").is_file()
