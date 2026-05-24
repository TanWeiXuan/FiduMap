"""Theme loading helpers for tkinter/ttk."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
import tkinter as tk
from tkinter import ttk


def load_azure_theme(root: tk.Tk, mode: str = "light") -> bool:
    """Source the vendored Azure ttk theme when it is available."""

    normalized_mode = "light" if mode == "light" else "dark"
    for azure_tcl in _candidate_azure_tcl_paths():
        if not azure_tcl.exists():
            continue
        try:
            root.tk.call("source", str(azure_tcl))
            root.tk.call("set_theme", normalized_mode)
            return True
        except tk.TclError:
            continue

    try:
        ttk.Style(root).theme_use("clam")
    except tk.TclError:
        pass
    return False


def _candidate_azure_tcl_paths() -> list[Path]:
    candidates: list[Path] = []
    try:
        package_root = resources.files("map_builder")
        candidate = package_root / "vendor" / "azure" / "azure.tcl"
        if isinstance(candidate, Path):
            candidates.append(candidate)
        else:
            with resources.as_file(candidate) as path:
                candidates.append(path)
    except (FileNotFoundError, ModuleNotFoundError, TypeError):
        pass

    repo_root = Path(__file__).resolve().parents[3]
    candidates.append(repo_root / "vendor" / "azure_ttk_theme" / "azure.tcl")
    candidates.append(Path(__file__).resolve().parents[1] / "vendor" / "azure" / "azure.tcl")
    return candidates
