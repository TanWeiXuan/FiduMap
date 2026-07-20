import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="session")
def _tk_session_root():
    import tkinter as tk

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"tk display unavailable: {exc}")
    root.withdraw()
    try:
        yield root
    finally:
        root.destroy()


@pytest.fixture
def tk_root(_tk_session_root):
    import tkinter as tk

    window = tk.Toplevel(_tk_session_root)
    window.withdraw()
    try:
        yield window
    finally:
        window.destroy()
