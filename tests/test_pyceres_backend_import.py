import ast
from pathlib import Path

import pytest


def test_pyceres_backend_imports_when_pyceres_available() -> None:
    pytest.importorskip("pyceres")
    import map_builder.optimization.pyceres_backend as backend

    assert backend.PyCeresBASolver is not None


def test_pyceres_backend_does_not_import_scipy() -> None:
    source_path = Path(__file__).resolve().parents[1] / "src" / "map_builder" / "optimization" / "pyceres_backend.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert "scipy" not in imported_roots
