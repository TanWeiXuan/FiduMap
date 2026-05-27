from __future__ import annotations

import csv
from pathlib import Path

from .dense_store import DenseReconstructionStore


DEFAULT_DENSE_POINT_CLOUD_FILENAME = "dense_point_cloud.csv"


def export_dense_point_cloud_csv(store: DenseReconstructionStore, path: Path) -> int:
    rows = store.list_active_dense_points()
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z"])
        for row in rows:
            writer.writerow([float(row["x"]), float(row["y"]), float(row["z"])])
    return len(rows)
