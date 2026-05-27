from __future__ import annotations
import csv
from pathlib import Path
from .dense_store import DenseReconstructionStore

def export_dense_point_cloud_csv(store:DenseReconstructionStore,path:Path)->int:
    rows=store.list_active_dense_points()
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['x','y','z']);
        for r in rows: w.writerow([r['x'],r['y'],r['z']])
    return len(rows)
