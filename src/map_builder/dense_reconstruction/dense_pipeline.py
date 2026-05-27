from __future__ import annotations
from pathlib import Path
from .dense_store import DenseReconstructionStore
from .point_cloud_export import export_dense_point_cloud_csv

class DensePipeline:
    def __init__(self, folder:Path): self.store=DenseReconstructionStore.open(folder)
    def export_dense_csv(self,path:Path)->int: return export_dense_point_cloud_csv(self.store,path)
