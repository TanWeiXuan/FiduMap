from __future__ import annotations
from pathlib import Path

from map_builder.dense_reconstruction.dense_store import DenseReconstructionStore
from map_builder.dense_reconstruction.models import DenseStageSummary, PairSelectionConfig
from map_builder.dense_reconstruction.pair_selection import select_frame_pairs
from map_builder.dense_reconstruction.point_cloud_export import export_dense_point_cloud_csv
from map_builder.project import ProjectStore


class DensePipeline:
    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.store = DenseReconstructionStore.open(self.folder)

    def build_frame_pairs(self, config: PairSelectionConfig | None = None) -> DenseStageSummary:
        cfg = config or PairSelectionConfig()
        with ProjectStore.open(self.folder) as project:
            poses = project.get_optimized_camera_poses()
            if not poses:
                return DenseStageSummary(stage="pair_selection", details="Run marker-map BA before dense reconstruction.")
            detections_by_image = {img.id: project.get_detections_for_image(img.id) for img in project.list_images()}
        pairs = select_frame_pairs(poses, detections_by_image, cfg)
        for rec in pairs:
            self.store.upsert_frame_pair(rec)
        return DenseStageSummary(stage="pair_selection", total=len(poses), success=len(pairs), details=f"Selected {len(pairs)} candidate pairs")

    def export_dense_csv(self, path: Path) -> int:
        return export_dense_point_cloud_csv(self.store, path)
