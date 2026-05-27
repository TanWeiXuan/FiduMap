from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from map_builder.camera_models import load_camera_model_xml
from map_builder.dense_reconstruction.availability import (
    check_xfeat_extraction_availability,
    check_xfeat_matching_availability,
)
from map_builder.dense_reconstruction.dense_store import DenseReconstructionStore
from map_builder.dense_reconstruction.duplicate_merge import merge_duplicate_points, points_from_rows
from map_builder.dense_reconstruction.epipolar_filter import filter_pair_matches
from map_builder.dense_reconstruction.models import (
    DenseBAConfig,
    DenseStageSummary,
    DuplicateMergeConfig,
    EpipolarFilterConfig,
    MatchingConfig,
    PairSelectionConfig,
    TriangulationConfig,
    XFeatExtractionConfig,
)
from map_builder.dense_reconstruction.pair_selection import select_frame_pairs
from map_builder.dense_reconstruction.point_ba import run_dense_point_ba
from map_builder.dense_reconstruction.point_cloud_export import export_dense_point_cloud_csv
from map_builder.dense_reconstruction.track_builder import build_tracks_from_matches
from map_builder.dense_reconstruction.xfeat_extractor import XFeatSemiDenseExtractor
from map_builder.dense_reconstruction.xfeat_matching import XFeatLighterGlueMatcher
from map_builder.project import ProjectStore

ProgressCallback = Callable[[str], None]


class DensePipeline:
    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.store = DenseReconstructionStore.open(self.folder)

    def close(self) -> None:
        self.store.close()

    def extract_features(
        self,
        config: XFeatExtractionConfig | None = None,
        progress: ProgressCallback | None = None,
    ) -> DenseStageSummary:
        cfg = config or XFeatExtractionConfig()
        availability = check_xfeat_extraction_availability()
        if not availability.available:
            return DenseStageSummary(stage="feature_extraction", details=availability.details)
        import cv2  # type: ignore[import-not-found]

        with ProjectStore.open(self.folder) as project:
            images = [img for img in project.list_images(include_missing=False) if not img.ignored]
        extractor = XFeatSemiDenseExtractor(cfg)
        successes = 0
        failures = 0
        for index, image in enumerate(images, start=1):
            _emit(progress, f"Extracting features: {index}/{len(images)} images")
            existing = self.store.get_feature(image.id)
            if existing is not None and existing.status == "success" and not cfg.force_recompute:
                successes += 1
                continue
            try:
                arr = cv2.imread(str(image.absolute_path(self.folder)), cv2.IMREAD_COLOR)
                if arr is None:
                    raise RuntimeError(f"Could not load image: {image.rel_path}")
                record = extractor.extract(arr)
                record.image_id = image.id
                record.rel_path = image.rel_path
                record.width = image.width or record.width
                record.height = image.height or record.height
                self.store.upsert_feature_record(record)
                successes += 1
            except Exception:
                self.store.upsert_feature(image.id, image.rel_path, status="failed", width=image.width, height=image.height)
                failures += 1
        return DenseStageSummary(
            stage="feature_extraction",
            total=len(images),
            success=successes,
            failed=failures,
            details=f"Extracted features for {successes}/{len(images)} image(s)",
        )

    def build_frame_pairs(
        self,
        config: PairSelectionConfig | None = None,
        progress: ProgressCallback | None = None,
    ) -> DenseStageSummary:
        cfg = config or PairSelectionConfig()
        with ProjectStore.open(self.folder) as project:
            poses = project.get_optimized_camera_poses()
            if not poses:
                return DenseStageSummary(stage="pair_selection", details="Run marker-map BA before dense reconstruction.")
            detections_by_image = {img.id: project.get_detections_for_image(img.id) for img in project.list_images()}
        _emit(progress, f"Selecting frame pairs: 0/{len(poses)} cameras")
        pairs = select_frame_pairs(poses, detections_by_image, cfg)
        for rec in pairs:
            self.store.upsert_frame_pair(rec)
        _emit(progress, f"Selected {len(pairs)} candidate pairs")
        return DenseStageSummary(
            stage="pair_selection",
            total=len(poses),
            success=len(pairs),
            details=f"Selected {len(pairs)} candidate pairs",
        )

    def match_frame_pairs(
        self,
        config: MatchingConfig | None = None,
        progress: ProgressCallback | None = None,
    ) -> DenseStageSummary:
        cfg = config or MatchingConfig()
        availability = check_xfeat_matching_availability()
        if not availability.available:
            return DenseStageSummary(stage="pair_matching", details=availability.details)
        pairs = self.store.list_frame_pairs()
        pairs = [p for p in pairs if cfg.force_recompute or p.num_raw_matches == 0]
        if cfg.max_pairs_to_match is not None:
            pairs = pairs[: int(cfg.max_pairs_to_match)]
        matcher = XFeatLighterGlueMatcher(cfg) if pairs else None
        matched = 0
        failed = 0
        raw_matches = 0
        for index, pair in enumerate(pairs, start=1):
            _emit(progress, f"Matching pairs: {index}/{len(pairs)} pairs")
            assert pair.id is not None
            fa = self.store.get_feature(pair.image_id_a)
            fb = self.store.get_feature(pair.image_id_b)
            if fa is None or fb is None or fa.status != "success" or fb.status != "success":
                failed += 1
                continue
            if fa.descriptors is None or fb.descriptors is None:
                failed += 1
                continue
            try:
                assert matcher is not None
                matches = matcher.match(fa, fb, pair.id)
                self.store.replace_pair_matches(pair.id, matches)
                raw_matches += len(matches)
                matched += 1
            except Exception:
                failed += 1
        return DenseStageSummary(
            stage="pair_matching",
            total=len(pairs),
            success=matched,
            failed=failed,
            details=f"Matched {matched} pair(s), {raw_matches} raw match(es)",
        )

    def filter_matches(
        self,
        config: EpipolarFilterConfig | None = None,
        progress: ProgressCallback | None = None,
    ) -> DenseStageSummary:
        cfg = config or EpipolarFilterConfig()
        context = self._load_ba_context()
        if isinstance(context, str):
            return DenseStageSummary(stage="epipolar_filter", details=context)
        poses_by_image, camera_model = context
        pairs = [p for p in self.store.list_frame_pairs() if p.num_raw_matches > 0]
        filtered = 0
        inliers_total = 0
        for index, pair in enumerate(pairs, start=1):
            _emit(progress, f"Filtering matches: {index}/{len(pairs)} pairs")
            if pair.id is None or pair.image_id_a not in poses_by_image or pair.image_id_b not in poses_by_image:
                continue
            matches = self.store.list_pair_matches(pair.id)
            ids, errors, inliers = filter_pair_matches(
                matches,
                poses_by_image[pair.image_id_a],
                poses_by_image[pair.image_id_b],
                camera_model,
                cfg,
            )
            self.store.update_pair_epipolar_results(pair.id, ids, errors, inliers, cfg.min_inliers)
            filtered += 1
            inliers_total += int(inliers.sum())
        return DenseStageSummary(
            stage="epipolar_filter",
            total=len(pairs),
            success=filtered,
            details=f"Filtered {filtered} pair(s), {inliers_total} inlier match(es)",
        )

    def build_tracks_and_triangulate(
        self,
        config: TriangulationConfig | None = None,
        progress: ProgressCallback | None = None,
    ) -> DenseStageSummary:
        cfg = config or TriangulationConfig()
        context = self._load_ba_context()
        if isinstance(context, str):
            return DenseStageSummary(stage="track_triangulation", details=context)
        poses_by_image, camera_model = context
        pairs = self.store.list_frame_pairs()
        matches_by_pair = {
            int(pair.id): self.store.list_pair_matches(pair.id, epipolar_inliers_only=True)
            for pair in pairs
            if pair.id is not None
        }
        _emit(progress, f"Building tracks: 0/{sum(len(v) for v in matches_by_pair.values())} inlier matches")
        tracks = build_tracks_from_matches(pairs, matches_by_pair, poses_by_image, camera_model, cfg)
        _emit(progress, f"Triangulating tracks: {len(tracks)}/{len(tracks)} tracks")
        self.store.replace_tracks_and_points(tracks)
        return DenseStageSummary(
            stage="track_triangulation",
            total=sum(len(v) for v in matches_by_pair.values()),
            success=len(tracks),
            details=f"Built {len(tracks)} active dense point track(s)",
        )

    def merge_duplicates(
        self,
        config: DuplicateMergeConfig | None = None,
        progress: ProgressCallback | None = None,
    ) -> DenseStageSummary:
        cfg = config or DuplicateMergeConfig()
        points = points_from_rows(self.store.list_active_dense_points())
        observations = self.store.list_track_observations()
        obs_by_track: dict[int, list[Any]] = {}
        for obs in observations:
            obs_by_track.setdefault(obs.track_id, []).append(obs)
        _emit(progress, f"Merging duplicates: 0/{len(points)} candidate points")
        merged, accepted = merge_duplicate_points(points, obs_by_track, cfg)
        if accepted:
            self.store.replace_active_dense_points(merged, source="merged")
        return DenseStageSummary(
            stage="duplicate_merge",
            total=len(points),
            success=accepted,
            details=f"Merged {accepted} duplicate candidate(s); active points: {len(merged)}",
        )

    def run_dense_ba(
        self,
        config: DenseBAConfig | None = None,
        progress: ProgressCallback | None = None,
    ) -> DenseStageSummary:
        cfg = config or DenseBAConfig()
        context = self._load_ba_context()
        if isinstance(context, str):
            return DenseStageSummary(stage="dense_ba", details=context)
        _emit(progress, "Running dense BA...")
        poses_by_image, camera_model = context
        return run_dense_point_ba(self.store, poses_by_image, camera_model, cfg)

    def export_dense_csv(self, path: Path) -> int:
        return export_dense_point_cloud_csv(self.store, path)

    def _load_ba_context(self) -> tuple[dict[int, dict[str, Any]], Any] | str:
        with ProjectStore.open(self.folder) as project:
            poses = project.get_optimized_camera_poses()
            if not poses:
                return "Run marker-map BA before dense reconstruction."
            camera_path = project.get_camera_config_path()
            if camera_path is None:
                return "Choose a camera config XML before dense reconstruction."
            camera_model = load_camera_model_xml(camera_path)
        return {pose.image_id: pose.T_W_C for pose in poses}, camera_model


def _emit(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)
