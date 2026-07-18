from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import statistics
from typing import Any

from map_builder.camera_models import load_camera_model_xml
from map_builder.dense_reconstruction.availability import (
    check_dense_ba_availability,
    check_xfeat_extraction_availability,
    check_xfeat_matching_availability,
)
from map_builder.dense_reconstruction.dense_store import DenseReconstructionStore
from map_builder.dense_reconstruction.duplicate_merge import merge_duplicate_points, points_from_rows
from map_builder.dense_reconstruction.epipolar_filter import filter_pair_matches
from map_builder.dense_reconstruction.models import (
    DenseBAConfig,
    DensePipelineRunSummary,
    DenseReconstructionConfig,
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
from map_builder.dense_reconstruction.xfeat_matching import IncompatibleDenseFeatureError, XFeatSemiDenseMatcher
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
        with ProjectStore.open(self.folder) as project:
            images = [img for img in project.list_images(include_missing=False) if not img.ignored]
        if not images:
            return DenseStageSummary(stage="feature_extraction", details="No active images are available for feature extraction.")

        existing_by_image = {image.id: self.store.get_feature(image.id) for image in images}
        recompute_image_ids = {
            image.id
            for image in images
            if cfg.force_recompute
            or existing_by_image[image.id] is None
            or existing_by_image[image.id].status != "success"
            or not _is_current_semidense_feature(existing_by_image[image.id])
        }
        if recompute_image_ids:
            availability = check_xfeat_extraction_availability()
            if not availability.available:
                return DenseStageSummary(stage="feature_extraction", details=availability.details)
            import cv2  # type: ignore[import-not-found]

            extractor = XFeatSemiDenseExtractor(cfg)
            self.store.replace_frame_pairs([])
        else:
            extractor = None

        successes = 0
        failures = 0
        total_keypoints = 0
        failure_reasons: list[str] = []
        for index, image in enumerate(images, start=1):
            _emit(progress, f"Extracting features: {index}/{len(images)} images")
            existing = existing_by_image[image.id]
            if image.id not in recompute_image_ids:
                assert existing is not None
                successes += 1
                total_keypoints += int(existing.num_keypoints)
                continue
            try:
                assert extractor is not None
                arr = cv2.imread(str(image.absolute_path(self.folder)), cv2.IMREAD_COLOR)
                if arr is None:
                    raise RuntimeError(f"Could not load image: {image.rel_path}")
                record = extractor.extract(arr, image.id)
                record.rel_path = image.rel_path
                record.width = image.width or record.width
                record.height = image.height or record.height
                self.store.upsert_feature_record(record)
                successes += 1
                total_keypoints += int(record.num_keypoints)
            except Exception as exc:
                self.store.upsert_feature(image.id, image.rel_path, status="failed", width=image.width, height=image.height)
                failures += 1
                failure_reasons.append(f"{image.rel_path}: {str(exc) or exc.__class__.__name__}")
        details = (
            f"Dense features available for {successes}/{len(images)} image(s); "
            f"{total_keypoints:,} keypoints total"
        )
        if failures:
            details += f"; failed {failures} image(s)"
        if failure_reasons:
            details += "; reasons: " + " | ".join(failure_reasons[:3])
        return DenseStageSummary(
            stage="feature_extraction",
            total=len(images),
            success=successes,
            failed=failures,
            details=details,
        )

    def build_frame_pairs(
        self,
        config: PairSelectionConfig | None = None,
        progress: ProgressCallback | None = None,
    ) -> DenseStageSummary:
        cfg = config or PairSelectionConfig()
        with ProjectStore.open(self.folder) as project:
            all_poses = project.get_optimized_camera_poses()
            if not all_poses:
                self.store.replace_frame_pairs([])
                return DenseStageSummary(stage="pair_selection", details="Run marker-map BA before dense reconstruction.")
            detections_by_image = {img.id: project.get_detections_for_image(img.id) for img in project.list_images()}
        feature_image_ids = {feature.image_id for feature in self.store.list_features(status="success")}
        poses = [pose for pose in all_poses if pose.image_id in feature_image_ids]
        if len(poses) < 2:
            self.store.replace_frame_pairs([])
            return DenseStageSummary(
                stage="pair_selection",
                total=len(all_poses),
                details=(
                    "At least two optimized cameras with successful dense features are required; "
                    f"found {len(poses)}/{len(all_poses)}."
                ),
            )
        _emit(progress, f"Selecting frame pairs: 0/{len(poses)} feature-backed cameras")
        pairs = select_frame_pairs(poses, detections_by_image, cfg)
        self.store.replace_frame_pairs(pairs)
        _emit(progress, f"Selected {len(pairs)} candidate pairs")
        details = (
            f"Selected {len(pairs)} candidate pair(s) from {len(poses)}/{len(all_poses)} "
            "optimized cameras with valid dense features"
        )
        if pairs:
            baselines = [float(pair.baseline_m or 0.0) for pair in pairs]
            common_pairs = sum(1 for pair in pairs if pair.common_marker_count > 0)
            details += (
                f"; baseline min/median/max = {min(baselines):.3f}/"
                f"{statistics.median(baselines):.3f}/{max(baselines):.3f} m"
                f"; {common_pairs}/{len(pairs)} share markers"
            )
        return DenseStageSummary(
            stage="pair_selection",
            total=len(poses),
            success=len(pairs),
            details=details,
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
        candidate_pairs = self.store.list_frame_pairs()
        pairs = [p for p in candidate_pairs if cfg.force_recompute or p.num_raw_matches == 0]
        if cfg.max_pairs_to_match is not None:
            pairs = pairs[: int(cfg.max_pairs_to_match)]
        matcher = XFeatSemiDenseMatcher(cfg) if pairs else None
        if pairs:
            self.store.clear_tracks_and_points()
        matched = 0
        no_matches = 0
        skipped = 0
        failed = 0
        raw_matches = 0
        per_pair_counts: list[int] = []
        failure_reasons: list[str] = []
        for index, pair in enumerate(pairs, start=1):
            _emit(progress, f"Matching pairs: {index}/{len(pairs)} pairs")
            assert pair.id is not None
            fa = self.store.get_feature(pair.image_id_a)
            fb = self.store.get_feature(pair.image_id_b)
            if fa is None or fb is None or fa.status != "success" or fb.status != "success":
                skipped += 1
                reason = "Missing successful semi-dense feature records. Re-run feature extraction."
                failure_reasons.append(reason)
                self.store.update_frame_pair_matching_status(pair.id, "missing_features", clear_matches=cfg.force_recompute)
                continue
            try:
                assert matcher is not None
                matches = matcher.match(fa, fb, pair.id)
                self.store.replace_pair_matches(pair.id, matches)
                match_count = len(matches)
                raw_matches += match_count
                per_pair_counts.append(match_count)
                if match_count:
                    matched += 1
                else:
                    no_matches += 1
                    self.store.update_frame_pair_matching_status(pair.id, "no_matches")
            except IncompatibleDenseFeatureError as exc:
                skipped += 1
                failure_reasons.append(str(exc))
                self.store.update_frame_pair_matching_status(pair.id, "incompatible_features", clear_matches=cfg.force_recompute)
            except Exception as exc:
                failed += 1
                failure_reasons.append(str(exc) or exc.__class__.__name__)
                self.store.update_frame_pair_matching_status(pair.id, "match_failed", clear_matches=cfg.force_recompute)
        details = _matching_details(
            total_candidate_pairs=len(candidate_pairs),
            processed_pairs=len(pairs),
            matched_pairs=matched,
            no_match_pairs=no_matches,
            skipped_pairs=skipped,
            failed_pairs=failed,
            raw_matches=raw_matches,
            per_pair_counts=per_pair_counts,
            failure_reasons=failure_reasons,
        )
        return DenseStageSummary(
            stage="pair_matching",
            total=len(candidate_pairs),
            success=matched,
            failed=failed + skipped,
            details=details,
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
        processable_pairs = [
            pair
            for pair in pairs
            if pair.id is not None
            and pair.image_id_a in poses_by_image
            and pair.image_id_b in poses_by_image
        ]
        if processable_pairs:
            self.store.clear_tracks_and_points()
        filtered = 0
        usable_pairs = 0
        inliers_total = 0
        for index, pair in enumerate(processable_pairs, start=1):
            _emit(progress, f"Filtering matches: {index}/{len(processable_pairs)} pairs")
            assert pair.id is not None
            matches = self.store.list_pair_matches(pair.id)
            ids, errors, inliers = filter_pair_matches(
                matches,
                poses_by_image[pair.image_id_a],
                poses_by_image[pair.image_id_b],
                camera_model,
                cfg,
            )
            inlier_count = self.store.update_pair_epipolar_results(pair.id, ids, errors, inliers, cfg.min_inliers)
            filtered += 1
            inliers_total += inlier_count
            if inlier_count >= cfg.min_inliers:
                usable_pairs += 1
        return DenseStageSummary(
            stage="epipolar_filter",
            total=len(pairs),
            success=usable_pairs,
            failed=max(filtered - usable_pairs, 0),
            details=(
                f"Epipolar filtering kept {inliers_total:,} inlier match(es) across "
                f"{usable_pairs}/{filtered} processed pair(s); minimum {cfg.min_inliers} inliers per usable pair"
            ),
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
            if pair.id is not None and pair.status == "filtered" and pair.num_epipolar_inliers > 0
        }
        inlier_match_count = sum(len(v) for v in matches_by_pair.values())
        _emit(progress, f"Building tracks: 0/{inlier_match_count} inlier matches")
        tracks = build_tracks_from_matches(pairs, matches_by_pair, poses_by_image, camera_model, cfg)
        _emit(progress, f"Triangulating tracks: {len(tracks)}/{len(tracks)} accepted tracks")
        self.store.replace_tracks_and_points(tracks)
        observation_count = sum(len(observations) for _track, observations, _point in tracks)
        return DenseStageSummary(
            stage="track_triangulation",
            total=inlier_match_count,
            success=len(tracks),
            details=(
                f"Built {len(tracks)} active dense point track(s) from {inlier_match_count:,} epipolar inliers; "
                f"{observation_count:,} retained track observations"
            ),
        )

    def merge_duplicates(
        self,
        config: DuplicateMergeConfig | None = None,
        progress: ProgressCallback | None = None,
    ) -> DenseStageSummary:
        cfg = config or DuplicateMergeConfig()
        points = points_from_rows(self.store.list_active_dense_points())
        if not points:
            return DenseStageSummary(stage="duplicate_merge", details="No active dense points are available to merge.")
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

    def run_all(
        self,
        config: DenseReconstructionConfig | None = None,
        progress: ProgressCallback | None = None,
    ) -> DensePipelineRunSummary:
        cfg = config or DenseReconstructionConfig()
        stages: list[DenseStageSummary] = []
        skipped: list[str] = []
        total_stages = 7

        def run_stage(index: int, label: str, method: Callable[..., DenseStageSummary], stage_config: object) -> DenseStageSummary:
            _emit(progress, f"[{index}/{total_stages}] {label}")
            summary = method(
                stage_config,
                lambda message: _emit(progress, f"[{index}/{total_stages}] {label}: {message}"),
            )
            stages.append(summary)
            _emit(progress, f"[{index}/{total_stages}] {summary.details}")
            return summary

        feature_summary = run_stage(1, "Feature extraction", self.extract_features, cfg.extraction)
        counts = self.store.dense_counts()
        if feature_summary.success < 2 or counts["feature_images"] < 2:
            return _stopped_run(stages, skipped, "Feature extraction", "At least two successful feature images are required.")

        pair_summary = run_stage(2, "Frame-pair selection", self.build_frame_pairs, cfg.pair_selection)
        counts = self.store.dense_counts()
        if pair_summary.success == 0 or counts["pairs"] == 0:
            return _stopped_run(stages, skipped, "Frame-pair selection", "No candidate image pairs passed the overlap and baseline filters.")

        match_summary = run_stage(3, "XFeat matching", self.match_frame_pairs, cfg.matching)
        counts = self.store.dense_counts()
        if match_summary.success == 0 or counts["matches"] == 0:
            return _stopped_run(stages, skipped, "XFeat matching", "No refined feature matches were produced.")

        filter_summary = run_stage(4, "Epipolar filtering", self.filter_matches, cfg.epipolar)
        counts = self.store.dense_counts()
        if filter_summary.success == 0:
            return _stopped_run(
                stages,
                skipped,
                "Epipolar filtering",
                "No image pair retained the required number of camera-pose-consistent matches.",
            )

        track_summary = run_stage(5, "Track building and triangulation", self.build_tracks_and_triangulate, cfg.triangulation)
        counts = self.store.dense_counts()
        if track_summary.success == 0 or counts["points"] == 0:
            return _stopped_run(stages, skipped, "Track building and triangulation", "No geometrically valid 3D tracks were reconstructed.")

        ba_availability = check_dense_ba_availability()
        if ba_availability.available:
            ba_summary = run_stage(6, "Structure-only dense BA", self.run_dense_ba, cfg.dense_ba)
            counts = self.store.dense_counts()
            if ba_summary.success == 0 or counts["points"] == 0:
                return _stopped_run(stages, skipped, "Structure-only dense BA", "All dense points failed post-BA quality checks.")
        else:
            skipped.append("dense_ba")
            _emit(progress, f"[6/{total_stages}] Dense BA skipped: {ba_availability.details}")

        run_stage(7, "Duplicate merging", self.merge_duplicates, cfg.duplicate_merge)
        counts = self.store.dense_counts()
        details = (
            "Dense reconstruction complete: "
            f"{counts['feature_images']} feature image(s), {counts['pairs']} pair(s), "
            f"{counts['matches']:,} refined match(es), {counts['inliers']:,} epipolar inlier(s), "
            f"{counts['points']:,} active point(s)"
        )
        if skipped:
            details += "; skipped: " + ", ".join(skipped)
        return DensePipelineRunSummary(stages=stages, skipped_stages=skipped, success=True, details=details)

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


def _stopped_run(
    stages: list[DenseStageSummary],
    skipped: list[str],
    stage_label: str,
    reason: str,
) -> DensePipelineRunSummary:
    return DensePipelineRunSummary(
        stages=stages,
        skipped_stages=skipped,
        success=False,
        details=f"Dense reconstruction stopped after {stage_label}: {reason}",
    )


def _emit(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _is_current_semidense_feature(feature: Any) -> bool:
    return (
        getattr(feature, "extraction_mode", None) == "semi_dense_xfeat"
        and getattr(feature, "descriptor_source", None) == "detectAndComputeDense"
    )


def _matching_details(
    total_candidate_pairs: int,
    processed_pairs: int,
    matched_pairs: int,
    no_match_pairs: int,
    skipped_pairs: int,
    failed_pairs: int,
    raw_matches: int,
    per_pair_counts: list[int],
    failure_reasons: list[str],
) -> str:
    if per_pair_counts:
        min_count = min(per_pair_counts)
        median_count = int(statistics.median(per_pair_counts))
        max_count = max(per_pair_counts)
        per_pair = f"; per-pair matches min/median/max = {min_count:,}/{median_count:,}/{max_count:,}"
    else:
        per_pair = "; per-pair matches min/median/max = 0/0/0"
    parts = [
        f"Matched {matched_pairs} pair(s), {raw_matches:,} raw matches{per_pair}",
        f"processed {processed_pairs}/{total_candidate_pairs} candidate pair(s)",
        f"zero-match pairs {no_match_pairs}",
        f"skipped {skipped_pairs} incompatible/missing pair(s)",
        f"failed {failed_pairs} pair(s)",
    ]
    if failure_reasons:
        unique_reasons = list(dict.fromkeys(reason for reason in failure_reasons if reason))[:3]
        if unique_reasons:
            parts.append("reasons: " + " | ".join(unique_reasons))
    return "; ".join(parts)
