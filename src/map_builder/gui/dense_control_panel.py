from __future__ import annotations

from pathlib import Path
import tkinter as tk
from typing import Callable

from map_builder.dense_reconstruction.availability import (
    check_dense_ba_availability,
    check_dense_reconstruction_availability,
    check_xfeat_extraction_availability,
    check_xfeat_matching_availability,
)
from map_builder.dense_reconstruction.models import (
    DenseBAConfig,
    DuplicateMergeConfig,
    EpipolarFilterConfig,
    MatchingConfig,
    PairSelectionConfig,
    TriangulationConfig,
    XFeatExtractionConfig,
)

from .dense_control_panel_ui import build_dense_controls
from .scrollable_frame import ScrollableFrame


class DenseControlPanel(ScrollableFrame):
    def __init__(
        self,
        master: tk.Misc,
        run_extract_features: Callable[[], None],
        run_build_pairs: Callable[[], None],
        run_match_pairs: Callable[[], None],
        run_filter_matches: Callable[[], None],
        run_build_tracks: Callable[[], None],
        run_merge_duplicates: Callable[[], None],
        run_dense_ba: Callable[[], None],
        export_dense_csv: Callable[[], None],
        **kwargs: object,
    ):
        super().__init__(master, **kwargs)
        self.project_folder: Path | None = None
        self.available = False
        self.extraction_available = False
        self.matching_available = False
        self.ba_available = False
        self.running = False
        self._auto_running = False
        self._auto_index = 0
        self._auto_launch_attempts = 0
        self.counts: dict[str, int] = {}

        self.status_var = tk.StringVar(value="Choose an image folder first")
        self.counts_var = tk.StringVar(value="")
        self.max_keypoints_var = tk.StringVar(value="20000")
        self.device_var = tk.StringVar(value="auto")
        self.max_pairs_var = tk.StringVar(value="10")
        self.min_common_markers_var = tk.StringVar(value="1")
        self.min_match_score_var = tk.StringVar(value="0.10")
        self.max_reprojection_error_var = tk.StringVar(value="6.0")
        self.duplicate_radius_var = tk.StringVar(value="0.02")
        self.recompute_features_var = tk.BooleanVar(value=False)

        self._auto_stages: list[tuple[str, Callable[[], None], str, int]] = [
            ("Feature extraction", run_extract_features, "feature_images", 2),
            ("Frame-pair selection", run_build_pairs, "pairs", 1),
            ("Pair matching", run_match_pairs, "matches", 1),
            ("Epipolar filtering", run_filter_matches, "inliers", 1),
            ("Track triangulation", run_build_tracks, "points", 1),
            ("Dense point BA", run_dense_ba, "points", 1),
            ("Duplicate merging", run_merge_duplicates, "points", 1),
        ]
        callbacks = {
            "extract": run_extract_features,
            "pairs": run_build_pairs,
            "match": run_match_pairs,
            "filter": run_filter_matches,
            "tracks": run_build_tracks,
            "ba": run_dense_ba,
            "merge": run_merge_duplicates,
            "export": export_dense_csv,
        }
        self.buttons, self.progress = build_dense_controls(self, callbacks)
        self.button_by_stage = self.buttons
        self.run_all_button = self.buttons["all"]
        self.ba_button = self.buttons["ba"]
        self.refresh_availability()

    def _guard(
        self,
        label: str,
        config_factory: Callable[[], object],
        callback: Callable[[], None],
    ) -> Callable[[], None]:
        def run() -> None:
            try:
                config_factory()
            except (TypeError, ValueError) as exc:
                self.set_status(f"Cannot start {label}: {exc}")
                return
            callback()

        return run

    def set_project(self, folder: Path | None) -> None:
        self.project_folder = folder
        self.counts = {}
        self._auto_running = False
        self._auto_index = 0
        self.refresh_availability()

    def set_status(self, text: str) -> None:
        self.status_var.set(text)

    def refresh_availability(self) -> None:
        dense = check_dense_reconstruction_availability()
        extraction = check_xfeat_extraction_availability()
        matching = check_xfeat_matching_availability()
        ba = check_dense_ba_availability()
        self.available = dense.available
        self.extraction_available = extraction.available
        self.matching_available = matching.available
        self.ba_available = ba.available
        if self.project_folder is None:
            details = "Choose an image folder first"
        elif not dense.available:
            details = dense.details
        else:
            details = "Ready. Run marker-map BA before pair selection."
            if not ba.available:
                details += " Dense point BA will be skipped because pyceres is unavailable."
        self.status_var.set(details)
        self._update_button_states()

    def set_counts(self, counts: dict[str, int] | None) -> None:
        self.counts = dict(counts or {})
        if not counts:
            self.counts_var.set("")
            self._update_button_states()
            return

        feature_images = int(counts.get("feature_images", counts.get("features", 0)))
        keypoints = int(counts.get("keypoints", 0))
        frame_pairs = int(counts.get("pairs", 0))
        matched_pairs = int(counts.get("matched_pairs", 0))
        raw_matches = int(counts.get("matches", 0))
        inliers = int(counts.get("inliers", 0))
        tracks = int(counts.get("tracks", 0))
        points = int(counts.get("points", 0))
        inlier_rate = 100.0 * inliers / raw_matches if raw_matches else 0.0
        point_yield = 100.0 * points / inliers if inliers else 0.0
        self.counts_var.set(
            f"Feature images: {feature_images}  Keypoints: {keypoints:,}\n"
            f"Frame pairs: {frame_pairs}  Matched pairs: {matched_pairs}\n"
            f"Raw matches: {raw_matches:,}  Inliers: {inliers:,} ({inlier_rate:.1f}%)\n"
            f"Tracks: {tracks:,}  Active points: {points:,} ({point_yield:.1f}% of inliers)"
        )
        self._update_button_states()

    def set_running(self, running: bool) -> None:
        self.running = running
        if running:
            self.progress.start(12)
        else:
            self.progress.stop()
        self._update_button_states()
        if not running and self._auto_running:
            # The worker publishes its result before its finally block closes
            # the pipeline store. Delay the next launch so the old worker can
            # exit; _launch_current_auto_stage also retries if it is still alive.
            self.after(100, self._continue_auto)

    def _start_auto(self) -> None:
        if self.running or self._auto_running or self.project_folder is None:
            return
        try:
            self.extraction_config()
            self.pair_selection_config()
            self.matching_config()
            self.epipolar_config()
            self.triangulation_config()
            self.dense_ba_config()
            self.duplicate_config()
        except (TypeError, ValueError) as exc:
            self.set_status(f"Invalid dense setting: {exc}")
            return
        if not self.extraction_available or not self.matching_available:
            self.set_status("Dense reconstruction dependencies are unavailable.")
            return
        self._auto_running = True
        self._auto_index = 0
        self._next_auto_stage()

    def _next_auto_stage(self) -> None:
        while self._auto_index < len(self._auto_stages):
            label, _callback, _key, _minimum = self._auto_stages[self._auto_index]
            if label == "Dense point BA" and not self.ba_available:
                self._auto_index += 1
                continue
            self._auto_launch_attempts = 0
            self._launch_current_auto_stage()
            return
        point_count = int(self.counts.get("points", 0))
        self._finish_auto(f"Dense reconstruction complete; {point_count:,} active point(s).")

    def _launch_current_auto_stage(self) -> None:
        if not self._auto_running or self.running:
            return
        label, callback, _key, _minimum = self._auto_stages[self._auto_index]
        self.set_status(
            f"Automatic workflow {self._auto_index + 1}/{len(self._auto_stages)}: {label}"
        )
        callback()
        if self.running:
            return
        self._auto_launch_attempts += 1
        if self._auto_launch_attempts >= 20:
            self._finish_auto(
                f"Automatic workflow could not start {label}; another dense worker may still be active."
            )
            return
        self.after(100, self._launch_current_auto_stage)

    def _continue_auto(self) -> None:
        if not self._auto_running or self.running:
            return
        label, _callback, count_key, minimum = self._auto_stages[self._auto_index]
        if self.status_var.get() == "Dense reconstruction stage failed":
            self._finish_auto(f"Automatic workflow failed at {label}; review the error message.")
            return
        if int(self.counts.get(count_key, 0)) < minimum:
            self._finish_auto(f"Automatic workflow stopped at {label}; inspect the reconstruction funnel.")
            return
        self._auto_index += 1
        self._next_auto_stage()

    def _finish_auto(self, message: str) -> None:
        self._auto_running = False
        self.running = False
        self.progress.stop()
        self.set_status(message)
        self._update_button_states()

    def extraction_config(self) -> XFeatExtractionConfig:
        return XFeatExtractionConfig(
            max_keypoints=_int_value(self.max_keypoints_var, "Max keypoints / image", 1),
            device=self.device_var.get(),
            force_recompute=self.recompute_features_var.get(),
        )

    def pair_selection_config(self) -> PairSelectionConfig:
        return PairSelectionConfig(
            max_pairs_per_image=_int_value(self.max_pairs_var, "Max pairs / image", 1),
            min_common_markers=_int_value(self.min_common_markers_var, "Min common markers", 0),
        )

    def matching_config(self) -> MatchingConfig:
        return MatchingConfig(
            min_match_score=_float_value(
                self.min_match_score_var,
                "Min descriptor score",
                -1.0,
                1.0,
            ),
            device=self.device_var.get(),
            force_recompute=True,
        )

    def epipolar_config(self) -> EpipolarFilterConfig:
        return EpipolarFilterConfig()

    def triangulation_config(self) -> TriangulationConfig:
        error = _float_value(
            self.max_reprojection_error_var,
            "Max reprojection error",
            0.0,
            exclusive_lower=True,
        )
        return TriangulationConfig(
            max_reprojection_error_px=error,
            max_mean_reprojection_error_px=error / 2.0,
        )

    def duplicate_config(self) -> DuplicateMergeConfig:
        return DuplicateMergeConfig(
            duplicate_merge_radius_m=_float_value(
                self.duplicate_radius_var,
                "Duplicate radius",
                0.0,
            )
        )

    def dense_ba_config(self) -> DenseBAConfig:
        error_var = getattr(self, "max_reprojection_error_var", None)
        error = (
            6.0
            if error_var is None
            else _float_value(
                error_var,
                "Max reprojection error",
                0.0,
                exclusive_lower=True,
            )
        )
        return DenseBAConfig(
            mode="points_only",
            max_reprojection_error_px=error,
            max_mean_reprojection_error_px=error / 2.0,
        )

    def _update_button_states(self) -> None:
        for button in self.buttons.values():
            button.configure(state="disabled")
        if self.project_folder is None or self.running or self._auto_running:
            return

        feature_images = int(self.counts.get("feature_images", self.counts.get("features", 0)))
        pairs = int(self.counts.get("pairs", 0))
        matches = int(self.counts.get("matches", 0))
        inliers = int(self.counts.get("inliers", 0))
        points = int(self.counts.get("points", 0))

        self._set_button_enabled("extract", self.extraction_available)
        self._set_button_enabled("all", self.extraction_available and self.matching_available)
        self._set_button_enabled("pairs", feature_images >= 2)
        self._set_button_enabled("match", pairs > 0 and self.matching_available)
        self._set_button_enabled("filter", matches > 0)
        self._set_button_enabled("tracks", inliers > 0)
        self._set_button_enabled("ba", points > 0 and self.ba_available)
        self._set_button_enabled("merge", points > 0)
        self._set_button_enabled("export", points > 0)

    def _set_button_enabled(self, name: str, enabled: bool) -> None:
        self.buttons[name].configure(state="normal" if enabled else "disabled")


def _int_value(var: tk.StringVar, label: str, minimum: int) -> int:
    value = int(var.get())
    if value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return value


def _float_value(
    var: tk.StringVar,
    label: str,
    lower: float,
    upper: float | None = None,
    exclusive_lower: bool = False,
) -> float:
    value = float(var.get())
    lower_invalid = value <= lower if exclusive_lower else value < lower
    if lower_invalid or (upper is not None and value > upper):
        raise ValueError(f"{label} is outside the allowed range")
    return value
