from __future__ import annotations

from pathlib import Path
import tkinter as tk
from typing import Callable

from map_builder.dense_reconstruction.availability import (
    check_dense_ba_availability, check_dense_reconstruction_availability,
    check_xfeat_extraction_availability, check_xfeat_matching_availability,
)
from map_builder.dense_reconstruction.models import (
    DenseBAConfig, DuplicateMergeConfig, EpipolarFilterConfig, MatchingConfig,
    PairSelectionConfig, TriangulationConfig, XFeatExtractionConfig,
)
from .dense_control_panel_ui import build_dense_controls
from .scrollable_frame import ScrollableFrame


class DenseControlPanel(ScrollableFrame):
    def __init__(self, master: tk.Misc, run_extract_features: Callable[[], None],
                 run_build_pairs: Callable[[], None], run_match_pairs: Callable[[], None],
                 run_filter_matches: Callable[[], None], run_build_tracks: Callable[[], None],
                 run_merge_duplicates: Callable[[], None], run_dense_ba: Callable[[], None],
                 export_dense_csv: Callable[[], None], **kwargs: object):
        super().__init__(master, **kwargs)
        self.project_folder: Path | None = None
        self.available = self.extraction_available = self.matching_available = self.ba_available = False
        self.running = self._auto_running = False
        self._auto_index = 0
        self.counts: dict[str, int] = {}
        self.status_var, self.counts_var = tk.StringVar(value="Choose an image folder first"), tk.StringVar()
        for name, value in {
            "max_keypoints_var": "20000", "device_var": "auto", "max_pairs_var": "10",
            "min_common_markers_var": "1", "min_match_score_var": "0.10",
            "max_reprojection_error_var": "6.0", "duplicate_radius_var": "0.02",
        }.items(): setattr(self, name, tk.StringVar(value=value))
        self.recompute_features_var = tk.BooleanVar(value=False)
        self._auto = [
            ("Feature extraction", run_extract_features, "feature_images", 2),
            ("Frame-pair selection", run_build_pairs, "pairs", 1),
            ("Pair matching", run_match_pairs, "matches", 1),
            ("Epipolar filtering", run_filter_matches, "inliers", 1),
            ("Track triangulation", run_build_tracks, "points", 1),
            ("Dense point BA", run_dense_ba, "points", 1),
            ("Duplicate merging", run_merge_duplicates, "points", 1),
        ]
        callbacks = {
            "extract": run_extract_features, "pairs": run_build_pairs, "match": run_match_pairs,
            "filter": run_filter_matches, "tracks": run_build_tracks, "ba": run_dense_ba,
            "merge": run_merge_duplicates, "export": export_dense_csv,
        }
        self.buttons, self.progress = build_dense_controls(self, callbacks)
        self.button_by_stage, self.run_all_button, self.ba_button = self.buttons, self.buttons["all"], self.buttons["ba"]
        self.refresh_availability()

    def _guard(self, label: str, factory: Callable[[], object], callback: Callable[[], None]) -> Callable[[], None]:
        def run() -> None:
            try: factory()
            except (TypeError, ValueError) as exc: self.set_status(f"Cannot start {label}: {exc}"); return
            callback()
        return run

    def set_project(self, folder: Path | None) -> None: self.project_folder = folder; self.counts = {}; self.refresh_availability()
    def set_status(self, text: str) -> None: self.status_var.set(text)

    def refresh_availability(self) -> None:
        dense, extract = check_dense_reconstruction_availability(), check_xfeat_extraction_availability()
        match, ba = check_xfeat_matching_availability(), check_dense_ba_availability()
        self.available, self.extraction_available, self.matching_available, self.ba_available = \
            dense.available, extract.available, match.available, ba.available
        self.status_var.set("Choose an image folder first" if self.project_folder is None else
                            dense.details if not dense.available else "Ready. Marker-map BA is required after extraction.")
        self._states()

    def set_counts(self, counts: dict[str, int] | None) -> None:
        self.counts = dict(counts or {})
        if not counts: self.counts_var.set(""); self._states(); return
        f, k = int(counts.get("feature_images", counts.get("features", 0))), int(counts.get("keypoints", 0))
        p, mp = int(counts.get("pairs", 0)), int(counts.get("matched_pairs", 0))
        m, i = int(counts.get("matches", 0)), int(counts.get("inliers", 0))
        t, pts = int(counts.get("tracks", 0)), int(counts.get("points", 0))
        ir, pr = (100 * i / m if m else 0), (100 * pts / i if i else 0)
        self.counts_var.set(f"Feature images: {f}  Keypoints: {k:,}\nFrame pairs: {p}  Matched pairs: {mp}\n"
                            f"Raw matches: {m:,}  Inliers: {i:,} ({ir:.1f}%)\n"
                            f"Tracks: {t:,}  Active points: {pts:,} ({pr:.1f}% of inliers)")
        self._states()

    def set_running(self, running: bool) -> None:
        self.running = running; self.progress.start(12) if running else self.progress.stop(); self._states()
        if not running and self._auto_running: self.after_idle(self._continue_auto)

    def _start_auto(self) -> None:
        if self.running or self._auto_running or self.project_folder is None: return
        try: [factory() for factory in (self.extraction_config, self.pair_selection_config, self.matching_config,
                                        self.epipolar_config, self.triangulation_config, self.duplicate_config)]
        except (TypeError, ValueError) as exc: self.set_status(f"Invalid dense setting: {exc}"); return
        if not self.extraction_available or not self.matching_available:
            self.set_status("Dense reconstruction dependencies are unavailable."); return
        self._auto_running, self._auto_index = True, 0; self._next_auto()

    def _next_auto(self) -> None:
        while self._auto_index < len(self._auto):
            label, callback, _key, _minimum = self._auto[self._auto_index]
            if label == "Dense point BA" and not self.ba_available: self._auto_index += 1; continue
            self.set_status(f"Automatic workflow {self._auto_index + 1}/{len(self._auto)}: {label}"); callback(); return
        self._finish_auto(f"Dense reconstruction complete; {int(self.counts.get('points', 0)):,} active point(s).")

    def _continue_auto(self) -> None:
        if not self._auto_running: return
        label, _callback, key, minimum = self._auto[self._auto_index]
        if self.status_var.get() == "Dense reconstruction stage failed" or int(self.counts.get(key, 0)) < minimum:
            self._finish_auto(f"Automatic workflow stopped at {label}; inspect the funnel."); return
        self._auto_index += 1; self._next_auto()

    def _finish_auto(self, message: str) -> None:
        self._auto_running = self.running = False; self.progress.stop(); self.set_status(message); self._states()

    def extraction_config(self) -> XFeatExtractionConfig:
        return XFeatExtractionConfig(max_keypoints=_i(self.max_keypoints_var, "Max keypoints / image", 1),
                                     device=self.device_var.get(), force_recompute=self.recompute_features_var.get())
    def pair_selection_config(self) -> PairSelectionConfig:
        return PairSelectionConfig(max_pairs_per_image=_i(self.max_pairs_var, "Max pairs / image", 1),
                                   min_common_markers=_i(self.min_common_markers_var, "Min common markers", 0))
    def matching_config(self) -> MatchingConfig:
        return MatchingConfig(min_match_score=_f(self.min_match_score_var, "Min descriptor score", -1, 1),
                              device=self.device_var.get(), force_recompute=True)
    def epipolar_config(self) -> EpipolarFilterConfig: return EpipolarFilterConfig()
    def triangulation_config(self) -> TriangulationConfig:
        error = _f(self.max_reprojection_error_var, "Max reprojection error", 0, exclusive=True)
        return TriangulationConfig(max_reprojection_error_px=error, max_mean_reprojection_error_px=error / 2)
    def duplicate_config(self) -> DuplicateMergeConfig:
        return DuplicateMergeConfig(duplicate_merge_radius_m=_f(self.duplicate_radius_var, "Duplicate radius", 0))
    def dense_ba_config(self) -> DenseBAConfig:
        var = getattr(self, "max_reprojection_error_var", None); error = 6 if var is None else _f(var, "Max reprojection error", 0, exclusive=True)
        return DenseBAConfig(mode="points_only", max_reprojection_error_px=error, max_mean_reprojection_error_px=error / 2)

    def _states(self) -> None:
        for button in self.buttons.values(): button.configure(state="disabled")
        if self.project_folder is None or self.running: return
        f = int(self.counts.get("feature_images", self.counts.get("features", 0)))
        p, m, i, pts = (int(self.counts.get(key, 0)) for key in ("pairs", "matches", "inliers", "points"))
        enable = lambda name, ok: self.buttons[name].configure(state="normal" if ok else "disabled")
        enable("extract", self.extraction_available); enable("all", self.extraction_available and self.matching_available)
        enable("pairs", f >= 2); enable("match", bool(p and self.matching_available)); enable("filter", bool(m))
        enable("tracks", bool(i)); enable("ba", bool(pts and self.ba_available)); enable("merge", bool(pts)); enable("export", bool(pts))


def _i(var: tk.StringVar, label: str, minimum: int) -> int:
    value = int(var.get())
    if value < minimum: raise ValueError(f"{label} must be at least {minimum}")
    return value


def _f(var: tk.StringVar, label: str, lower: float, upper: float | None = None, exclusive: bool = False) -> float:
    value = float(var.get())
    if (value <= lower if exclusive else value < lower) or (upper is not None and value > upper):
        raise ValueError(f"{label} is outside the allowed range")
    return value
