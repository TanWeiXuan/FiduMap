from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import ttk
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


class DenseControlPanel(ttk.LabelFrame):
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
        super().__init__(master, text="Dense Reconstruction (Optional)", **kwargs)
        self.project_folder: Path | None = None
        self.available = False
        self.extraction_available = False
        self.matching_available = False
        self.ba_available = False
        self.running = False
        self.status_var = tk.StringVar(value="Choose an image folder first")
        self.counts_var = tk.StringVar(value="")
        self.max_keypoints_var = tk.StringVar(value="20000")
        self.device_var = tk.StringVar(value="auto")
        self.max_pairs_var = tk.StringVar(value="10")
        self.min_baseline_var = tk.StringVar(value="0.03")
        self.max_axis_angle_var = tk.StringVar(value="60.0")
        self.epipolar_threshold_var = tk.StringVar(value="0.003")
        self.min_triangulation_angle_var = tk.StringVar(value="1.0")
        self.max_reprojection_error_var = tk.StringVar(value="6.0")
        self.duplicate_radius_var = tk.StringVar(value="0.02")
        self.ba_mode_var = tk.StringVar(value="points_only")
        self.buttons: list[ttk.Button] = []
        self.button_by_stage: dict[str, ttk.Button] = {}
        self.ba_button: ttk.Button | None = None

        self.columnconfigure(0, weight=1)
        ttk.Label(self, textvariable=self.status_var, wraplength=340, justify="left").grid(
            row=0, column=0, sticky="ew", padx=6, pady=(6, 2)
        )
        ttk.Label(self, textvariable=self.counts_var, wraplength=340, justify="left").grid(
            row=1, column=0, sticky="ew", padx=6, pady=(0, 6)
        )
        config = ttk.Frame(self)
        config.grid(row=2, column=0, sticky="ew", padx=6, pady=4)
        config.columnconfigure(1, weight=1)
        fields = [
            ("Max keypoints", self.max_keypoints_var),
            ("Device", self.device_var),
            ("Max pairs / image", self.max_pairs_var),
            ("Min baseline (m)", self.min_baseline_var),
            ("Max axis angle", self.max_axis_angle_var),
            ("Epipolar threshold", self.epipolar_threshold_var),
            ("Min tri angle", self.min_triangulation_angle_var),
            ("Max reproj px", self.max_reprojection_error_var),
            ("Duplicate radius", self.duplicate_radius_var),
            ("BA mode", self.ba_mode_var),
        ]
        for row, (label, var) in enumerate(fields):
            ttk.Label(config, text=label).grid(row=row, column=0, sticky="w", padx=(0, 6), pady=1)
            if label in {"Device", "BA mode"}:
                values = ["auto", "cpu", "cuda"] if label == "Device" else ["points_only", "points_and_cameras", "full"]
                ttk.Combobox(config, textvariable=var, values=values, state="readonly", width=16).grid(
                    row=row, column=1, sticky="ew", pady=1
                )
            else:
                ttk.Entry(config, textvariable=var, width=16).grid(row=row, column=1, sticky="ew", pady=1)

        actions = ttk.Frame(self)
        actions.grid(row=3, column=0, sticky="ew", padx=6, pady=4)
        actions.columnconfigure(0, weight=1)
        specs = [
            ("extract", "Run XFeat Feature Extraction", run_extract_features),
            ("pairs", "Build Frame Pair Overlap Graph", run_build_pairs),
            ("match", "Run Pair Matching", run_match_pairs),
            ("filter", "Run Epipolar Filtering", run_filter_matches),
            ("tracks", "Build Tracks + Triangulate", run_build_tracks),
            ("merge", "Merge Duplicates", run_merge_duplicates),
            ("ba", "Run Dense Point BA", run_dense_ba),
            ("export", "Export Dense Point Cloud CSV", export_dense_csv),
        ]
        for row, (stage, text, command) in enumerate(specs):
            btn = ttk.Button(actions, text=text, command=command)
            btn.grid(row=row, column=0, sticky="ew", pady=2)
            self.buttons.append(btn)
            self.button_by_stage[stage] = btn
            if stage == "ba":
                self.ba_button = btn
        self.refresh_availability()

    def set_project(self, folder: Path | None) -> None:
        self.project_folder = folder
        self.refresh_availability()

    def refresh_availability(self) -> None:
        res = check_dense_reconstruction_availability()
        extraction_res = check_xfeat_extraction_availability()
        matching_res = check_xfeat_matching_availability()
        ba_res = check_dense_ba_availability()
        self.available = res.available
        self.extraction_available = extraction_res.available
        self.matching_available = matching_res.available
        self.ba_available = ba_res.available
        details = res.details
        if not extraction_res.available:
            details = f"{details}\n{extraction_res.details}"
        if not matching_res.available:
            details = f"{details}\n{matching_res.details}"
        if not ba_res.available:
            details = f"{details}\n{ba_res.details}"
        self.status_var.set(details if self.project_folder is not None else "Choose an image folder first")
        self._update_button_states()

    def set_status(self, text: str) -> None:
        self.status_var.set(text)

    def set_counts(self, counts: dict[str, int] | None) -> None:
        if not counts:
            self.counts_var.set("")
            return
        self.counts_var.set(
            "Features: {features}  Pairs: {pairs}  Matches: {matches}\n"
            "Inliers: {inliers}  Tracks: {tracks}  Points: {points}".format(**counts)
        )

    def set_running(self, running: bool) -> None:
        self.running = running
        self._update_button_states()

    def extraction_config(self) -> XFeatExtractionConfig:
        return XFeatExtractionConfig(max_keypoints=int(self.max_keypoints_var.get()), device=self.device_var.get())

    def pair_selection_config(self) -> PairSelectionConfig:
        return PairSelectionConfig(
            max_pairs_per_image=int(self.max_pairs_var.get()),
            min_baseline_m=float(self.min_baseline_var.get()),
            max_optical_axis_angle_deg=float(self.max_axis_angle_var.get()),
        )

    def matching_config(self) -> MatchingConfig:
        return MatchingConfig(device=self.device_var.get())

    def epipolar_config(self) -> EpipolarFilterConfig:
        return EpipolarFilterConfig(max_ray_angular_error=float(self.epipolar_threshold_var.get()))

    def triangulation_config(self) -> TriangulationConfig:
        return TriangulationConfig(
            min_triangulation_angle_deg=float(self.min_triangulation_angle_var.get()),
            max_reprojection_error_px=float(self.max_reprojection_error_var.get()),
            max_mean_reprojection_error_px=float(self.max_reprojection_error_var.get()) / 2.0,
        )

    def duplicate_config(self) -> DuplicateMergeConfig:
        return DuplicateMergeConfig(duplicate_merge_radius_m=float(self.duplicate_radius_var.get()))

    def dense_ba_config(self) -> DenseBAConfig:
        return DenseBAConfig(mode=self.ba_mode_var.get())

    def _update_button_states(self) -> None:
        has_project = self.project_folder is not None
        export_state = "normal" if has_project and not self.running else "disabled"
        for btn in self.buttons:
            btn.configure(state="disabled")
        base_state = "normal" if has_project and not self.running else "disabled"
        for stage in ["pairs", "filter", "tracks", "merge"]:
            self.button_by_stage[stage].configure(state=base_state)
        self.button_by_stage["extract"].configure(
            state="normal" if has_project and self.extraction_available and not self.running else "disabled"
        )
        self.button_by_stage["match"].configure(
            state="normal" if has_project and self.matching_available and not self.running else "disabled"
        )
        if self.ba_button is not None:
            self.ba_button.configure(
                state="normal" if has_project and self.ba_available and not self.running else "disabled"
            )
        self.button_by_stage["export"].configure(state=export_state)
