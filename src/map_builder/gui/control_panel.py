"""Left-side workflow controls."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from map_builder.detection import APRILTAG_DICTIONARY_CHOICES, ARUCO_DICTIONARY_CHOICES

from .scrollable_frame import ScrollableFrame


class ControlPanel(ScrollableFrame):
    def __init__(
        self,
        master: tk.Misc,
        refresh_images: object,
        ignore_selected: object,
        unignore_selected: object,
        run_detection: object,
        save_initialization_settings: object,
        run_pnp_initialization: object,
        build_graph_and_seed: object,
        run_ba_refinement: object,
        export_optimized_csv: object,
        **kwargs: object,
    ):
        super().__init__(master, **kwargs)
        self.folder_var = tk.StringVar(value="No image folder selected")
        self.camera_var = tk.StringVar(value="No camera config selected")
        self.status_var = tk.StringVar(value="Ready")
        self.detector_type_var = tk.StringVar(value="ArUco")
        self.dictionary_var = tk.StringVar(value=ARUCO_DICTIONARY_CHOICES[0])
        self.marker_size_var = tk.StringVar(value="")
        self.anchor_marker_var = tk.StringVar(value="")
        self.pnp_status_var = tk.StringVar(value="PnP not run")
        self.graph_status_var = tk.StringVar(value="Graph not built")
        self.ba_status_var = tk.StringVar(value="BA not run")
        self.robust_loss_type_var = tk.StringVar(value="huber")
        self.robust_loss_scale_var = tk.StringVar(value="3.0")
        self.corner_outlier_threshold_var = tk.StringVar(value="10.0")
        self.marker_outlier_threshold_var = tk.StringVar(value="5.0")
        self.ba_second_pass_var = tk.BooleanVar(value=True)

        frame = self.inner
        frame.columnconfigure(0, weight=1)

        ttk.Label(frame, text="Project").grid(row=0, column=0, sticky="w", padx=8, pady=(8, 2))
        ttk.Label(frame, text="Image folder").grid(row=1, column=0, sticky="w", padx=8, pady=(4, 0))
        self.folder_label = ttk.Label(frame, textvariable=self.folder_var, justify="left")
        self.folder_label.grid(row=2, column=0, sticky="ew", padx=8)
        ttk.Label(frame, text="Camera config XML").grid(row=3, column=0, sticky="w", padx=8, pady=(8, 0))
        self.camera_label = ttk.Label(frame, textvariable=self.camera_var, justify="left")
        self.camera_label.grid(row=4, column=0, sticky="ew", padx=8, pady=(0, 8))

        ttk.Button(frame, text="Refresh / Re-index Images", command=refresh_images).grid(
            row=5, column=0, sticky="ew", padx=8, pady=3
        )
        selection_actions = ttk.Frame(frame)
        selection_actions.grid(row=6, column=0, sticky="ew", padx=8, pady=3)
        selection_actions.columnconfigure(0, weight=1, uniform="selection_actions")
        selection_actions.columnconfigure(1, weight=1, uniform="selection_actions")
        ttk.Button(selection_actions, text="Ignore Selected", command=ignore_selected).grid(
            row=0, column=0, sticky="ew", padx=(0, 3)
        )
        ttk.Button(selection_actions, text="Unignore Selected", command=unignore_selected).grid(
            row=0, column=1, sticky="ew", padx=(3, 0)
        )

        ttk.Separator(frame).grid(row=7, column=0, sticky="ew", padx=8, pady=10)
        ttk.Label(frame, text="Detector").grid(row=8, column=0, sticky="w", padx=8)
        self.detector_combo = ttk.Combobox(
            frame,
            textvariable=self.detector_type_var,
            values=["ArUco", "AprilTag"],
            state="readonly",
        )
        self.detector_combo.grid(row=9, column=0, sticky="ew", padx=8, pady=3)
        self.detector_combo.bind("<<ComboboxSelected>>", self._on_detector_type_changed)

        ttk.Label(frame, text="Dictionary").grid(row=10, column=0, sticky="w", padx=8, pady=(6, 0))
        self.dictionary_combo = ttk.Combobox(
            frame,
            textvariable=self.dictionary_var,
            values=ARUCO_DICTIONARY_CHOICES,
            state="readonly",
        )
        self.dictionary_combo.grid(row=11, column=0, sticky="ew", padx=8, pady=3)

        self.run_button = ttk.Button(frame, text="Run Detection On Non-Ignored Images", command=run_detection)
        self.run_button.grid(row=12, column=0, sticky="ew", padx=8, pady=(10, 3))
        self.status_label = ttk.Label(frame, textvariable=self.status_var, justify="left")
        self.status_label.grid(row=13, column=0, sticky="ew", padx=8, pady=8)

        ttk.Separator(frame).grid(row=14, column=0, sticky="ew", padx=8, pady=10)
        ttk.Label(frame, text="Marker Geometry / PnP").grid(row=15, column=0, sticky="w", padx=8)
        marker_grid = ttk.Frame(frame)
        marker_grid.grid(row=16, column=0, sticky="ew", padx=8, pady=3)
        marker_grid.columnconfigure(1, weight=1)
        ttk.Label(marker_grid, text="Marker side length (m)").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=2)
        ttk.Entry(marker_grid, textvariable=self.marker_size_var, width=12).grid(row=0, column=1, sticky="ew", pady=2)
        ttk.Label(marker_grid, text="Anchor marker ID (blank = smallest detected)").grid(
            row=1, column=0, sticky="w", padx=(0, 6), pady=2
        )
        ttk.Entry(marker_grid, textvariable=self.anchor_marker_var, width=12).grid(row=1, column=1, sticky="ew", pady=2)
        pnp_actions = ttk.Frame(frame)
        pnp_actions.grid(row=17, column=0, sticky="ew", padx=8, pady=3)
        pnp_actions.columnconfigure(0, weight=1)
        pnp_actions.columnconfigure(1, weight=1)
        ttk.Button(pnp_actions, text="Save Settings", command=save_initialization_settings).grid(
            row=0, column=0, sticky="ew", padx=(0, 3)
        )
        self.pnp_button = ttk.Button(pnp_actions, text="Run PnP Initialization", command=run_pnp_initialization)
        self.pnp_button.grid(row=0, column=1, sticky="ew", padx=(3, 0))
        self.pnp_status_label = ttk.Label(frame, textvariable=self.pnp_status_var, justify="left")
        self.pnp_status_label.grid(row=18, column=0, sticky="ew", padx=8, pady=6)

        ttk.Separator(frame).grid(row=19, column=0, sticky="ew", padx=8, pady=10)
        ttk.Label(frame, text="Graph / Seed Initialization").grid(row=20, column=0, sticky="w", padx=8)
        self.graph_button = ttk.Button(frame, text="Build Graph + Initialize Seed Poses", command=build_graph_and_seed)
        self.graph_button.grid(row=21, column=0, sticky="ew", padx=8, pady=3)
        self.graph_status_label = ttk.Label(frame, textvariable=self.graph_status_var, justify="left")
        self.graph_status_label.grid(row=22, column=0, sticky="ew", padx=8, pady=6)

        ttk.Separator(frame).grid(row=23, column=0, sticky="ew", padx=8, pady=10)
        ttk.Label(frame, text="BA Refinement").grid(row=24, column=0, sticky="w", padx=8)
        ba_grid = ttk.Frame(frame)
        ba_grid.grid(row=25, column=0, sticky="ew", padx=8, pady=3)
        ba_grid.columnconfigure(1, weight=1)
        ttk.Label(ba_grid, text="Robust loss").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=2)
        ttk.Combobox(
            ba_grid,
            textvariable=self.robust_loss_type_var,
            values=["none", "huber", "cauchy"],
            state="readonly",
            width=12,
        ).grid(row=0, column=1, sticky="ew", pady=2)
        ttk.Label(ba_grid, text="Loss scale (px)").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=2)
        ttk.Entry(ba_grid, textvariable=self.robust_loss_scale_var, width=12).grid(row=1, column=1, sticky="ew", pady=2)
        ttk.Label(ba_grid, text="Corner outlier (px)").grid(row=2, column=0, sticky="w", padx=(0, 6), pady=2)
        ttk.Entry(ba_grid, textvariable=self.corner_outlier_threshold_var, width=12).grid(
            row=2, column=1, sticky="ew", pady=2
        )
        ttk.Label(ba_grid, text="Marker outlier mean (px)").grid(row=3, column=0, sticky="w", padx=(0, 6), pady=2)
        ttk.Entry(ba_grid, textvariable=self.marker_outlier_threshold_var, width=12).grid(
            row=3, column=1, sticky="ew", pady=2
        )
        ttk.Checkbutton(
            frame,
            text="Run second pass after outlier rejection",
            variable=self.ba_second_pass_var,
        ).grid(row=26, column=0, sticky="w", padx=8, pady=3)
        ba_actions = ttk.Frame(frame)
        ba_actions.grid(row=27, column=0, sticky="ew", padx=8, pady=3)
        ba_actions.columnconfigure(0, weight=1)
        ba_actions.columnconfigure(1, weight=1)
        self.ba_button = ttk.Button(ba_actions, text="Run BA Refinement", command=run_ba_refinement)
        self.ba_button.grid(row=0, column=0, sticky="ew", padx=(0, 3))
        self.export_button = ttk.Button(ba_actions, text="Export Optimized CSV", command=export_optimized_csv)
        self.export_button.grid(row=0, column=1, sticky="ew", padx=(3, 0))
        self.ba_status_label = ttk.Label(frame, textvariable=self.ba_status_var, justify="left")
        self.ba_status_label.grid(row=28, column=0, sticky="ew", padx=8, pady=6)
        frame.bind("<Configure>", self._update_wraplengths)

    def set_paths(self, folder: str | None, camera: str | None) -> None:
        self.folder_var.set(folder or "No image folder selected")
        self.camera_var.set(camera or "No camera config selected")

    def set_status(self, text: str) -> None:
        self.status_var.set(text)

    def set_detection_running(self, running: bool) -> None:
        self.run_button.configure(state="disabled" if running else "normal")

    def set_pnp_running(self, running: bool) -> None:
        self.pnp_button.configure(state="disabled" if running else "normal")

    def set_graph_running(self, running: bool) -> None:
        self.graph_button.configure(state="disabled" if running else "normal")

    def set_ba_running(self, running: bool) -> None:
        state = "disabled" if running else "normal"
        self.ba_button.configure(state=state)
        self.export_button.configure(state=state)

    def detector_type(self) -> str:
        return self.detector_type_var.get()

    def dictionary_name(self) -> str:
        return self.dictionary_var.get()

    def marker_size_m(self) -> float:
        return float(self.marker_size_var.get())

    def anchor_marker_id(self) -> int | None:
        text = self.anchor_marker_var.get().strip()
        return None if text == "" else int(text)

    def set_initialization_settings(self, marker_size_m: float | None, anchor_marker_id: int | None) -> None:
        self.marker_size_var.set("" if marker_size_m is None else f"{marker_size_m:g}")
        self.anchor_marker_var.set("" if anchor_marker_id is None else str(anchor_marker_id))

    def set_pnp_status(self, text: str) -> None:
        self.pnp_status_var.set(text)

    def set_graph_status(self, text: str) -> None:
        self.graph_status_var.set(text)

    def ba_config_values(self) -> tuple[str, float, float, float, bool]:
        return (
            self.robust_loss_type_var.get(),
            float(self.robust_loss_scale_var.get()),
            float(self.corner_outlier_threshold_var.get()),
            float(self.marker_outlier_threshold_var.get()),
            bool(self.ba_second_pass_var.get()),
        )

    def set_ba_status(self, text: str) -> None:
        self.ba_status_var.set(text)

    def set_ba_config_values(
        self,
        robust_loss_type: str,
        robust_loss_scale_px: float,
        corner_outlier_threshold_px: float,
        marker_observation_outlier_threshold_px: float,
        run_outlier_second_pass: bool,
    ) -> None:
        self.robust_loss_type_var.set(robust_loss_type)
        self.robust_loss_scale_var.set(f"{robust_loss_scale_px:g}")
        self.corner_outlier_threshold_var.set(f"{corner_outlier_threshold_px:g}")
        self.marker_outlier_threshold_var.set(f"{marker_observation_outlier_threshold_px:g}")
        self.ba_second_pass_var.set(run_outlier_second_pass)

    def _on_detector_type_changed(self, _event: tk.Event) -> None:
        if self.detector_type_var.get() == "AprilTag":
            choices = APRILTAG_DICTIONARY_CHOICES
        else:
            choices = ARUCO_DICTIONARY_CHOICES
        self.dictionary_combo.configure(values=choices)
        self.dictionary_var.set(choices[0])

    def _update_wraplengths(self, event: tk.Event) -> None:
        wraplength = max(event.width - 24, 160)
        self.folder_label.configure(wraplength=wraplength)
        self.camera_label.configure(wraplength=wraplength)
        self.status_label.configure(wraplength=wraplength)
        self.pnp_status_label.configure(wraplength=wraplength)
        self.graph_status_label.configure(wraplength=wraplength)
        self.ba_status_label.configure(wraplength=wraplength)
