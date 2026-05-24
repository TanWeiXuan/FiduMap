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
        **kwargs: object,
    ):
        super().__init__(master, **kwargs)
        self.folder_var = tk.StringVar(value="No image folder selected")
        self.camera_var = tk.StringVar(value="No camera config selected")
        self.status_var = tk.StringVar(value="Ready")
        self.detector_type_var = tk.StringVar(value="ArUco")
        self.dictionary_var = tk.StringVar(value=ARUCO_DICTIONARY_CHOICES[0])

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
        frame.bind("<Configure>", self._update_wraplengths)

    def set_paths(self, folder: str | None, camera: str | None) -> None:
        self.folder_var.set(folder or "No image folder selected")
        self.camera_var.set(camera or "No camera config selected")

    def set_status(self, text: str) -> None:
        self.status_var.set(text)

    def set_detection_running(self, running: bool) -> None:
        self.run_button.configure(state="disabled" if running else "normal")

    def detector_type(self) -> str:
        return self.detector_type_var.get()

    def dictionary_name(self) -> str:
        return self.dictionary_var.get()

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
