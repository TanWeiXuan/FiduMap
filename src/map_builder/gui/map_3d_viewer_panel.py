"""Matplotlib-backed 3D seed map viewer."""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import ttk
from typing import Any

import numpy as np

from map_builder.geometry import SE3, marker_corners_y_up
from map_builder.project import ProjectStore


class Map3DViewerPanel(ttk.Frame):
    def __init__(self, master: tk.Misc, **kwargs: object):
        super().__init__(master, **kwargs)
        self.project_folder: Path | None = None
        self.store: ProjectStore | None = None
        self.selected_image_id: int | None = None
        self.show_cameras_var = tk.BooleanVar(value=True)
        self.show_markers_var = tk.BooleanVar(value=True)
        self.show_labels_var = tk.BooleanVar(value=True)
        self.selected_only_var = tk.BooleanVar(value=False)
        self._matplotlib_ready = False

        controls = ttk.Frame(self)
        controls.grid(row=0, column=0, sticky="ew", padx=6, pady=6)
        controls.columnconfigure(4, weight=1)
        ttk.Button(controls, text="Refresh 3D View", command=self.refresh).grid(row=0, column=0, padx=(0, 8))
        ttk.Checkbutton(controls, text="Cameras", variable=self.show_cameras_var, command=self.refresh).grid(row=0, column=1)
        ttk.Checkbutton(controls, text="Markers", variable=self.show_markers_var, command=self.refresh).grid(row=0, column=2)
        ttk.Checkbutton(controls, text="Labels", variable=self.show_labels_var, command=self.refresh).grid(row=0, column=3)
        ttk.Checkbutton(
            controls,
            text="Selected image only",
            variable=self.selected_only_var,
            command=self.refresh,
        ).grid(row=0, column=4, sticky="w")

        self.plot_frame = ttk.Frame(self)
        self.plot_frame.grid(row=1, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        try:
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
            from matplotlib.figure import Figure

            self.figure = Figure(figsize=(6, 5), dpi=100)
            self.axes = self.figure.add_subplot(111, projection="3d")
            self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
            self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
            self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame, pack_toolbar=False)
            self.toolbar.grid(row=1, column=0, sticky="ew")
            self.plot_frame.columnconfigure(0, weight=1)
            self.plot_frame.rowconfigure(0, weight=1)
            self._matplotlib_ready = True
            self._FigureCanvasTkAgg = FigureCanvasTkAgg
            self._NavigationToolbar2Tk = NavigationToolbar2Tk
            self.refresh()
        except Exception as exc:
            ttk.Label(self.plot_frame, text=f"Matplotlib 3D view unavailable: {exc}", wraplength=500).grid(
                row=0, column=0, sticky="nsew", padx=12, pady=12
            )

    def set_project(self, project_folder: Path | None, store: ProjectStore | None) -> None:
        self.project_folder = project_folder
        self.store = store
        self.refresh()

    def set_selected_image_id(self, image_id: int | None) -> None:
        self.selected_image_id = image_id
        if self.selected_only_var.get():
            self.refresh()

    def refresh(self) -> None:
        if not self._matplotlib_ready:
            return
        self.axes.clear()
        self.axes.set_xlabel("X")
        self.axes.set_ylabel("Y")
        self.axes.set_zlabel("Z")
        if self.store is None:
            self.axes.text(0, 0, 0, "No project loaded")
            self.canvas.draw_idle()
            return

        marker_size = self.store.get_marker_size_m() or 0.1
        seed_cameras = self.store.get_seed_camera_poses()
        seed_markers = self.store.get_seed_marker_poses()
        if seed_cameras or seed_markers:
            self._draw_seed_map(marker_size, seed_cameras, seed_markers)
        else:
            self._draw_selected_frame_local(marker_size)
        self._set_equal_axes()
        self.canvas.draw_idle()

    def _draw_seed_map(self, marker_size: float, seed_cameras: list[Any], seed_markers: list[Any]) -> None:
        if self.show_cameras_var.get():
            for pose in seed_cameras:
                if self.selected_only_var.get() and self.selected_image_id is not None and pose.image_id != self.selected_image_id:
                    continue
                T_W_C = SE3.from_json_dict(pose.T_W_C)
                self._draw_camera(T_W_C, f"C{pose.image_id}")
        if self.show_markers_var.get():
            local_corners = marker_corners_y_up(marker_size)
            for pose in seed_markers:
                T_W_M = SE3.from_json_dict(pose.T_W_M)
                corners_w = T_W_M.transform_points(local_corners)
                self._draw_marker_square(corners_w, pose.marker_id)

    def _draw_selected_frame_local(self, marker_size: float) -> None:
        self.axes.text(0, 0, 0, "No seed poses yet: showing selected image PnP frame")
        if self.selected_image_id is None:
            return
        observations = [
            obs
            for obs in self.store.list_pnp_observations(success_only=True)  # type: ignore[union-attr]
            if obs.image_id == self.selected_image_id and obs.T_C_M is not None
        ]
        self._draw_camera(SE3.identity(), "C")
        local_corners = marker_corners_y_up(marker_size)
        for obs in observations:
            T_C_M = SE3.from_json_dict(obs.T_C_M)
            self._draw_marker_square(T_C_M.transform_points(local_corners), obs.marker_id)

    def _draw_camera(self, T_W_C: SE3, label: str) -> None:
        center = T_W_C.t
        self.axes.scatter([center[0]], [center[1]], [center[2]], c="tab:blue", s=24)
        axis_len = 0.06
        axes = T_W_C.R @ np.eye(3) * axis_len
        colors = ["r", "g", "b"]
        for axis, color in zip(axes.T, colors):
            self.axes.plot(
                [center[0], center[0] + axis[0]],
                [center[1], center[1] + axis[1]],
                [center[2], center[2] + axis[2]],
                color=color,
            )
        if self.show_labels_var.get():
            self.axes.text(center[0], center[1], center[2], label)

    def _draw_marker_square(self, corners: np.ndarray, marker_id: int) -> None:
        closed = np.vstack([corners, corners[0]])
        self.axes.plot(closed[:, 0], closed[:, 1], closed[:, 2], color="tab:orange")
        self.axes.scatter(corners[:, 0], corners[:, 1], corners[:, 2], c="tab:orange", s=10)
        if self.show_labels_var.get():
            center = np.mean(corners, axis=0)
            self.axes.text(center[0], center[1], center[2], f"M{marker_id}")

    def _set_equal_axes(self) -> None:
        xs: list[float] = []
        ys: list[float] = []
        zs: list[float] = []
        for line in self.axes.lines:
            x, y, z = line.get_data_3d()
            xs.extend(x)
            ys.extend(y)
            zs.extend(z)
        for collection in self.axes.collections:
            offsets = getattr(collection, "_offsets3d", None)
            if offsets is not None:
                xs.extend(offsets[0])
                ys.extend(offsets[1])
                zs.extend(offsets[2])
        if not xs:
            self.axes.set_xlim(-0.5, 0.5)
            self.axes.set_ylim(-0.5, 0.5)
            self.axes.set_zlim(-0.5, 0.5)
            return
        mins = np.array([min(xs), min(ys), min(zs)], dtype=float)
        maxs = np.array([max(xs), max(ys), max(zs)], dtype=float)
        center = (mins + maxs) / 2.0
        radius = max(float(np.max(maxs - mins)) / 2.0, 0.1)
        self.axes.set_xlim(center[0] - radius, center[0] + radius)
        self.axes.set_ylim(center[1] - radius, center[1] + radius)
        self.axes.set_zlim(center[2] - radius, center[2] + radius)
