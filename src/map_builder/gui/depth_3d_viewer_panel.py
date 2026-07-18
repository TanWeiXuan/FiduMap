from __future__ import annotations

import logging
import math
import tkinter as tk
from tkinter import ttk
from typing import Any

import numpy as np

from map_builder.metric_depth.availability import check_vtk_availability
from map_builder.metric_depth.depth_scene import DepthCloudLayer
from map_builder.metric_depth.geometry import deterministic_decimate
from map_builder.geometry.marker_geometry import marker_corners_y_up
from map_builder.geometry.se3 import SE3


LOG = logging.getLogger(__name__)


def range_scalars(values: np.ndarray) -> np.ndarray:
    value = np.asarray(values, dtype=float)
    if not len(value):
        return np.empty((0, 3), dtype=np.uint8)
    lo, hi = np.percentile(value[np.isfinite(value)], [2, 98])
    normalized = np.clip((value - lo) / max(hi - lo, 1e-9), 0, 1)
    return np.column_stack((255 * normalized, 255 * (1 - np.abs(normalized - 0.5) * 2), 255 * (1 - normalized))).astype(np.uint8)


def confidence_scalars(values: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(values, dtype=float), 0, 1)
    return np.column_stack((255 * (1 - value), 255 * value, np.zeros_like(value))).astype(np.uint8)


class Depth3DViewerPanel(ttk.Frame):
    def __init__(self, master: tk.Misc, **kwargs: object):
        super().__init__(master, **kwargs)
        self.layers: list[DepthCloudLayer] = []
        self.camera_model = None
        self.selected_count = 0
        self.skipped_count = 0
        self._scene_status = "No selected metric-depth maps"
        self._renderer_status = ""
        self._interaction_status = ""
        self.marker_poses: list[Any] = []
        self.marker_size_m: float | None = None
        self.dense_points: list[Any] = []
        self._dirty = True
        self._renderer = None
        self._render_window = None
        self._interactor = None
        self._point_actor = None
        self._frustum_actor = None
        self._marker_actor = None
        self._dense_actor = None
        self._scene_rgb = np.empty((0, 3), dtype=np.uint8)
        self._scene_ranges = np.empty(0, dtype=np.float32)
        self._scene_confidence = np.empty(0, dtype=np.float32)
        self._reset_camera_pending = True
        self._reload_callback = None
        self._canvas: tk.Canvas | None = None
        self._canvas_photo: tk.PhotoImage | None = None
        self._canvas_image_id: int | None = None
        self._canvas_resize_after_id: str | None = None
        self._canvas_present_after_id: str | None = None
        self._canvas_full_after_id: str | None = None
        self._capture_filter = None
        self._canvas_requested_size = (640, 480)
        self._drag_mode: str | None = None
        self._drag_last: tuple[int, int] | None = None
        controls = ttk.Frame(self)
        controls.pack(fill="x", padx=6, pady=4)
        self.color_mode_var = tk.StringVar(value="RGB")
        self.confidence_var = tk.DoubleVar(value=0.0)
        self.point_size_var = tk.DoubleVar(value=2.0)
        self.maximum_points_var = tk.IntVar(value=200000)
        ttk.Button(controls, text="Refresh", command=self._request_reload).grid(row=0, column=0)
        ttk.Button(controls, text="Reset camera", command=self.reset_camera).grid(row=0, column=1)
        color_mode = ttk.Combobox(controls, textvariable=self.color_mode_var, values=["RGB", "Range", "Confidence"], state="readonly", width=22)
        color_mode.grid(row=0, column=2)
        color_mode.bind("<<ComboboxSelected>>", lambda _event: self._update_presentation())
        ttk.Label(controls, text="Confidence threshold").grid(row=1, column=0)
        ttk.Entry(controls, textvariable=self.confidence_var, width=8).grid(row=1, column=1)
        ttk.Label(controls, text="Point size").grid(row=1, column=2)
        point_size = ttk.Entry(controls, textvariable=self.point_size_var, width=8)
        point_size.grid(row=1, column=3)
        point_size.bind("<Return>", lambda _event: self._update_presentation())
        point_size.bind("<FocusOut>", lambda _event: self._update_presentation())
        ttk.Label(controls, text="Maximum scene points").grid(row=2, column=0)
        ttk.Entry(controls, textvariable=self.maximum_points_var, width=12).grid(row=2, column=1)
        self.show_frustum_var = tk.BooleanVar(value=True)
        self.show_markers_var = tk.BooleanVar(value=True)
        self.show_dense_var = tk.BooleanVar(value=False)
        overlay_controls = (
            ("Show selected camera frustums", self.show_frustum_var, self._update_presentation),
            ("Show marker geometry", self.show_markers_var, self._update_presentation),
            ("Show original dense reconstruction points", self.show_dense_var, self._on_dense_toggle),
        )
        for col, (label, var, callback) in enumerate(overlay_controls):
            ttk.Checkbutton(controls, text=label, variable=var, command=callback).grid(row=3 + col // 2, column=(col % 2) * 2, columnspan=2, sticky="w")
        self.body = ttk.Frame(self)
        self.body.pack(fill="both", expand=True)
        self.viewer_status_var = tk.StringVar(value="")
        ttk.Label(self, textvariable=self.viewer_status_var, anchor="w", wraplength=700).pack(fill="x", padx=6, pady=(0, 4))
        availability = check_vtk_availability()
        self.vtk_available = availability.available
        self._vtk_mode = "uninitialized" if self.vtk_available else "unavailable"
        if not self.vtk_available:
            self._show_body_message(availability.details)

    def set_reload_callback(self, callback: Any) -> None:
        self._reload_callback = callback

    def scene_parameters(self) -> tuple[float, int]:
        threshold = float(self.confidence_var.get())
        maximum = int(self.maximum_points_var.get())
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0 and 1.")
        if maximum <= 0:
            raise ValueError("Maximum scene points must be a positive integer.")
        return threshold, maximum

    def set_depth_clouds(
        self,
        layers: list[DepthCloudLayer],
        camera_model: Any,
        selected_count: int,
        skipped_count: int,
        marker_poses: list[Any] | None = None,
        marker_size_m: float | None = None,
        dense_points: list[Any] | None = None,
    ) -> None:
        self.layers = list(layers)
        self.camera_model = camera_model
        self.selected_count = int(selected_count)
        self.skipped_count = int(skipped_count)
        self.marker_poses = list(marker_poses or [])
        self.marker_size_m = marker_size_m
        self.dense_points = list(dense_points or [])
        points = sum(layer.point_count for layer in self.layers)
        self._scene_status = (
            f"Rendered {len(self.layers)}/{self.selected_count} selected image(s), {points:,} points; "
            f"skipped {self.skipped_count}."
        )
        self._reset_camera_pending = True
        self._dirty = True
        self._update_status()

    def set_loading_status(self, message: str) -> None:
        self._scene_status = message
        self._update_status()

    def clear_metric_depth(self) -> None:
        self.layers = []
        self.camera_model = None
        self.selected_count = 0
        self.skipped_count = 0
        self._scene_status = "No selected metric-depth maps"
        self._dirty = True
        self._update_status()
        if self._renderer is not None:
            self._clear_rendered_scene()

    def on_tab_visible(self) -> None:
        try:
            if self._dirty:
                self.refresh()
            elif self._vtk_mode == "canvas":
                self._schedule_canvas_redraw()
        except Exception as exc:  # Tk callback boundary: never terminate the GUI.
            self._disable_viewer("Depth 3D View could not be displayed", exc)

    def request_refresh(self) -> None:
        self._dirty = True

    def _request_reload(self) -> None:
        try:
            self.scene_parameters()
        except (ValueError, tk.TclError) as exc:
            self._scene_status = str(exc)
            self._update_status()
            return
        if self._reload_callback is not None:
            self._reload_callback()
        else:
            self.refresh()

    def refresh(self) -> None:
        try:
            self._refresh_scene()
        except (ValueError, tk.TclError) as exc:
            self._scene_status = str(exc)
            self._update_status()
        except Exception as exc:  # Refresh is also called directly by a Tk button.
            self._disable_viewer("Depth 3D View refresh failed", exc)

    def _refresh_scene(self) -> None:
        if not self.vtk_available:
            return
        if not self.layers or self.camera_model is None:
            self._clear_rendered_scene()
            return
        if not self._ensure_vtk():
            return
        points = np.concatenate([layer.points_world for layer in self.layers], axis=0)
        self._scene_rgb = np.concatenate([layer.rgb for layer in self.layers], axis=0)
        self._scene_ranges = np.concatenate([layer.range_m for layer in self.layers], axis=0)
        self._scene_confidence = np.concatenate([layer.confidence for layer in self.layers], axis=0)
        colors = self._current_colors()
        self._renderer.RemoveAllViewProps()
        self._point_actor = _vtk_point_actor(points, colors, self._point_size())
        self._renderer.AddActor(self._point_actor)
        self._dense_actor = None
        if self.dense_points:
            dense_xyz = np.array([[_record_value(row, "x"), _record_value(row, "y"), _record_value(row, "z")] for row in self.dense_points], dtype=np.float32)
            dense_xyz = dense_xyz[np.all(np.isfinite(dense_xyz), axis=1)]
            _threshold, maximum_points = self.scene_parameters()
            dense_xyz = dense_xyz[deterministic_decimate(len(dense_xyz), maximum_points)]
            self._dense_actor = _vtk_point_actor(dense_xyz, np.tile(np.array([[170, 170, 170]], dtype=np.uint8), (len(dense_xyz), 1)), 1.0)
            self._renderer.AddActor(self._dense_actor)
        self._frustum_actor = None
        if self.layers:
            frustum_points, frustum_edges = _camera_frustums_geometry(self.camera_model, self.layers)
            self._frustum_actor = _vtk_line_actor(frustum_points, frustum_edges, (255, 220, 40))
            self._renderer.AddActor(self._frustum_actor)
        self._marker_actor = None
        if self.marker_size_m and self.marker_poses:
            marker_points, marker_edges = _marker_line_geometry(self.marker_poses, self.marker_size_m)
            self._marker_actor = _vtk_line_actor(marker_points, marker_edges, (40, 220, 255))
            self._renderer.AddActor(self._marker_actor)
        self._set_overlay_visibility()
        if self._reset_camera_pending:
            self._renderer.ResetCamera()
            self._reset_camera_pending = False
        self._present()
        self._dirty = False

    def _clear_rendered_scene(self) -> None:
        self._point_actor = self._frustum_actor = self._marker_actor = self._dense_actor = None
        self._scene_rgb = np.empty((0, 3), dtype=np.uint8)
        self._scene_ranges = np.empty(0, dtype=np.float32)
        self._scene_confidence = np.empty(0, dtype=np.float32)
        if self._renderer is not None:
            try:
                self._renderer.RemoveAllViewProps()
                self._present()
            except Exception as exc:
                self._disable_viewer("Depth 3D View clear failed", exc)
        self._dirty = False

    def _current_colors(self) -> np.ndarray:
        mode = self.color_mode_var.get()
        if mode == "Range":
            return range_scalars(self._scene_ranges)
        if mode == "Confidence":
            return confidence_scalars(self._scene_confidence)
        return self._scene_rgb

    def _point_size(self) -> float:
        value = float(self.point_size_var.get())
        if value <= 0.0:
            raise ValueError("Point size must be positive.")
        return value

    def _update_presentation(self) -> None:
        if self._renderer is None or self._point_actor is None:
            return
        try:
            _set_vtk_actor_colors(self._point_actor, self._current_colors())
            self._point_actor.GetProperty().SetPointSize(self._point_size())
            self._set_overlay_visibility()
            self._present()
        except (ValueError, tk.TclError) as exc:
            self._scene_status = str(exc)
            self._update_status()
        except Exception as exc:
            self._disable_viewer("Depth 3D View presentation failed", exc)

    def _set_overlay_visibility(self) -> None:
        for actor, visible in (
            (self._frustum_actor, self.show_frustum_var.get()),
            (self._marker_actor, self.show_markers_var.get()),
            (self._dense_actor, self.show_dense_var.get()),
        ):
            if actor is not None:
                actor.SetVisibility(bool(visible))

    def _on_dense_toggle(self) -> None:
        if self.show_dense_var.get() and not self.dense_points:
            self._request_reload()
        else:
            self._update_presentation()

    def reset_camera(self) -> None:
        if self._renderer is not None:
            try:
                self._renderer.ResetCamera()
                self._present()
            except Exception as exc:
                self._disable_viewer("Depth 3D View camera reset failed", exc)

    def _ensure_vtk(self) -> bool:
        if self._vtk_mode in {"native", "canvas"}:
            return True
        if self._vtk_mode == "unavailable" or not self.vtk_available:
            return False
        try:
            self._initialize_native_vtk()
            self._vtk_mode = "native"
            self._renderer_status = ""
            self._update_status()
            return True
        except (tk.TclError, OSError) as exc:
            LOG.warning("Native VTK/Tk bridge is unavailable; using off-screen canvas fallback: %s", exc)
            try:
                self._initialize_canvas_vtk()
                self._vtk_mode = "canvas"
                self._renderer_status = (
                    "The installed VTK wheel lacks RenderingTk; using the embedded off-screen canvas fallback."
                )
                self._update_status()
                return True
            except Exception as fallback_exc:
                self._disable_viewer("VTK off-screen canvas fallback could not be initialized", fallback_exc)
                return False

    def _initialize_native_vtk(self) -> None:
        from vtkmodules.tk.vtkTkRenderWindowInteractor import vtkTkRenderWindowInteractor
        from vtkmodules.vtkRenderingCore import vtkRenderer

        import vtkmodules.vtkRenderingOpenGL2  # noqa: F401

        interactor = vtkTkRenderWindowInteractor(self.body)
        renderer = vtkRenderer()
        renderer.SetBackground(0.08, 0.08, 0.1)
        render_window = interactor.GetRenderWindow()
        render_window.AddRenderer(renderer)
        interactor.pack(fill="both", expand=True)
        interactor.Initialize()
        self._interactor = interactor
        self._renderer = renderer
        self._render_window = render_window

    def _initialize_canvas_vtk(self) -> None:
        from vtkmodules.vtkRenderingCore import vtkRenderer, vtkRenderWindow, vtkWindowToImageFilter

        import vtkmodules.vtkRenderingOpenGL2  # noqa: F401

        self._clear_body()
        canvas = tk.Canvas(self.body, bg="#141419", highlightthickness=0)
        canvas.pack(fill="both", expand=True)
        renderer = vtkRenderer()
        renderer.SetBackground(0.08, 0.08, 0.1)
        render_window = vtkRenderWindow()
        render_window.SetOffScreenRendering(1)
        render_window.SetMultiSamples(0)
        render_window.SetSize(*self._canvas_requested_size)
        render_window.AddRenderer(renderer)
        capture = vtkWindowToImageFilter()
        capture.SetInput(render_window)
        capture.SetInputBufferTypeToRGB()
        capture.ReadFrontBufferOff()
        capture.ShouldRerenderOff()
        self._canvas = canvas
        self._renderer = renderer
        self._render_window = render_window
        self._capture_filter = capture
        self._interactor = None
        self._bind_canvas_interactions(canvas)

    def _present(self, preview: bool = False) -> None:
        if self._render_window is None:
            return
        full_width, full_height = self._canvas_requested_size
        if self._vtk_mode == "canvas":
            size = (
                (max((full_width + 1) // 2, 1), max((full_height + 1) // 2, 1))
                if preview
                else (full_width, full_height)
            )
            self._render_window.SetSize(*size)
        self._render_window.Render()
        if self._vtk_mode == "canvas":
            self._update_canvas_frame(preview)
            self._interaction_status = "Reduced-quality interaction preview active." if preview else ""
            self._update_status()

    def _update_canvas_frame(self, preview: bool = False) -> None:
        if self._canvas is None or self._render_window is None or self._capture_filter is None:
            return
        self._capture_filter.Modified()
        self._capture_filter.Update()
        rgb = vtk_image_to_rgb(self._capture_filter.GetOutput())
        photo = tk.PhotoImage(data=_rgb_to_ppm(rgb), format="PPM")
        if preview:
            photo = photo.zoom(2, 2)
        self._canvas_photo = photo
        if self._canvas_image_id is None:
            self._canvas_image_id = self._canvas.create_image(0, 0, image=photo, anchor="nw")
        else:
            self._canvas.itemconfigure(self._canvas_image_id, image=photo)

    def _bind_canvas_interactions(self, canvas: tk.Canvas) -> None:
        canvas.bind("<Configure>", self._on_canvas_configure)
        canvas.bind("<ButtonPress-1>", lambda event: self._start_canvas_drag(event, "orbit"))
        canvas.bind("<ButtonPress-2>", lambda event: self._start_canvas_drag(event, "pan"))
        canvas.bind("<ButtonPress-3>", lambda event: self._start_canvas_drag(event, "pan"))
        canvas.bind("<B1-Motion>", self._move_canvas_drag)
        canvas.bind("<B2-Motion>", self._move_canvas_drag)
        canvas.bind("<B3-Motion>", self._move_canvas_drag)
        canvas.bind("<ButtonRelease-1>", self._end_canvas_drag)
        canvas.bind("<ButtonRelease-2>", self._end_canvas_drag)
        canvas.bind("<ButtonRelease-3>", self._end_canvas_drag)
        canvas.bind("<MouseWheel>", self._on_canvas_wheel)
        canvas.bind("<Button-4>", self._on_canvas_wheel)
        canvas.bind("<Button-5>", self._on_canvas_wheel)

    def _on_canvas_configure(self, event: tk.Event) -> None:
        self._canvas_requested_size = (max(int(event.width), 1), max(int(event.height), 1))
        if self._canvas_resize_after_id is not None:
            self.after_cancel(self._canvas_resize_after_id)
        self._canvas_resize_after_id = self.after(75, self._apply_canvas_resize)

    def _apply_canvas_resize(self) -> None:
        self._canvas_resize_after_id = None
        if self._vtk_mode != "canvas" or self._render_window is None:
            return
        try:
            self._render_window.SetSize(*self._canvas_requested_size)
            self._present()
        except Exception as exc:
            self._disable_viewer("VTK canvas resize failed", exc)

    def _schedule_canvas_redraw(self) -> None:
        if self._vtk_mode == "canvas":
            self._request_canvas_full()

    def _safe_present(self, preview: bool = False) -> None:
        try:
            self._present(preview=preview)
        except Exception as exc:
            self._disable_viewer("VTK canvas redraw failed", exc)

    def _request_canvas_preview(self) -> None:
        if self._vtk_mode != "canvas" or self._canvas_present_after_id is not None:
            return
        self._canvas_present_after_id = self.after(33, self._render_canvas_preview)

    def _render_canvas_preview(self) -> None:
        self._canvas_present_after_id = None
        self._safe_present(preview=True)

    def _request_canvas_full(self, delay_ms: int = 0) -> None:
        if self._vtk_mode != "canvas":
            return
        if delay_ms == 0 and self._canvas_present_after_id is not None:
            self.after_cancel(self._canvas_present_after_id)
            self._canvas_present_after_id = None
        if self._canvas_full_after_id is not None:
            self.after_cancel(self._canvas_full_after_id)
        self._canvas_full_after_id = self.after(delay_ms, self._render_canvas_full)

    def _render_canvas_full(self) -> None:
        self._canvas_full_after_id = None
        self._safe_present(preview=False)

    def _start_canvas_drag(self, event: tk.Event, mode: str) -> None:
        if self._canvas_full_after_id is not None:
            self.after_cancel(self._canvas_full_after_id)
            self._canvas_full_after_id = None
        self._drag_mode = mode
        self._drag_last = (int(event.x), int(event.y))

    def _move_canvas_drag(self, event: tk.Event) -> None:
        if self._drag_mode is None or self._drag_last is None or self._renderer is None:
            return
        current = (int(event.x), int(event.y))
        dx, dy = current[0] - self._drag_last[0], current[1] - self._drag_last[1]
        camera = self._renderer.GetActiveCamera()
        if self._drag_mode == "orbit":
            orbit_camera(camera, dx, dy)
        else:
            width, height = self._canvas_requested_size
            pan_camera(camera, dx, dy, width, height)
        self._renderer.ResetCameraClippingRange()
        self._drag_last = current
        self._request_canvas_preview()

    def _end_canvas_drag(self, _event: tk.Event) -> None:
        self._drag_mode = None
        self._drag_last = None
        self._request_canvas_full()

    def _on_canvas_wheel(self, event: tk.Event) -> None:
        if self._renderer is None:
            return
        number = getattr(event, "num", None)
        delta = getattr(event, "delta", 0)
        forward = number == 4 or delta > 0
        backward = number == 5 or delta < 0
        if not forward and not backward:
            return
        zoom_camera(self._renderer.GetActiveCamera(), 1.15 if forward else 1 / 1.15)
        self._renderer.ResetCameraClippingRange()
        self._request_canvas_preview()
        self._request_canvas_full(delay_ms=120)

    def _disable_viewer(self, context: str, exc: Exception) -> None:
        message = f"{context}: {exc}"
        LOG.exception(message, exc_info=exc)
        self.vtk_available = False
        self._vtk_mode = "unavailable"
        self._dirty = False
        self._renderer_status = message
        self._update_status()
        if self._canvas_resize_after_id is not None:
            try:
                self.after_cancel(self._canvas_resize_after_id)
            except tk.TclError:
                pass
            self._canvas_resize_after_id = None
        for name in ("_canvas_present_after_id", "_canvas_full_after_id"):
            after_id = getattr(self, name)
            if after_id is not None:
                try:
                    self.after_cancel(after_id)
                except tk.TclError:
                    pass
                setattr(self, name, None)
        if self._render_window is not None:
            try:
                self._render_window.Finalize()
            except Exception:
                pass
        self._renderer = None
        self._render_window = None
        self._interactor = None
        self._canvas = None
        self._capture_filter = None
        self._canvas_photo = None
        self._canvas_image_id = None
        self._show_body_message(message)

    def _update_status(self) -> None:
        parts = [part for part in (self._scene_status, self._renderer_status, self._interaction_status) if part]
        self.viewer_status_var.set(" ".join(parts))

    def _show_body_message(self, message: str) -> None:
        self._clear_body()
        ttk.Label(self.body, text=message, wraplength=600, justify="center").pack(expand=True, padx=20, pady=20)

    def _clear_body(self) -> None:
        for child in self.body.winfo_children():
            child.destroy()


def vtk_image_to_rgb(image: Any, array_converter: Any | None = None) -> np.ndarray:
    """Convert a VTK RGB image (bottom-up) to a top-down uint8 RGB array."""
    if array_converter is None:
        from vtkmodules.util.numpy_support import vtk_to_numpy

        array_converter = vtk_to_numpy
    width, height, depth = (int(value) for value in image.GetDimensions())
    if width <= 0 or height <= 0 or depth <= 0:
        raise ValueError(f"VTK capture has invalid dimensions: {(width, height, depth)}")
    scalars = image.GetPointData().GetScalars()
    if scalars is None:
        raise ValueError("VTK capture has no pixel scalars.")
    components = int(scalars.GetNumberOfComponents())
    if components < 3:
        raise ValueError(f"VTK capture requires at least 3 color components, got {components}.")
    values = np.asarray(array_converter(scalars), dtype=np.uint8).reshape(depth, height, width, components)
    return np.flipud(values[0, :, :, :3]).copy()


def orbit_camera(camera: Any, dx: float, dy: float) -> None:
    camera.Azimuth(-0.4 * float(dx))
    camera.Elevation(0.4 * float(dy))
    camera.OrthogonalizeViewUp()


def pan_camera(camera: Any, dx: float, dy: float, viewport_width: int, viewport_height: int) -> None:
    position = np.asarray(camera.GetPosition(), dtype=float)
    focal = np.asarray(camera.GetFocalPoint(), dtype=float)
    view_up = np.asarray(camera.GetViewUp(), dtype=float)
    direction = focal - position
    distance = float(np.linalg.norm(direction))
    if distance <= 1e-12:
        return
    direction /= distance
    view_up /= max(float(np.linalg.norm(view_up)), 1e-12)
    right = np.cross(direction, view_up)
    right /= max(float(np.linalg.norm(right)), 1e-12)
    height = max(int(viewport_height), 1)
    if bool(camera.GetParallelProjection()):
        world_per_pixel = 2.0 * float(camera.GetParallelScale()) / height
    else:
        world_per_pixel = 2.0 * distance * math.tan(math.radians(float(camera.GetViewAngle())) / 2.0) / height
    translation = (-float(dx) * right + float(dy) * view_up) * world_per_pixel
    camera.SetPosition(*(position + translation))
    camera.SetFocalPoint(*(focal + translation))


def zoom_camera(camera: Any, factor: float) -> None:
    factor = float(factor)
    if factor <= 0.0:
        raise ValueError("Zoom factor must be positive.")
    if bool(camera.GetParallelProjection()):
        camera.SetParallelScale(max(float(camera.GetParallelScale()) / factor, 1e-9))
    else:
        camera.Dolly(factor)


def _rgb_to_ppm(rgb: np.ndarray) -> bytes:
    image = np.ascontiguousarray(rgb, dtype=np.uint8)
    height, width = image.shape[:2]
    return f"P6 {width} {height} 255\n".encode("ascii") + image.tobytes()


def _vtk_point_actor(points: np.ndarray, colors: np.ndarray, point_size: float):
    from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
    from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper
    vtk_points = vtkPoints(); vtk_points.SetData(numpy_to_vtk(np.asarray(points, dtype=np.float32), deep=True))
    vertices = vtkCellArray()
    offsets = numpy_to_vtkIdTypeArray(np.arange(len(points) + 1, dtype=np.int64), deep=True)
    connectivity = numpy_to_vtkIdTypeArray(np.arange(len(points), dtype=np.int64), deep=True)
    vertices.SetData(offsets, connectivity)
    poly = vtkPolyData(); poly.SetPoints(vtk_points); poly.SetVerts(vertices)
    scalars = numpy_to_vtk(np.asarray(colors, dtype=np.uint8), deep=True); scalars.SetName("colors"); poly.GetPointData().SetScalars(scalars)
    mapper = vtkPolyDataMapper(); mapper.SetInputData(poly); mapper.SetColorModeToDirectScalars()
    actor = vtkActor(); actor.SetMapper(mapper); actor.GetProperty().SetPointSize(point_size)
    return actor


def _set_vtk_actor_colors(actor: Any, colors: np.ndarray) -> None:
    from vtkmodules.util.numpy_support import numpy_to_vtk

    poly = actor.GetMapper().GetInput()
    scalars = numpy_to_vtk(np.asarray(colors, dtype=np.uint8), deep=True)
    scalars.SetName("colors")
    poly.GetPointData().SetScalars(scalars)
    poly.Modified()


def _vtk_line_actor(points: np.ndarray, edges: list[tuple[int, int]], color: tuple[int, int, int]):
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
    from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper
    vtk_points = vtkPoints()
    for point in points:
        vtk_points.InsertNextPoint(*(float(v) for v in point))
    lines = vtkCellArray()
    for first, second in edges:
        lines.InsertNextCell(2); lines.InsertCellPoint(first); lines.InsertCellPoint(second)
    poly = vtkPolyData(); poly.SetPoints(vtk_points); poly.SetLines(lines)
    mapper = vtkPolyDataMapper(); mapper.SetInputData(poly)
    actor = vtkActor(); actor.SetMapper(mapper); actor.GetProperty().SetColor(*(v / 255.0 for v in color)); actor.GetProperty().SetLineWidth(2)
    return actor


def _camera_frustum_geometry(camera_model: Any, T_W_C: Any, length: float) -> tuple[np.ndarray, list[tuple[int, int]]]:
    width, height = camera_model.image_width, camera_model.image_height
    pixels = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=float)
    rays = camera_model.unproject_many(pixels)
    forward = np.isfinite(rays[:, 2]) & (rays[:, 2] > 1e-8)
    rays = rays[forward]
    pose = T_W_C if isinstance(T_W_C, SE3) else SE3.from_json_dict(T_W_C)
    corners = pose.transform_points(rays * length)
    points = np.vstack((pose.t, corners))
    edges = [(0, index + 1) for index in range(len(corners))]
    if len(corners) == 4:
        edges.extend([(1, 2), (2, 3), (3, 4), (4, 1)])
    return points, edges


def _camera_frustums_geometry(camera_model: Any, layers: list[DepthCloudLayer]) -> tuple[np.ndarray, list[tuple[int, int]]]:
    points: list[np.ndarray] = []
    edges: list[tuple[int, int]] = []
    for layer in layers:
        finite_ranges = layer.range_m[np.isfinite(layer.range_m) & (layer.range_m > 0.0)]
        scene_scale = float(np.median(finite_ranges)) if len(finite_ranges) else 1.0
        layer_points, layer_edges = _camera_frustum_geometry(
            camera_model,
            layer.T_W_C,
            max(scene_scale * 0.15, 0.05),
        )
        start = len(points)
        points.extend(layer_points)
        edges.extend((start + first, start + second) for first, second in layer_edges)
    return np.asarray(points, dtype=np.float32), edges


def _marker_line_geometry(marker_poses: list[Any], marker_size_m: float) -> tuple[np.ndarray, list[tuple[int, int]]]:
    local = marker_corners_y_up(marker_size_m)
    points: list[np.ndarray] = []
    edges: list[tuple[int, int]] = []
    for record in marker_poses:
        data = _record_value(record, "T_W_M")
        pose = data if isinstance(data, SE3) else SE3.from_json_dict(data)
        start = len(points); points.extend(pose.transform_points(local))
        edges.extend([(start, start + 1), (start + 1, start + 2), (start + 2, start + 3), (start + 3, start)])
    return np.asarray(points, dtype=np.float32), edges


def _record_value(record: Any, name: str) -> Any:
    try:
        return record[name]
    except (TypeError, KeyError, IndexError):
        return getattr(record, name)
