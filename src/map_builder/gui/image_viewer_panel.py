"""Simple image preview canvas."""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import ttk
from typing import Any

from map_builder.project.models import MarkerDetection


class ImageViewerPanel(ttk.Frame):
    MIN_ZOOM = 0.05
    MAX_ZOOM = 20.0
    ZOOM_STEP = 1.15

    def __init__(self, master: tk.Misc, **kwargs: object):
        super().__init__(master, **kwargs)
        controls = ttk.Frame(self)
        controls.grid(row=0, column=0, sticky="ew", padx=6, pady=4)
        controls.columnconfigure(2, weight=1)
        self.show_markers_var = tk.BooleanVar(value=True)
        self.show_xfeat_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(controls, text="Show markers", variable=self.show_markers_var, command=self._render).grid(
            row=0, column=0, sticky="w", padx=(0, 8)
        )
        ttk.Checkbutton(controls, text="Show XFeat keypoints", variable=self.show_xfeat_var, command=self._render).grid(
            row=0, column=1, sticky="w"
        )
        self.canvas = tk.Canvas(self, bg="#f2f2f2", highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self._photo: tk.PhotoImage | None = None
        self._current_path: Path | None = None
        self._current_detections: list[MarkerDetection] = []
        self._xfeat_keypoints = None
        self._source_bgr: Any | None = None
        self._fit_scale = 1.0
        self._zoom = 1.0
        self._offset_x = 0.0
        self._offset_y = 0.0
        self._drag_start: tuple[int, int] | None = None
        self._drag_origin: tuple[float, float] | None = None

        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind("<ButtonPress-1>", self._on_pan_start)
        self.canvas.bind("<B1-Motion>", self._on_pan_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_pan_end)
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Button-4>", self._on_mouse_wheel)
        self.canvas.bind("<Button-5>", self._on_mouse_wheel)
        self.canvas.bind("<Double-Button-1>", lambda _event: self.reset_view())
        self.show_placeholder("No image selected")

    def show_placeholder(self, text: str) -> None:
        self._photo = None
        self._source_bgr = None
        self._current_path = None
        self._current_detections = []
        self.canvas.delete("all")
        self.canvas.create_text(
            max(self.canvas.winfo_width() // 2, 220),
            max(self.canvas.winfo_height() // 2, 160),
            text=text,
            fill="#555555",
        )

    def show_image(self, path: Path, detections: list[MarkerDetection] | None = None) -> None:
        try:
            source_bgr = _load_image_bgr(path)
        except RuntimeError as exc:
            self.show_placeholder(str(exc))
            return
        self._current_path = path
        self._source_bgr = source_bgr
        self._current_detections = detections or []
        self.reset_view()

    def set_xfeat_keypoints(self, keypoints: Any | None) -> None:
        self._xfeat_keypoints = keypoints
        self._render()

    def reset_view(self) -> None:
        if self._source_bgr is None:
            return
        self._zoom = 1.0
        self._fit_scale = self._compute_fit_scale()
        display_width, display_height = self._display_size()
        self._offset_x = (self.canvas.winfo_width() - display_width) / 2
        self._offset_y = (self.canvas.winfo_height() - display_height) / 2
        self._render()

    def _on_canvas_configure(self, _event: tk.Event) -> None:
        if self._source_bgr is None:
            if self._current_path is None:
                self.show_placeholder("No image selected")
            return
        old_scale = self._scale()
        old_center = self._canvas_center_image_point(old_scale)
        self._fit_scale = self._compute_fit_scale()
        new_scale = self._scale()
        self._offset_x = self.canvas.winfo_width() / 2 - old_center[0] * new_scale
        self._offset_y = self.canvas.winfo_height() / 2 - old_center[1] * new_scale
        self._render()

    def _on_pan_start(self, event: tk.Event) -> None:
        if self._source_bgr is None:
            return
        self._drag_start = (event.x, event.y)
        self._drag_origin = (self._offset_x, self._offset_y)
        self.canvas.configure(cursor="fleur")

    def _on_pan_move(self, event: tk.Event) -> None:
        if self._source_bgr is None or self._drag_start is None or self._drag_origin is None:
            return
        dx = event.x - self._drag_start[0]
        dy = event.y - self._drag_start[1]
        self._offset_x = self._drag_origin[0] + dx
        self._offset_y = self._drag_origin[1] + dy
        self._render()

    def _on_pan_end(self, _event: tk.Event) -> None:
        self._drag_start = None
        self._drag_origin = None
        self.canvas.configure(cursor="")

    def _on_mouse_wheel(self, event: tk.Event) -> None:
        if self._source_bgr is None:
            return
        direction = _wheel_direction(event)
        if direction == 0:
            return

        old_scale = self._scale()
        image_x = (event.x - self._offset_x) / old_scale
        image_y = (event.y - self._offset_y) / old_scale

        factor = self.ZOOM_STEP if direction > 0 else 1 / self.ZOOM_STEP
        self._zoom = min(max(self._zoom * factor, self.MIN_ZOOM), self.MAX_ZOOM)
        new_scale = self._scale()
        self._offset_x = event.x - image_x * new_scale
        self._offset_y = event.y - image_y * new_scale
        self._render()

    def _render(self) -> None:
        if self._source_bgr is None:
            return
        self._clamp_offsets()
        scale = self._scale()
        photo = _render_photo(self._source_bgr, scale, self._current_detections, self._xfeat_keypoints, bool(self.show_markers_var.get()), bool(self.show_xfeat_var.get()))
        self._photo = photo
        self.canvas.delete("all")
        self.canvas.create_image(round(self._offset_x), round(self._offset_y), image=photo, anchor="nw")

    def _compute_fit_scale(self) -> float:
        if self._source_bgr is None:
            return 1.0
        height, width = self._source_bgr.shape[:2]
        canvas_width = max(self.canvas.winfo_width(), 1)
        canvas_height = max(self.canvas.winfo_height(), 1)
        return max(min(canvas_width / width, canvas_height / height, 1.0), 0.001)

    def _scale(self) -> float:
        return max(self._fit_scale * self._zoom, 0.001)

    def _display_size(self) -> tuple[int, int]:
        if self._source_bgr is None:
            return 1, 1
        height, width = self._source_bgr.shape[:2]
        scale = self._scale()
        return max(int(round(width * scale)), 1), max(int(round(height * scale)), 1)

    def _canvas_center_image_point(self, scale: float) -> tuple[float, float]:
        center_x = self.canvas.winfo_width() / 2
        center_y = self.canvas.winfo_height() / 2
        return (center_x - self._offset_x) / scale, (center_y - self._offset_y) / scale

    def _clamp_offsets(self) -> None:
        display_width, display_height = self._display_size()
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if display_width <= canvas_width:
            self._offset_x = (canvas_width - display_width) / 2
        else:
            self._offset_x = min(0.0, max(self._offset_x, canvas_width - display_width))

        if display_height <= canvas_height:
            self._offset_y = (canvas_height - display_height) / 2
        else:
            self._offset_y = min(0.0, max(self._offset_y, canvas_height - display_height))


def _wheel_direction(event: tk.Event) -> int:
    if getattr(event, "num", None) == 4:
        return 1
    if getattr(event, "num", None) == 5:
        return -1
    delta = getattr(event, "delta", 0)
    if delta > 0:
        return 1
    if delta < 0:
        return -1
    return 0


def _load_image_bgr(path: Path) -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for image preview.") from exc

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not load image: {path.name}")
    return image


def _render_photo(image_bgr: Any, scale: float, detections: list[MarkerDetection], xfeat_keypoints: Any | None = None, show_markers: bool = True, show_xfeat: bool = False) -> tk.PhotoImage:
    import cv2  # type: ignore[import-not-found]

    scale = max(scale, 0.001)
    image = image_bgr.copy()
    height, width = image.shape[:2]
    new_size = (max(int(width * scale), 1), max(int(height * scale), 1))
    resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    if show_markers and detections:
        _draw_detections(cv2, resized, detections, scale)
    if show_xfeat and xfeat_keypoints is not None:
        _draw_xfeat(cv2, resized, xfeat_keypoints, scale)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    ppm = _rgb_to_ppm(rgb)
    return tk.PhotoImage(data=ppm, format="PPM")


def _draw_detections(cv2: Any, image: Any, detections: list[MarkerDetection], scale: float) -> None:
    for detection in detections:
        points = []
        for u, v in detection.corners:
            points.append([int(round(u * scale)), int(round(v * scale))])
        import numpy as np  # type: ignore[import-not-found]

        pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        line_width = max(int(round(2 * max(scale, 1.0))), 2)
        cv2.polylines(image, [pts], True, (0, 180, 255), line_width)
        if points:
            cv2.putText(
                image,
                str(detection.marker_id),
                tuple(points[0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.45 * max(scale, 1.0), 0.45),
                (0, 180, 255),
                line_width,
                cv2.LINE_AA,
            )


def _rgb_to_ppm(rgb: Any) -> bytes:
    height, width = rgb.shape[:2]
    header = f"P6 {width} {height} 255\n".encode("ascii")
    return header + rgb.tobytes()


def _draw_xfeat(cv2: Any, image: Any, keypoints: Any, scale: float) -> None:
    import numpy as np
    pts=np.asarray(keypoints)
    if pts.size==0:return
    if len(pts)>5000:
        idx=np.linspace(0,len(pts)-1,5000,dtype=int); pts=pts[idx]
    for u,v in pts[:, :2]:
        cv2.circle(image,(int(round(u*scale)),int(round(v*scale))),1,(20,220,20),-1)
