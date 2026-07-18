from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog
from typing import Callable

from map_builder.metric_depth.availability import check_inference_availability
from map_builder.metric_depth.models import BACKEND_DAV2, BACKEND_PROMPT_DA, MetricDepthConfig, MetricDepthProgress

from .metric_depth_control_panel_ui import build_metric_depth_controls
from .scrollable_frame import ScrollableFrame


class MetricDepthControlPanel(ScrollableFrame):
    BACKEND_LABELS = {
        "Depth Anything V2 Small — aligned": BACKEND_DAV2,
        "Prompt Depth Anything Small": BACKEND_PROMPT_DA,
    }
    DEFAULT_MODELS = {
        BACKEND_DAV2: "depth-anything/Depth-Anything-V2-Small-hf",
        BACKEND_PROMPT_DA: "depth-anything/prompt-depth-anything-vits-hf",
    }

    def __init__(self, master: tk.Misc, generate_selected: Callable[[], None], generate_all: Callable[[], None], cancel: Callable[[], None], export_selected: Callable[[], None], export_run: Callable[[], None], **kwargs: object):
        super().__init__(master, **kwargs)
        self.project_ready = False
        self.selected_ready = False
        self.export_selected_ready = False
        self.export_run_ready = False
        self.running = False
        self.current_run_id: int | None = None
        self.status_var = tk.StringVar(value="Choose an image folder first")
        self.stage_var = tk.StringVar(value="")
        self.summary_var = tk.StringVar(value="")
        self.backend_var = tk.StringVar(value=next(iter(self.BACKEND_LABELS)))
        self.model_var = tk.StringVar(value=self.DEFAULT_MODELS[BACKEND_DAV2])
        self.device_var = tk.StringVar(value="auto")
        self.inference_size_var = tk.StringVar(value="518")
        self.include_dense_var = tk.BooleanVar(value=True)
        self.include_markers_var = tk.BooleanVar(value=True)
        self.allow_download_var = tk.BooleanVar(value=False)
        self.recompute_var = tk.BooleanVar(value=False)
        self.scope_var = tk.StringVar(value="selected")
        self.buttons = build_metric_depth_controls(self, {
            "browse": self._browse_model, "selected": generate_selected, "all": generate_all, "cancel": cancel,
            "export_selected": export_selected, "export_run": export_run,
        })
        self.backend_var.trace_add("write", self._backend_changed)
        self.refresh_availability()

    @property
    def backend(self) -> str:
        return self.BACKEND_LABELS[self.backend_var.get()]

    def config(self) -> MetricDepthConfig:
        try:
            size = int(self.inference_size_var.get())
        except ValueError as exc:
            raise ValueError("Inference size must be a positive integer.") from exc
        config = MetricDepthConfig(
            backend=self.backend, model_id_or_path=self.model_var.get().strip(), device=self.device_var.get(),
            allow_download=bool(self.allow_download_var.get()), inference_size=size,
            include_dense_track_points=bool(self.include_dense_var.get()), include_marker_surfaces=bool(self.include_markers_var.get()),
            recompute=bool(self.recompute_var.get()),
        )
        config.validate()
        return config

    def set_prerequisites(self, project_ready: bool, selected_ready: bool, export_selected_ready: bool = False, export_run_ready: bool = False) -> None:
        self.project_ready, self.selected_ready = project_ready, selected_ready
        self.export_selected_ready, self.export_run_ready = export_selected_ready, export_run_ready
        self._update_buttons()

    def set_running(self, running: bool) -> None:
        self.running = running
        if not running:
            self.progress["value"] = 0
        self._update_buttons()

    def set_progress(self, event: MetricDepthProgress) -> None:
        self.stage_var.set(event.message)
        if event.fraction is not None:
            self.progress["value"] = event.fraction * 100.0
        self.append_log(event.message)

    def set_status(self, text: str) -> None:
        self.status_var.set(text)
        self.append_log(text)

    def set_summary(self, counts: dict[str, object] | None, run_id: int | None = None) -> None:
        counts = counts or {}
        self.current_run_id = run_id
        self.summary_var.set(
            f"Images eligible: {int(counts.get('eligible', 0))}  Completed: {int(counts.get('completed', 0))}  Failed: {int(counts.get('failed', 0))}\n"
            f"Mean prompts: {float(counts.get('mean_prompt_count', 0)):.1f}  Prompt coverage: {100 * float(counts.get('mean_prompt_coverage', 0)):.1f}%\n"
            f"Valid depth: {100 * float(counts.get('mean_valid_fraction', 0)):.1f}%  Median anchor error: {_metric(counts.get('median_anchor_error_m'))}\n"
            f"Mean time/image: {float(counts.get('mean_processing_seconds', 0)):.2f}s  Stale maps: {int(counts.get('stale', 0))}\n"
            f"Current backend: {self.backend}  Run ID: {run_id if run_id is not None else 'none'}"
        )

    def append_log(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text.rstrip() + "\n")
        lines = int(self.log_text.index("end-1c").split(".")[0])
        if lines > 120:
            self.log_text.delete("1.0", f"{lines - 100}.0")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def refresh_availability(self) -> None:
        availability = check_inference_availability()
        self.available = availability.available
        if not availability.available:
            self.status_var.set(availability.details)
        self._update_buttons()

    def _update_buttons(self) -> None:
        generation = self.available and self.project_ready and not self.running
        self.buttons["selected"].configure(state="normal" if generation and self.selected_ready else "disabled")
        self.buttons["all"].configure(state="normal" if generation else "disabled")
        self.buttons["cancel"].configure(state="normal" if self.running else "disabled")
        self.buttons["export_selected"].configure(state="normal" if self.export_selected_ready and not self.running else "disabled")
        self.buttons["export_run"].configure(state="normal" if self.export_run_ready and not self.running else "disabled")

    def _backend_changed(self, *_args: object) -> None:
        current = self.model_var.get()
        if current in self.DEFAULT_MODELS.values():
            self.model_var.set(self.DEFAULT_MODELS[self.backend])
        self._update_buttons()

    def _browse_model(self) -> None:
        folder = filedialog.askdirectory(title="Select local Hugging Face model directory")
        if folder:
            self.model_var.set(str(Path(folder).resolve()))


def _metric(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.3f} m"
