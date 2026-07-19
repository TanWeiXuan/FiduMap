from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Callable


def build_metric_depth_controls(panel: Any, callbacks: dict[str, Callable[[], None]]) -> dict[str, ttk.Button]:
    frame = panel.inner
    frame.columnconfigure(0, weight=1)
    ttk.Label(frame, text="Metric Depth Maps", font=("TkDefaultFont", 10, "bold")).grid(row=0, column=0, sticky="w", padx=6, pady=(6, 2))
    ttk.Label(frame, textvariable=panel.status_var, wraplength=340, justify="left").grid(row=1, column=0, sticky="ew", padx=6)
    panel.progress = ttk.Progressbar(frame, mode="determinate", maximum=100)
    panel.progress.grid(row=2, column=0, sticky="ew", padx=6, pady=2)
    ttk.Label(frame, textvariable=panel.stage_var, wraplength=340, justify="left").grid(row=3, column=0, sticky="ew", padx=6)

    settings = ttk.LabelFrame(frame, text="Model and alignment settings")
    settings.grid(row=4, column=0, sticky="ew", padx=6, pady=4)
    settings.columnconfigure(1, weight=1)
    ttk.Label(settings, text="Model").grid(row=0, column=0, sticky="w", padx=6)
    ttk.Label(settings, text=panel.BACKEND_LABEL).grid(row=0, column=1, sticky="w", padx=6)
    ttk.Label(settings, text="Model ID or local path").grid(row=1, column=0, sticky="w", padx=6)
    ttk.Entry(settings, textvariable=panel.model_var).grid(row=1, column=1, sticky="ew", padx=6)
    ttk.Button(settings, text="Browse local model directory", command=callbacks["browse"]).grid(row=2, column=0, columnspan=2, sticky="ew", padx=6, pady=2)
    ttk.Label(settings, text="Device").grid(row=3, column=0, sticky="w", padx=6)
    ttk.Combobox(settings, textvariable=panel.device_var, values=["auto", "cpu", "cuda"], state="readonly").grid(row=3, column=1, sticky="ew", padx=6)
    ttk.Label(settings, text="Inference size").grid(row=4, column=0, sticky="w", padx=6)
    ttk.Entry(settings, textvariable=panel.inference_size_var).grid(row=4, column=1, sticky="ew", padx=6)
    ttk.Label(settings, text="Alignment method").grid(row=5, column=0, sticky="w", padx=6)
    panel.alignment_mode_combobox = ttk.Combobox(
        settings, textvariable=panel.alignment_mode_var, values=list(panel.ALIGNMENT_LABELS), state="readonly"
    )
    panel.alignment_mode_combobox.grid(row=5, column=1, sticky="ew", padx=6)
    checks = [
        ("Include optimized dense tracks", panel.include_dense_var),
        ("Include visible marker surfaces", panel.include_markers_var),
        ("Allow model download", panel.allow_download_var),
        ("Recompute existing successful maps", panel.recompute_var),
    ]
    for row, (label, variable) in enumerate(checks, 6):
        ttk.Checkbutton(settings, text=label, variable=variable).grid(row=row, column=0, columnspan=2, sticky="w", padx=6)
    ttk.Label(settings, text="Scope").grid(row=10, column=0, sticky="w", padx=6)
    scope = ttk.Frame(settings); scope.grid(row=10, column=1, sticky="w")
    ttk.Radiobutton(scope, text="selected image", variable=panel.scope_var, value="selected").pack(side="left")
    ttk.Radiobutton(scope, text="all optimized images", variable=panel.scope_var, value="all").pack(side="left")

    advanced = ttk.LabelFrame(frame, text="Advanced alignment")
    advanced.grid(row=5, column=0, sticky="ew", padx=6, pady=4)
    advanced.columnconfigure(1, weight=1)
    advanced_fields = [
        ("Spline knot count", panel.spline_knots_var),
        ("Spatial grid columns", panel.spatial_grid_columns_var),
        ("Spatial grid rows", panel.spatial_grid_rows_var),
        ("Maximum log-depth correction", panel.maximum_log_correction_var),
        ("Holdout fraction", panel.holdout_fraction_var),
    ]
    for row, (label, variable) in enumerate(advanced_fields):
        ttk.Label(advanced, text=label).grid(row=row, column=0, sticky="w", padx=6)
        ttk.Entry(advanced, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=6)

    actions = ttk.LabelFrame(frame, text="Actions")
    actions.grid(row=6, column=0, sticky="ew", padx=6, pady=4)
    actions.columnconfigure(0, weight=1)
    specs = [
        ("selected", "Generate Selected Image"), ("all", "Generate All Images"), ("cancel", "Cancel"),
        ("export_selected", "Export Selected Map"), ("export_run", "Export Current Run"),
    ]
    buttons: dict[str, ttk.Button] = {}
    for row, (key, label) in enumerate(specs):
        button = ttk.Button(actions, text=label, command=callbacks[key])
        button.grid(row=row, column=0, sticky="ew", padx=6, pady=2)
        buttons[key] = button

    ttk.Label(frame, textvariable=panel.summary_var, wraplength=340, justify="left").grid(row=7, column=0, sticky="ew", padx=6, pady=4)
    log_frame = ttk.LabelFrame(frame, text="Recent log")
    log_frame.grid(row=8, column=0, sticky="nsew", padx=6, pady=4)
    log_frame.columnconfigure(0, weight=1)
    panel.log_text = tk.Text(log_frame, height=7, wrap="word", state="disabled")
    panel.log_text.grid(row=0, column=0, sticky="nsew")
    return buttons
