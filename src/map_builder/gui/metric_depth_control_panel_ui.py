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
    checks = [
        ("Include optimized dense tracks", panel.include_dense_var),
        ("Include visible marker surfaces", panel.include_markers_var),
        ("Allow model download", panel.allow_download_var),
        ("Recompute existing successful maps", panel.recompute_var),
    ]
    for row, (label, variable) in enumerate(checks, 5):
        ttk.Checkbutton(settings, text=label, variable=variable).grid(row=row, column=0, columnspan=2, sticky="w", padx=6)
    ttk.Label(settings, text="Scope").grid(row=9, column=0, sticky="w", padx=6)
    scope = ttk.Frame(settings); scope.grid(row=9, column=1, sticky="w")
    ttk.Radiobutton(scope, text="selected image", variable=panel.scope_var, value="selected").pack(side="left")
    ttk.Radiobutton(scope, text="all optimized images", variable=panel.scope_var, value="all").pack(side="left")

    actions = ttk.LabelFrame(frame, text="Actions")
    actions.grid(row=5, column=0, sticky="ew", padx=6, pady=4)
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

    ttk.Label(frame, textvariable=panel.summary_var, wraplength=340, justify="left").grid(row=6, column=0, sticky="ew", padx=6, pady=4)
    log_frame = ttk.LabelFrame(frame, text="Recent log")
    log_frame.grid(row=7, column=0, sticky="nsew", padx=6, pady=4)
    log_frame.columnconfigure(0, weight=1)
    panel.log_text = tk.Text(log_frame, height=7, wrap="word", state="disabled")
    panel.log_text.grid(row=0, column=0, sticky="nsew")
    return buttons
