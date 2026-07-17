from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Callable


def build_dense_controls(
    panel: Any,
    callbacks: dict[str, Callable[[], None]],
) -> tuple[dict[str, ttk.Button], ttk.Progressbar]:
    frame = panel.inner
    frame.columnconfigure(0, weight=1)
    ttk.Label(
        frame,
        text="Dense Reconstruction",
        font=("TkDefaultFont", 10, "bold"),
    ).grid(row=0, column=0, sticky="w", padx=6, pady=(6, 2))
    ttk.Label(
        frame,
        textvariable=panel.status_var,
        wraplength=340,
        justify="left",
    ).grid(row=1, column=0, sticky="ew", padx=6)
    progress = ttk.Progressbar(frame, mode="indeterminate")
    progress.grid(row=2, column=0, sticky="ew", padx=6, pady=2)
    ttk.Label(
        frame,
        textvariable=panel.counts_var,
        wraplength=340,
        justify="left",
    ).grid(row=3, column=0, sticky="ew", padx=6, pady=(0, 6))

    auto = ttk.LabelFrame(frame, text="Automatic workflow")
    auto.grid(row=4, column=0, sticky="ew", padx=6, pady=4)
    auto.columnconfigure(0, weight=1)
    run_all = ttk.Button(
        auto,
        text="Run Full Dense Reconstruction",
        command=panel._start_auto,
    )
    run_all.grid(row=0, column=0, sticky="ew", padx=6, pady=(6, 2))
    ttk.Label(
        auto,
        text="Extract → pair → match → filter → triangulate → BA → merge",
    ).grid(row=1, column=0, sticky="w", padx=6, pady=(0, 6))

    settings = ttk.LabelFrame(frame, text="Main quality settings")
    settings.grid(row=5, column=0, sticky="ew", padx=6, pady=4)
    settings.columnconfigure(1, weight=1)
    fields = [
        ("Max keypoints / image", panel.max_keypoints_var),
        ("Device", panel.device_var),
        ("Max pairs / image", panel.max_pairs_var),
        ("Min common markers", panel.min_common_markers_var),
        ("Min descriptor score", panel.min_match_score_var),
        ("Max reprojection error (px)", panel.max_reprojection_error_var),
        ("Duplicate radius (m)", panel.duplicate_radius_var),
    ]
    for row, (label, variable) in enumerate(fields):
        ttk.Label(settings, text=label).grid(
            row=row,
            column=0,
            sticky="w",
            padx=(6, 8),
        )
        if label == "Device":
            widget: tk.Widget = ttk.Combobox(
                settings,
                textvariable=variable,
                values=["auto", "cpu", "cuda"],
                state="readonly",
            )
        else:
            widget = ttk.Entry(settings, textvariable=variable)
        widget.grid(row=row, column=1, sticky="ew", padx=(0, 6))
    ttk.Checkbutton(
        settings,
        text="Recompute stored features",
        variable=panel.recompute_features_var,
    ).grid(row=len(fields), column=0, columnspan=2, sticky="w", padx=6)
    ttk.Label(
        settings,
        text="Advanced geometry thresholds use conservative defaults.",
    ).grid(row=len(fields) + 1, column=0, columnspan=2, sticky="w", padx=6)

    actions = ttk.LabelFrame(frame, text="Manual stages")
    actions.grid(row=6, column=0, sticky="ew", padx=6, pady=4)
    actions.columnconfigure(0, weight=1)
    stage_specs = [
        ("extract", "1. Extract XFeat Features", panel.extraction_config),
        ("pairs", "2. Select Frame Pairs", panel.pair_selection_config),
        ("match", "3. Match Selected Pairs", panel.matching_config),
        ("filter", "4. Filter Matches Geometrically", panel.epipolar_config),
        ("tracks", "5. Build Tracks + Triangulate", panel.triangulation_config),
        ("ba", "6. Refine Dense Points", panel.dense_ba_config),
        ("merge", "7. Merge Duplicate Points", panel.duplicate_config),
    ]
    buttons = {"all": run_all}
    for row, (stage, label, config_factory) in enumerate(stage_specs):
        button = ttk.Button(
            actions,
            text=label,
            command=panel._guard(label, config_factory, callbacks[stage]),
        )
        button.grid(row=row, column=0, sticky="ew", padx=6, pady=2)
        buttons[stage] = button
    export = ttk.Button(
        actions,
        text="Export Active Point Cloud CSV",
        command=callbacks["export"],
    )
    export.grid(row=len(stage_specs), column=0, sticky="ew", padx=6, pady=2)
    buttons["export"] = export
    return buttons, progress
