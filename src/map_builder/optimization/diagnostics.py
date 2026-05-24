"""Formatting helpers for BA diagnostics."""

from __future__ import annotations

from map_builder.project import BARunSummary


def format_ba_summary(summary: BARunSummary | None) -> str:
    if summary is None:
        return "BA not run"
    if not summary.success:
        return f"BA failed: {summary.error_message or 'unknown error'}"
    mean = "n/a" if summary.mean_reprojection_error_px is None else f"{summary.mean_reprojection_error_px:.3f} px"
    median = "n/a" if summary.median_reprojection_error_px is None else f"{summary.median_reprojection_error_px:.3f} px"
    max_error = "n/a" if summary.max_reprojection_error_px is None else f"{summary.max_reprojection_error_px:.3f} px"
    return (
        f"BA run {summary.id}: success\n"
        f"Iterations: {summary.num_iterations}\n"
        f"Observations: {summary.num_observations}\n"
        f"Corners: {summary.num_corners}\n"
        f"Mean error: {mean}\n"
        f"Median error: {median}\n"
        f"Max error: {max_error}\n"
        f"Outlier observations: {summary.num_outlier_observations}"
    )
