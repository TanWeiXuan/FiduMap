"""Optimization backend adapter.

The rest of the project should not depend on pyceres directly. This adapter
checks that pyceres is installed/configured, then solves the Python residual
function using SciPy's least-squares routine in this environment because the
available pyceres binding does not expose a usable Python custom cost callback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class AdapterResult:
    x: np.ndarray
    success: bool
    num_iterations: int
    initial_cost: float
    final_cost: float
    message: str


def require_pyceres() -> object:
    try:
        import pyceres  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("pyceres is required for BA refinement. Install/configure pyceres first.") from exc
    return pyceres


def solve_least_squares(
    residual_function: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    robust_loss_type: str,
    robust_loss_scale_px: float,
    max_num_iterations: int,
) -> AdapterResult:
    require_pyceres()
    try:
        from scipy.optimize import least_squares  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("SciPy is required by this pyceres adapter for Python residual callbacks.") from exc

    x0 = np.asarray(x0, dtype=float)
    initial_residuals = residual_function(x0)
    loss = _scipy_loss_name(robust_loss_type)
    result = least_squares(
        residual_function,
        x0,
        loss=loss,
        f_scale=float(robust_loss_scale_px),
        max_nfev=max_num_iterations,
        x_scale="jac",
    )
    final_residuals = residual_function(result.x)
    return AdapterResult(
        x=np.asarray(result.x, dtype=float),
        success=bool(result.success),
        num_iterations=int(result.nfev),
        initial_cost=float(0.5 * np.dot(initial_residuals, initial_residuals)),
        final_cost=float(0.5 * np.dot(final_residuals, final_residuals)),
        message=str(result.message),
    )


def _scipy_loss_name(robust_loss_type: str) -> str:
    normalized = robust_loss_type.strip().lower()
    if normalized == "none":
        return "linear"
    if normalized == "huber":
        return "huber"
    if normalized == "cauchy":
        return "cauchy"
    raise ValueError(f"Unsupported robust loss type: {robust_loss_type}")
