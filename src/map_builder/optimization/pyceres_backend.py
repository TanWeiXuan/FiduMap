"""Real pyceres backend for marker-map BA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np

from map_builder.geometry import SE3, marker_corners_for_detector_order
from map_builder.project import BAConfig

from .ba_costs import MarkerObservationReprojectionCost
from .ba_problem import BAProblem


@dataclass(frozen=True)
class PyCeresSolveResult:
    success: bool
    camera_poses: dict[int, SE3]
    marker_poses: dict[int, SE3]
    initial_cost: float | None
    final_cost: float | None
    num_iterations: int | None
    solver_report: str
    backend_name: str = "pyceres"


class PyCeresBASolver:
    def __init__(self) -> None:
        self.pyceres = require_pyceres()
        self.camera_params: dict[int, np.ndarray] = {}
        self.marker_params: dict[int, np.ndarray] = {}
        self.costs: list[object] = []
        self.losses: list[object] = []

    def solve(
        self,
        problem_data: BAProblem,
        config: BAConfig,
        excluded_observations: set[tuple[int, int]] | None = None,
    ) -> PyCeresSolveResult:
        pyceres = self.pyceres
        excluded = excluded_observations or set()
        problem = pyceres.Problem()
        self._initialize_parameter_blocks(problem_data)
        object_points = marker_corners_for_detector_order(problem_data.marker_size_m)

        for obs in problem_data.observations:
            if (obs.image_id, obs.marker_id) in excluded:
                continue
            if obs.image_id not in self.camera_params or obs.marker_id not in self.marker_params:
                continue
            cost = MarkerObservationReprojectionCost(
                camera_model=problem_data.camera_model,
                marker_size_m=problem_data.marker_size_m,
                observed_corners_px=obs.corners_px,
                detector_order_object_points=object_points,
                finite_diff_step=config.finite_diff_step,
                invalid_projection_penalty_px=config.invalid_projection_penalty_px,
            )
            loss = make_loss(pyceres, config.robust_loss_type, config.robust_loss_scale_px)
            problem.add_residual_block(
                cost,
                loss,
                [self.camera_params[obs.image_id], self.marker_params[obs.marker_id]],
            )
            self.costs.append(cost)
            if loss is not None:
                self.losses.append(loss)

        if problem.num_residual_blocks() == 0:
            raise RuntimeError("No residual blocks were added to the pyceres problem.")
        set_parameter_block_constant(problem, self.marker_params[problem_data.anchor_marker_id])

        options = make_solver_options(pyceres, config)
        summary = pyceres.SolverSummary()
        pyceres.solve(options, problem, summary)

        camera_poses = {image_id: SE3.exp(params) for image_id, params in self.camera_params.items()}
        marker_poses = {marker_id: SE3.exp(params) for marker_id, params in self.marker_params.items()}
        marker_poses[problem_data.anchor_marker_id] = SE3.identity()
        return PyCeresSolveResult(
            success=bool(_summary_solution_usable(summary)),
            camera_poses=camera_poses,
            marker_poses=marker_poses,
            initial_cost=_summary_float(summary, "initial_cost"),
            final_cost=_summary_float(summary, "final_cost"),
            num_iterations=_summary_int(summary, "num_successful_steps"),
            solver_report=brief_report(summary),
        )

    def _initialize_parameter_blocks(self, problem_data: BAProblem) -> None:
        self.camera_params = {
            image_id: np.ascontiguousarray(problem_data.initial_camera_poses[image_id].log(), dtype=np.float64)
            for image_id in problem_data.camera_ids
        }
        self.marker_params = {}
        for marker_id in problem_data.marker_ids:
            if marker_id == problem_data.anchor_marker_id:
                self.marker_params[marker_id] = np.zeros(6, dtype=np.float64)
            else:
                self.marker_params[marker_id] = np.ascontiguousarray(
                    problem_data.initial_marker_poses[marker_id].log(),
                    dtype=np.float64,
                )


def solve_marker_ba(
    problem_data: BAProblem,
    config: BAConfig,
    excluded_observations: set[tuple[int, int]] | None = None,
) -> PyCeresSolveResult:
    return PyCeresBASolver().solve(problem_data, config, excluded_observations)


def require_pyceres() -> Any:
    try:
        import pyceres  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "pyceres backend requested but pyceres is not installed. Install pyceres or select backend='scipy'."
        ) from exc
    return pyceres


def make_loss(pyceres: Any, loss_type: str, scale: float) -> object | None:
    normalized = loss_type.strip().lower()
    if normalized == "none":
        return None
    if normalized == "huber":
        return pyceres.HuberLoss(float(scale))
    if normalized == "cauchy":
        return pyceres.CauchyLoss(float(scale))
    raise ValueError(f"Unsupported robust loss type: {loss_type}")


def set_parameter_block_constant(problem: object, arr: np.ndarray) -> None:
    if hasattr(problem, "set_parameter_block_constant"):
        problem.set_parameter_block_constant(arr)  # type: ignore[attr-defined]
        return
    if hasattr(problem, "SetParameterBlockConstant"):
        problem.SetParameterBlockConstant(arr)  # type: ignore[attr-defined]
        return
    raise RuntimeError("Installed pyceres does not expose parameter-block constant API.")


def make_solver_options(pyceres: Any, config: BAConfig) -> object:
    options = pyceres.SolverOptions()
    options.max_num_iterations = int(config.max_num_iterations)
    options.minimizer_progress_to_stdout = bool(config.minimizer_progress_to_stdout)
    if hasattr(options, "num_threads"):
        options.num_threads = int(config.num_threads)
    for attr, value in [
        ("function_tolerance", config.function_tolerance),
        ("gradient_tolerance", config.gradient_tolerance),
        ("parameter_tolerance", config.parameter_tolerance),
    ]:
        if hasattr(options, attr):
            setattr(options, attr, float(value))
    set_linear_solver_type(pyceres, options, config.linear_solver_type)
    return options


def set_linear_solver_type(pyceres: Any, options: object, linear_solver_type: str) -> None:
    if not hasattr(options, "linear_solver_type"):
        return

    enum = getattr(pyceres, "LinearSolverType", None)
    solver_map = {
        "SPARSE_NORMAL_CHOLESKY": getattr(pyceres, "SPARSE_NORMAL_CHOLESKY", None)
        or (getattr(enum, "SPARSE_NORMAL_CHOLESKY", None) if enum is not None else None),
        "DENSE_QR": getattr(pyceres, "DENSE_QR", None)
        or (getattr(enum, "DENSE_QR", None) if enum is not None else None),
        "DENSE_NORMAL_CHOLESKY": getattr(pyceres, "DENSE_NORMAL_CHOLESKY", None)
        or (getattr(enum, "DENSE_NORMAL_CHOLESKY", None) if enum is not None else None),
        "SPARSE_SCHUR": getattr(pyceres, "SPARSE_SCHUR", None)
        or (getattr(enum, "SPARSE_SCHUR", None) if enum is not None else None),
        "DENSE_SCHUR": getattr(pyceres, "DENSE_SCHUR", None)
        or (getattr(enum, "DENSE_SCHUR", None) if enum is not None else None),
        "ITERATIVE_SCHUR": getattr(pyceres, "ITERATIVE_SCHUR", None)
        or (getattr(enum, "ITERATIVE_SCHUR", None) if enum is not None else None),
    }

    solver_value = solver_map.get(linear_solver_type.strip().upper())
    if solver_value is not None:
        setattr(options, "linear_solver_type", solver_value)
        return

    warnings.warn(f"pyceres linear solver type '{linear_solver_type}' is unavailable; using default.", RuntimeWarning)


def brief_report(summary: object) -> str:
    if hasattr(summary, "BriefReport"):
        return str(summary.BriefReport())
    if hasattr(summary, "brief_report"):
        return str(summary.brief_report())
    return str(summary)


def _summary_solution_usable(summary: object) -> bool:
    if hasattr(summary, "IsSolutionUsable"):
        return bool(summary.IsSolutionUsable())
    return True


def _summary_float(summary: object, attr: str) -> float | None:
    return None if not hasattr(summary, attr) else float(getattr(summary, attr))


def _summary_int(summary: object, attr: str) -> int | None:
    return None if not hasattr(summary, attr) else int(getattr(summary, attr))
