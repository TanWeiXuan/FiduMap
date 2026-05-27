from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from map_builder.optimization.residual_math import finite_difference_jacobian, write_jacobian

from .dense_store import DenseReconstructionStore
from .models import DenseBAConfig, DenseStageSummary


def run_dense_point_ba(
    store: DenseReconstructionStore,
    poses_by_image: dict[int, dict[str, Any]],
    camera_model: Any,
    config: DenseBAConfig,
) -> DenseStageSummary:
    if config.mode == "full":
        raise NotImplementedError("Dense BA full mode is not implemented; use points_only.")
    if config.mode not in {"points_only", "points_and_cameras"}:
        raise ValueError(f"Unsupported dense BA mode: {config.mode}")
    try:
        pyceres = importlib.import_module("pyceres")
    except ImportError as exc:
        raise RuntimeError("Dense point BA unavailable: pyceres not installed.") from exc

    run_id = store.create_dense_ba_run(config.mode, backend_name="pyceres")
    try:
        points = store.list_active_dense_points()
        observations = store.list_track_observations()
        obs_by_track: dict[int, list[Any]] = {}
        for obs in observations:
            obs_by_track.setdefault(obs.track_id, []).append(obs)

        problem = pyceres.Problem()
        costs: list[Any] = []
        losses: list[Any] = []
        point_params: dict[int, np.ndarray] = {}
        residual_count = 0
        for point in points:
            if point["track_id"] is None:
                continue
            point_id = int(point["id"])
            params = np.ascontiguousarray([point["x"], point["y"], point["z"]], dtype=np.float64)
            point_params[point_id] = params
            for obs in obs_by_track.get(int(point["track_id"]), []):
                if obs.image_id not in poses_by_image:
                    continue
                cost = _PointReprojectionCost(
                    pyceres,
                    camera_model,
                    poses_by_image[obs.image_id],
                    np.array([obs.x, obs.y], dtype=np.float64),
                    finite_diff_step=config.finite_diff_step,
                )
                loss = pyceres.HuberLoss(float(config.huber_scale_px))
                problem.add_residual_block(cost, loss, [params])
                costs.append(cost)
                losses.append(loss)
                residual_count += 1
        if residual_count == 0:
            raise RuntimeError("No dense point observations are available for BA.")

        options = pyceres.SolverOptions()
        options.max_num_iterations = int(config.max_num_iterations)
        if hasattr(options, "minimizer_progress_to_stdout"):
            options.minimizer_progress_to_stdout = False
        summary = pyceres.SolverSummary()
        pyceres.solve(options, problem, summary)

        errors: list[float] = []
        for point_id, params in point_params.items():
            store.update_dense_point_coordinates(point_id, params, source="dense_ba")
        for cost in costs:
            # Evaluate after solve for user-facing reprojection statistics.
            params = cost.parameter_block
            errors.extend(np.abs(cost.compute_residual(params)).tolist())
        error_norms = np.linalg.norm(np.reshape(errors, (-1, 2)), axis=1) if errors else np.array([], dtype=float)
        mean_err = None if not len(error_norms) else float(error_norms.mean())
        max_err = None if not len(error_norms) else float(error_norms.max())
        store.complete_dense_ba_run(
            run_id,
            _summary_solution_usable(summary),
            initial_cost=_summary_float(summary, "initial_cost"),
            final_cost=_summary_float(summary, "final_cost"),
            mean_reprojection_error_px=mean_err,
            max_reprojection_error_px=max_err,
            num_points=len(point_params),
            num_observations=residual_count,
        )
        return DenseStageSummary(
            stage="dense_ba",
            total=len(point_params),
            success=len(point_params),
            details=f"Dense BA complete ({config.mode}); optimized {len(point_params)} point(s)",
        )
    except Exception as exc:
        store.complete_dense_ba_run(run_id, False, error_message=str(exc))
        raise


class _PointReprojectionCost:
    def __new__(cls, pyceres: Any, *args: Any, **kwargs: Any) -> Any:
        base = pyceres.CostFunction

        class Cost(base):  # type: ignore[misc, valid-type]
            def __init__(
                self,
                camera_model: Any,
                T_W_C: dict[str, Any],
                observed_px: np.ndarray,
                finite_diff_step: float,
            ):
                base.__init__(self)
                self.set_num_residuals(2)
                self.set_parameter_block_sizes([3])
                self.camera_model = camera_model
                self.R = np.asarray(T_W_C["R"], dtype=np.float64)
                self.C = np.asarray(T_W_C["t"], dtype=np.float64).reshape(3)
                self.observed_px = np.asarray(observed_px, dtype=np.float64).reshape(2)
                self.finite_diff_step = float(finite_diff_step)
                self.parameter_block: np.ndarray | None = None

            def compute_residual(self, point_w: np.ndarray) -> np.ndarray:
                X = np.asarray(point_w, dtype=np.float64).reshape(3)
                self.parameter_block = X
                point_c = self.R.T @ (X - self.C)
                if point_c[2] <= 1e-12 or np.linalg.norm(point_c) <= 1e-12:
                    return np.full(2, 1e6, dtype=np.float64)
                projected = np.asarray(self.camera_model.project(point_c), dtype=np.float64).reshape(2)
                if not np.all(np.isfinite(projected)):
                    return np.full(2, 1e6, dtype=np.float64)
                return self.observed_px - projected

            def Evaluate(self, parameters: object, residuals: object, jacobians: object) -> bool:
                point_w = np.asarray(parameters[0], dtype=np.float64)  # type: ignore[index]
                r = self.compute_residual(point_w)
                np.asarray(residuals)[:] = r
                if jacobians is not None and jacobians[0] is not None:  # type: ignore[index]
                    J = finite_difference_jacobian(self.compute_residual, point_w, self.finite_diff_step)
                    write_jacobian(jacobians[0], J)  # type: ignore[index]
                return True

        return Cost(*args, **kwargs)


def _summary_solution_usable(summary: object) -> bool:
    if hasattr(summary, "IsSolutionUsable"):
        return bool(summary.IsSolutionUsable())
    return True


def _summary_float(summary: object, attr: str) -> float | None:
    return None if not hasattr(summary, attr) else float(getattr(summary, attr))
