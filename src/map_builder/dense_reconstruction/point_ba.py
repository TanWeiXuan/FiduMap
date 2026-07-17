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
    if config.mode != "points_only":
        raise NotImplementedError(
            f"Dense BA stage supports only points_only structure optimization; got {config.mode!r}. "
            "Camera poses remain fixed to the marker-map BA result."
        )
    try:
        pyceres = importlib.import_module("pyceres")
    except ImportError as exc:
        raise RuntimeError("Dense point BA unavailable: pyceres not installed.") from exc

    run_id = store.create_dense_ba_run(config.mode, backend_name="pyceres")
    try:
        points = store.list_active_dense_points()
        if any(str(point["source"]) == "merged" for point in points):
            raise RuntimeError(
                "Dense BA stage cannot run on an active merged point set. Run dense BA before "
                "duplicate merging; retriangulation restores track-backed points."
            )
        observations = store.list_track_observations()
        obs_by_track: dict[int, list[Any]] = {}
        for obs in observations:
            obs_by_track.setdefault(obs.track_id, []).append(obs)

        problem = pyceres.Problem()
        costs: list[Any] = []
        losses: list[Any] = []
        costs_by_point: dict[int, list[Any]] = {}
        point_params: dict[int, np.ndarray] = {}
        track_id_by_point: dict[int, int] = {}
        skipped_points: list[tuple[int, int | None]] = []
        residual_count = 0
        for point in points:
            point_id = int(point["id"])
            if point["track_id"] is None:
                skipped_points.append((point_id, None))
                continue
            valid_observations = [
                obs
                for obs in obs_by_track.get(int(point["track_id"]), [])
                if obs.image_id in poses_by_image
            ]
            if len(valid_observations) < max(int(config.min_observations), 2):
                skipped_points.append((point_id, int(point["track_id"])))
                continue

            params = np.ascontiguousarray([point["x"], point["y"], point["z"]], dtype=np.float64)
            point_costs: list[Any] = []
            for obs in valid_observations:
                cost = _PointReprojectionCost(
                    pyceres,
                    camera_model,
                    poses_by_image[obs.image_id],
                    np.array([obs.x, obs.y], dtype=np.float64),
                    finite_diff_step=config.finite_diff_step,
                )
                loss = pyceres.HuberLoss(float(config.huber_scale_px))
                problem.add_residual_block(cost, loss, [params])
                point_costs.append(cost)
                costs.append(cost)
                losses.append(loss)
                residual_count += 1
            point_params[point_id] = params
            track_id_by_point[point_id] = int(point["track_id"])
            costs_by_point[point_id] = point_costs

        if residual_count == 0:
            raise RuntimeError("No dense point observations are available for BA.")

        options = pyceres.SolverOptions()
        options.max_num_iterations = int(config.max_num_iterations)
        if hasattr(options, "minimizer_progress_to_stdout"):
            options.minimizer_progress_to_stdout = False
        summary = pyceres.SolverSummary()
        pyceres.solve(options, problem, summary)
        if not _summary_solution_usable(summary):
            raise RuntimeError(
                "Dense BA stage received an unusable solver result; stored point coordinates were not updated."
            )

        point_results: dict[int, tuple[np.ndarray, float, float, bool]] = {}
        all_error_norms: list[float] = []
        for point_id, params in point_params.items():
            norms = np.array(
                [
                    float(np.linalg.norm(cost.compute_residual(params)))
                    for cost in costs_by_point[point_id]
                ],
                dtype=float,
            )
            finite = norms[np.isfinite(norms)]
            mean_err = float(np.mean(finite)) if len(finite) == len(norms) and len(finite) else np.inf
            max_err = float(np.max(finite)) if len(finite) == len(norms) and len(finite) else np.inf
            accepted = bool(
                np.all(np.isfinite(params))
                and mean_err <= float(config.max_mean_reprojection_error_px)
                and max_err <= float(config.max_reprojection_error_px)
            )
            point_results[point_id] = (params.copy(), mean_err, max_err, accepted)
            all_error_norms.extend(finite.tolist())

        # Persist only after Ceres reports a usable solution and all post-solve
        # point diagnostics have been computed. This keeps a failed run from
        # leaving a partially updated cloud.
        with store.conn:
            for point_id, (params, mean_err, max_err, accepted) in point_results.items():
                stored_mean = None if not np.isfinite(mean_err) else mean_err
                stored_max = None if not np.isfinite(max_err) else max_err
                status = "active" if accepted else "rejected"
                store.conn.execute(
                    """
                    UPDATE dense_points
                    SET x=?, y=?, z=?, source=?, mean_reprojection_error_px=?,
                        max_reprojection_error_px=?, is_active=?
                    WHERE id=?
                    """,
                    (
                        float(params[0]),
                        float(params[1]),
                        float(params[2]),
                        "dense_ba" if accepted else "dense_ba_rejected",
                        stored_mean,
                        stored_max,
                        1 if accepted else 0,
                        point_id,
                    ),
                )
                store.conn.execute(
                    """
                    UPDATE tracks
                    SET x=?, y=?, z=?, mean_reprojection_error_px=?,
                        max_reprojection_error_px=?, status=?
                    WHERE id=?
                    """,
                    (
                        float(params[0]),
                        float(params[1]),
                        float(params[2]),
                        stored_mean,
                        stored_max,
                        status,
                        track_id_by_point[point_id],
                    ),
                )
            for point_id, track_id in skipped_points:
                store.conn.execute(
                    "UPDATE dense_points SET source='dense_ba_rejected', is_active=0 WHERE id=?",
                    (point_id,),
                )
                if track_id is not None:
                    store.conn.execute("UPDATE tracks SET status='rejected' WHERE id=?", (track_id,))

        error_norms = np.asarray(all_error_norms, dtype=float)
        mean_err = None if not len(error_norms) else float(error_norms.mean())
        max_err = None if not len(error_norms) else float(error_norms.max())
        accepted_count = sum(1 for _params, _mean, _max, accepted in point_results.values() if accepted)
        rejected_count = len(point_results) - accepted_count + len(skipped_points)
        store.complete_dense_ba_run(
            run_id,
            True,
            initial_cost=_summary_float(summary, "initial_cost"),
            final_cost=_summary_float(summary, "final_cost"),
            mean_reprojection_error_px=mean_err,
            max_reprojection_error_px=max_err,
            num_points=accepted_count,
            num_observations=residual_count,
        )
        return DenseStageSummary(
            stage="dense_ba",
            total=len(points),
            success=accepted_count,
            failed=rejected_count,
            details=(
                f"Dense BA complete ({config.mode}); kept {accepted_count}/{len(points)} point(s)"
                + (f", rejected {rejected_count} by observation/reprojection checks" if rejected_count else "")
            ),
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

            def compute_residual(self, point_w: np.ndarray) -> np.ndarray:
                X = np.asarray(point_w, dtype=np.float64).reshape(3)
                point_c = self.R.T @ (X - self.C)
                if np.linalg.norm(point_c) <= 1e-12:
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
