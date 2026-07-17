from __future__ import annotations

from typing import Any

import numpy as np

from .models import TriangulationConfig


def triangulate_two_view(C1: np.ndarray, d1: np.ndarray, C2: np.ndarray, d2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    C1 = np.asarray(C1, dtype=float).reshape(3)
    C2 = np.asarray(C2, dtype=float).reshape(3)
    d1 = _normalize_rows(np.asarray(d1, dtype=float))
    d2 = _normalize_rows(np.asarray(d2, dtype=float))
    w0 = C1 - C2
    a = np.sum(d1 * d1, axis=1)
    b = np.sum(d1 * d2, axis=1)
    c = np.sum(d2 * d2, axis=1)
    d = np.sum(d1 * w0, axis=1)
    e = np.sum(d2 * w0, axis=1)
    den = np.maximum(a * c - b * b, 1e-12)
    s = (b * e - c * d) / den
    t = (a * e - b * d) / den
    p1 = C1 + s[:, None] * d1
    p2 = C2 + t[:, None] * d2
    return 0.5 * (p1 + p2), np.linalg.norm(p1 - p2, axis=1)


def triangulate_pair_matches(
    pixels1: np.ndarray,
    pixels2: np.ndarray,
    T_W_C1: dict[str, Any],
    T_W_C2: dict[str, Any],
    camera_model: Any,
    config: TriangulationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    R1, C1 = _pose_parts(T_W_C1)
    R2, C2 = _pose_parts(T_W_C2)
    f1 = camera_model.unproject_many(np.asarray(pixels1, dtype=float))
    f2 = camera_model.unproject_many(np.asarray(pixels2, dtype=float))
    d1 = _normalize_rows((R1 @ f1.T).T)
    d2 = _normalize_rows((R2 @ f2.T).T)
    X, gaps = triangulate_two_view(C1, d1, C2, d2)
    angles = ray_angles_deg(d1, d2)
    depths1 = np.sum((X - C1) * d1, axis=1)
    depths2 = np.sum((X - C2) * d2, axis=1)
    err1 = reprojection_errors(X, np.asarray(pixels1, dtype=float), T_W_C1, camera_model)
    err2 = reprojection_errors(X, np.asarray(pixels2, dtype=float), T_W_C2, camera_model)
    ranges = np.maximum(np.linalg.norm(X - C1, axis=1), np.linalg.norm(X - C2, axis=1))
    valid = (
        np.all(np.isfinite(X), axis=1)
        & np.isfinite(gaps)
        & np.isfinite(err1)
        & np.isfinite(err2)
        & (depths1 > 0.0)
        & (depths2 > 0.0)
        & (angles >= config.min_triangulation_angle_deg)
        & (err1 <= config.max_reprojection_error_px)
        & (err2 <= config.max_reprojection_error_px)
        & (gaps <= config.max_ray_gap_m)
        & (ranges <= config.max_depth_m)
    )
    return X, valid


def triangulate_multiview(
    observations: list[tuple[int, int, float, float]],
    poses_by_image: dict[int, dict[str, Any]],
    camera_model: Any,
    config: TriangulationConfig,
) -> tuple[np.ndarray | None, dict[str, float]]:
    X, metrics, _inlier_mask = triangulate_multiview_robust(
        observations,
        poses_by_image,
        camera_model,
        config,
    )
    return X, metrics


def triangulate_multiview_robust(
    observations: list[tuple[int, int, float, float]],
    poses_by_image: dict[int, dict[str, Any]],
    camera_model: Any,
    config: TriangulationConfig,
) -> tuple[np.ndarray | None, dict[str, float], np.ndarray]:
    empty_mask = np.zeros((len(observations),), dtype=bool)
    if len(observations) < config.min_observations:
        return None, {}, empty_mask
    image_ids = [int(obs[0]) for obs in observations]
    if len(set(image_ids)) != len(image_ids):
        return None, {}, empty_mask

    centers: list[np.ndarray] = []
    directions: list[np.ndarray] = []
    rotations: list[np.ndarray] = []
    pixels: list[list[float]] = []
    for image_id, _feature_idx, x, y in observations:
        if image_id not in poses_by_image:
            return None, {}, empty_mask
        R, C = _pose_parts(poses_by_image[image_id])
        ray = np.asarray(camera_model.unproject(np.array([x, y], dtype=float)), dtype=float).reshape(3)
        if not np.all(np.isfinite(ray)) or np.linalg.norm(ray) <= 1e-12:
            return None, {}, empty_mask
        centers.append(C)
        directions.append(R @ ray)
        rotations.append(R)
        pixels.append([float(x), float(y)])

    C_arr = np.asarray(centers, dtype=float)
    d_arr = _normalize_rows(np.asarray(directions, dtype=float))
    R_arr = np.asarray(rotations, dtype=float)
    pixel_arr = np.asarray(pixels, dtype=float)

    seed = _select_seed_point(C_arr, d_arr, R_arr, pixel_arr, camera_model, config)
    if seed is None:
        return None, {}, empty_mask
    X = seed
    active = np.ones((len(observations),), dtype=bool)
    max_prunes = max(int(config.max_outlier_iterations), 0)
    pruned = 0

    # Use the geometrically verified two-view seed to identify gross track
    # outliers before fitting all rays. Then iteratively refit and remove only
    # the single worst observation when the track still has redundancy.
    for _iteration in range(max_prunes + 1):
        active_indices = np.flatnonzero(active)
        if len(active_indices) < config.min_observations:
            return None, {}, empty_mask

        seed_errors, seed_valid = _evaluate_point(
            X,
            active_indices,
            C_arr,
            d_arr,
            R_arr,
            pixel_arr,
            camera_model,
            config,
        )
        gross = (~seed_valid) | (seed_errors > float(config.max_reprojection_error_px))
        if np.any(gross) and len(active_indices) > config.min_observations and pruned < max_prunes:
            local_worst = int(np.argmax(np.where(gross, seed_errors, -1.0)))
            active[active_indices[local_worst]] = False
            pruned += 1
            continue

        solved = _solve_multiview_point(C_arr[active], d_arr[active])
        if solved is None:
            return None, {}, empty_mask
        X = solved
        active_indices = np.flatnonzero(active)
        errors, valid = _evaluate_point(
            X,
            active_indices,
            C_arr,
            d_arr,
            R_arr,
            pixel_arr,
            camera_model,
            config,
        )
        finite_errors = errors[valid]
        if len(finite_errors) == len(active_indices):
            mean_err = float(np.mean(finite_errors))
            max_err = float(np.max(finite_errors))
            active_dirs = d_arr[active]
            max_angle = max_pairwise_ray_angle_deg(active_dirs)
            if (
                max_angle >= config.min_triangulation_angle_deg
                and mean_err <= config.max_mean_reprojection_error_px
                and max_err <= config.max_reprojection_error_px
            ):
                min_angle = min_pairwise_ray_angle_deg(active_dirs)
                return X, {
                    "mean_reprojection_error_px": mean_err,
                    "max_reprojection_error_px": max_err,
                    "min_triangulation_angle_deg": float(min_angle),
                    "max_triangulation_angle_deg": float(max_angle),
                    "num_inlier_observations": float(len(active_indices)),
                    "num_rejected_observations": float(len(observations) - len(active_indices)),
                }, active.copy()

        if len(active_indices) <= config.min_observations or pruned >= max_prunes:
            return None, {}, empty_mask
        ranked_errors = np.where(valid, errors, np.inf)
        worst_local = int(np.argmax(ranked_errors))
        active[active_indices[worst_local]] = False
        pruned += 1

    return None, {}, empty_mask


def _select_seed_point(
    centers: np.ndarray,
    directions: np.ndarray,
    rotations: np.ndarray,
    pixels: np.ndarray,
    camera_model: Any,
    config: TriangulationConfig,
) -> np.ndarray | None:
    best_key: tuple[int, float, float] | None = None
    best_point: np.ndarray | None = None
    threshold = float(config.max_reprojection_error_px)
    for i in range(len(directions)):
        for j in range(i + 1, len(directions)):
            angle = float(ray_angles_deg(directions[i : i + 1], directions[j : j + 1])[0])
            if angle < config.min_triangulation_angle_deg:
                continue
            candidate, gap = triangulate_two_view(
                centers[i],
                directions[i : i + 1],
                centers[j],
                directions[j : j + 1],
            )
            X = candidate[0]
            if not np.all(np.isfinite(X)) or not np.isfinite(gap[0]) or gap[0] > config.max_ray_gap_m:
                continue
            indices = np.arange(len(directions), dtype=int)
            errors, valid = _evaluate_point(
                X,
                indices,
                centers,
                directions,
                rotations,
                pixels,
                camera_model,
                config,
            )
            inliers = valid & (errors <= threshold)
            inlier_count = int(np.count_nonzero(inliers))
            if inlier_count < config.min_observations:
                continue
            clipped = np.minimum(np.where(valid, errors, threshold * 2.0), threshold * 2.0)
            truncated_cost = float(np.sum(clipped * clipped))
            key = (inlier_count, -truncated_cost, angle)
            if best_key is None or key > best_key:
                best_key = key
                best_point = X
    return best_point


def _solve_multiview_point(centers: np.ndarray, directions: np.ndarray) -> np.ndarray | None:
    if len(centers) < 2:
        return None
    A = np.zeros((3, 3), dtype=float)
    b = np.zeros(3, dtype=float)
    eye = np.eye(3)
    for C, d in zip(centers, directions):
        P = eye - np.outer(d, d)
        A += P
        b += P @ C
    condition = float(np.linalg.cond(A))
    if not np.isfinite(condition) or condition > 1e10:
        return None
    X = np.linalg.solve(A, b)
    return X if np.all(np.isfinite(X)) else None


def _evaluate_point(
    X: np.ndarray,
    indices: np.ndarray,
    centers: np.ndarray,
    directions: np.ndarray,
    rotations: np.ndarray,
    pixels: np.ndarray,
    camera_model: Any,
    config: TriangulationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    errors = np.full((len(indices),), np.inf, dtype=float)
    valid = np.zeros((len(indices),), dtype=bool)
    for local_index, observation_index in enumerate(indices):
        C = centers[observation_index]
        d = directions[observation_index]
        delta = X - C
        depth = float(np.dot(delta, d))
        distance = float(np.linalg.norm(delta))
        if depth <= 0.0 or not np.isfinite(distance) or distance > config.max_depth_m:
            continue
        point_c = rotations[observation_index].T @ delta
        projected = np.asarray(camera_model.project(point_c), dtype=float).reshape(2)
        if not np.all(np.isfinite(projected)):
            continue
        error = float(np.linalg.norm(projected - pixels[observation_index]))
        if not np.isfinite(error):
            continue
        errors[local_index] = error
        valid[local_index] = True
    return errors, valid


def reprojection_errors(
    points_w: np.ndarray, pixels: np.ndarray, T_W_C: dict[str, Any], camera_model: Any
) -> np.ndarray:
    R, C = _pose_parts(T_W_C)
    X = np.asarray(points_w, dtype=float)
    rays_c = (R.T @ (X - C).T).T
    projected = camera_model.project_many(rays_c)
    return np.linalg.norm(projected - np.asarray(pixels, dtype=float), axis=1)


def ray_angles_deg(d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    dot = np.clip(np.sum(_normalize_rows(d1) * _normalize_rows(d2), axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def min_pairwise_ray_angle_deg(dirs: np.ndarray) -> float:
    d = _normalize_rows(np.asarray(dirs, dtype=float))
    best = 180.0
    for i in range(len(d)):
        for j in range(i + 1, len(d)):
            angle = float(np.degrees(np.arccos(np.clip(np.dot(d[i], d[j]), -1.0, 1.0))))
            best = min(best, angle)
    return best


def max_pairwise_ray_angle_deg(dirs: np.ndarray) -> float:
    d = _normalize_rows(np.asarray(dirs, dtype=float))
    best = 0.0
    for i in range(len(d)):
        for j in range(i + 1, len(d)):
            angle = float(np.degrees(np.arccos(np.clip(np.dot(d[i], d[j]), -1.0, 1.0))))
            best = max(best, angle)
    return best


def _pose_parts(T_W_C: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(T_W_C["R"], dtype=float), np.asarray(T_W_C["t"], dtype=float).reshape(3)


def _normalize_rows(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
