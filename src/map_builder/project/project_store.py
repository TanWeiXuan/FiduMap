"""SQLite-backed project store for the fiducial map builder."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any

from .models import (
    BAConfig,
    BARunSummary,
    DetectorRunConfig,
    ImageRecord,
    MarkerDetection,
    OptimizedCameraPose,
    OptimizedMarkerPose,
    PnPObservation,
    ReprojectionErrorRecord,
    SeedCameraPose,
    SeedMarkerPose,
)


PROJECT_DIR_NAME = ".map_builder"
PROJECT_DB_NAME = "project.sqlite"


class ProjectStore:
    """Persistent project/index state for one image folder."""

    def __init__(self, folder: Path, connection: sqlite3.Connection):
        self.folder = folder
        self.db_path = folder / PROJECT_DIR_NAME / PROJECT_DB_NAME
        self._conn = connection
        self._conn.row_factory = sqlite3.Row

    @classmethod
    def open(cls, folder: Path) -> "ProjectStore":
        project_folder = Path(folder).expanduser().resolve()
        if not project_folder.exists() or not project_folder.is_dir():
            raise FileNotFoundError(f"Image folder does not exist: {project_folder}")
        sidecar = project_folder / PROJECT_DIR_NAME
        sidecar.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(sidecar / PROJECT_DB_NAME)
        store = cls(project_folder, conn)
        store._initialize_schema()
        return store

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "ProjectStore":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _initialize_schema(self) -> None:
        with self._conn:
            self._conn.executescript(
                """
                PRAGMA foreign_keys = ON;

                CREATE TABLE IF NOT EXISTS project_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT
                );

                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY,
                    rel_path TEXT UNIQUE NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    width INTEGER,
                    height INTEGER,
                    ignored INTEGER NOT NULL DEFAULT 0,
                    missing INTEGER NOT NULL DEFAULT 0,
                    detection_status TEXT NOT NULL DEFAULT 'not_run',
                    modified_since_detection INTEGER NOT NULL DEFAULT 0,
                    indexed_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS detector_runs (
                    id INTEGER PRIMARY KEY,
                    started_at TEXT NOT NULL,
                    detector_type TEXT NOT NULL,
                    dictionary_name TEXT NOT NULL,
                    detector_params_json TEXT NOT NULL,
                    opencv_version TEXT
                );

                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY,
                    image_id INTEGER NOT NULL,
                    detector_run_id INTEGER NOT NULL,
                    marker_family TEXT NOT NULL,
                    dictionary_name TEXT NOT NULL,
                    marker_id INTEGER NOT NULL,
                    corners_json TEXT NOT NULL,
                    corner_refinement_method TEXT NOT NULL,
                    FOREIGN KEY(image_id) REFERENCES images(id),
                    FOREIGN KEY(detector_run_id) REFERENCES detector_runs(id)
                );

                CREATE INDEX IF NOT EXISTS idx_images_rel_path ON images(rel_path);
                CREATE INDEX IF NOT EXISTS idx_images_ignored ON images(ignored);
                CREATE INDEX IF NOT EXISTS idx_images_detection_status ON images(detection_status);
                CREATE INDEX IF NOT EXISTS idx_detections_image_id ON detections(image_id);
                CREATE INDEX IF NOT EXISTS idx_detections_marker_id ON detections(marker_id);

                CREATE TABLE IF NOT EXISTS pnp_observations (
                    id INTEGER PRIMARY KEY,
                    image_id INTEGER NOT NULL,
                    detection_id INTEGER NOT NULL,
                    marker_id INTEGER NOT NULL,
                    success INTEGER NOT NULL,
                    rvec_json TEXT,
                    tvec_json TEXT,
                    T_C_M_json TEXT,
                    reprojection_error_px REAL,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(image_id) REFERENCES images(id),
                    FOREIGN KEY(detection_id) REFERENCES detections(id)
                );

                CREATE TABLE IF NOT EXISTS seed_camera_poses (
                    image_id INTEGER PRIMARY KEY,
                    T_W_C_json TEXT NOT NULL,
                    source_marker_id INTEGER,
                    reprojection_error_px REAL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(image_id) REFERENCES images(id)
                );

                CREATE TABLE IF NOT EXISTS seed_marker_poses (
                    marker_id INTEGER PRIMARY KEY,
                    T_W_M_json TEXT NOT NULL,
                    source_image_id INTEGER,
                    reprojection_error_px REAL,
                    is_anchor INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS graph_diagnostics (
                    key TEXT PRIMARY KEY,
                    value TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_pnp_image_id ON pnp_observations(image_id);
                CREATE INDEX IF NOT EXISTS idx_pnp_marker_id ON pnp_observations(marker_id);
                CREATE INDEX IF NOT EXISTS idx_pnp_success ON pnp_observations(success);

                CREATE TABLE IF NOT EXISTS ba_runs (
                    id INTEGER PRIMARY KEY,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    success INTEGER NOT NULL,
                    backend_name TEXT,
                    num_iterations INTEGER,
                    initial_cost REAL,
                    final_cost REAL,
                    mean_reprojection_error_px REAL,
                    median_reprojection_error_px REAL,
                    max_reprojection_error_px REAL,
                    num_observations INTEGER NOT NULL DEFAULT 0,
                    num_corners INTEGER NOT NULL DEFAULT 0,
                    num_outlier_observations INTEGER NOT NULL DEFAULT 0,
                    robust_loss_type TEXT,
                    robust_loss_scale_px REAL,
                    corner_outlier_threshold_px REAL,
                    marker_observation_outlier_threshold_px REAL,
                    marker_outlier_threshold_px REAL,
                    solver_report TEXT,
                    error_message TEXT
                );

                CREATE TABLE IF NOT EXISTS optimized_camera_poses (
                    image_id INTEGER NOT NULL,
                    ba_run_id INTEGER NOT NULL,
                    T_W_C_json TEXT NOT NULL,
                    PRIMARY KEY(image_id, ba_run_id),
                    FOREIGN KEY(image_id) REFERENCES images(id),
                    FOREIGN KEY(ba_run_id) REFERENCES ba_runs(id)
                );

                CREATE TABLE IF NOT EXISTS optimized_marker_poses (
                    marker_id INTEGER NOT NULL,
                    ba_run_id INTEGER NOT NULL,
                    T_W_M_json TEXT NOT NULL,
                    is_anchor INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(marker_id, ba_run_id),
                    FOREIGN KEY(ba_run_id) REFERENCES ba_runs(id)
                );

                CREATE TABLE IF NOT EXISTS reprojection_errors (
                    id INTEGER PRIMARY KEY,
                    ba_run_id INTEGER NOT NULL,
                    image_id INTEGER NOT NULL,
                    marker_id INTEGER NOT NULL,
                    corner_index_detector_order INTEGER NOT NULL,
                    error_px REAL NOT NULL,
                    residual_x_px REAL NOT NULL,
                    residual_y_px REAL NOT NULL,
                    is_outlier INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY(ba_run_id) REFERENCES ba_runs(id),
                    FOREIGN KEY(image_id) REFERENCES images(id)
                );

                CREATE INDEX IF NOT EXISTS idx_optimized_marker_ba ON optimized_marker_poses(ba_run_id);
                CREATE INDEX IF NOT EXISTS idx_optimized_camera_ba ON optimized_camera_poses(ba_run_id);
                CREATE INDEX IF NOT EXISTS idx_reprojection_errors_ba ON reprojection_errors(ba_run_id);
                """
            )
            self._ensure_column("ba_runs", "backend_name", "TEXT")
            self._ensure_column("ba_runs", "marker_observation_outlier_threshold_px", "REAL")
            self._ensure_column("ba_runs", "solver_report", "TEXT")

    def _ensure_column(self, table_name: str, column_name: str, column_type: str) -> None:
        rows = self._conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        if column_name not in {str(row["name"]) for row in rows}:
            self._conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")

    @property
    def connection(self) -> sqlite3.Connection:
        return self._conn

    def set_camera_config_path(self, path: Path) -> None:
        value = _path_to_stored_string(self.folder, Path(path))
        self.set_metadata("camera_config_path", value)

    def get_camera_config_path(self) -> Path | None:
        value = self.get_metadata("camera_config_path")
        if not value:
            return None
        path = Path(value)
        if path.is_absolute():
            return path
        return self.folder / path

    def set_marker_size_m(self, marker_size_m: float) -> None:
        size = float(marker_size_m)
        if size <= 0.0:
            raise ValueError("Marker side length must be positive.")
        self.set_metadata("marker_size_m", repr(size))

    def get_marker_size_m(self) -> float | None:
        value = self.get_metadata("marker_size_m")
        return None if value is None else float(value)

    def set_ba_config(self, config: BAConfig) -> None:
        self.set_metadata(
            "ba_config",
            json.dumps(
                {
                    "robust_loss_type": config.robust_loss_type,
                    "robust_loss_scale_px": config.robust_loss_scale_px,
                    "corner_outlier_threshold_px": config.corner_outlier_threshold_px,
                    "marker_observation_outlier_threshold_px": config.marker_observation_outlier_threshold_px,
                    "run_outlier_second_pass": config.run_outlier_second_pass,
                    "max_num_iterations": config.max_num_iterations,
                    "backend_name": config.backend_name,
                    "finite_diff_step": config.finite_diff_step,
                    "invalid_projection_penalty_px": config.invalid_projection_penalty_px,
                    "function_tolerance": config.function_tolerance,
                    "gradient_tolerance": config.gradient_tolerance,
                    "parameter_tolerance": config.parameter_tolerance,
                    "num_threads": config.num_threads,
                    "linear_solver_type": config.linear_solver_type,
                    "minimizer_progress_to_stdout": config.minimizer_progress_to_stdout,
                },
                sort_keys=True,
            ),
        )

    def get_ba_config(self) -> BAConfig:
        value = self.get_metadata("ba_config")
        if value is None:
            return BAConfig()
        data = json.loads(value)
        if "marker_outlier_threshold_px" in data and "marker_observation_outlier_threshold_px" not in data:
            data["marker_observation_outlier_threshold_px"] = data.pop("marker_outlier_threshold_px")
        return BAConfig(**data)

    def set_anchor_marker_id(self, marker_id: int | None) -> None:
        if marker_id is None:
            self.delete_metadata("anchor_marker_id")
            return
        self.set_metadata("anchor_marker_id", str(int(marker_id)))

    def get_anchor_marker_id(self) -> int:
        configured = self.get_configured_anchor_marker_id()
        if configured is not None:
            return configured
        return self.get_default_anchor_marker_id()

    def get_configured_anchor_marker_id(self) -> int | None:
        value = self.get_metadata("anchor_marker_id")
        return None if value is None else int(value)

    def get_default_anchor_marker_id(self) -> int:
        row = self._conn.execute(
            """
            SELECT MIN(marker_id) AS marker_id
            FROM pnp_observations
            WHERE success = 1
            """
        ).fetchone()
        if row is not None and row["marker_id"] is not None:
            return int(row["marker_id"])

        row = self._conn.execute(
            """
            SELECT MIN(detections.marker_id) AS marker_id
            FROM detections
            JOIN images ON images.id = detections.image_id
            WHERE images.ignored = 0
              AND images.missing = 0
            """
        ).fetchone()
        if row is not None and row["marker_id"] is not None:
            return int(row["marker_id"])
        return 0

    def set_metadata(self, key: str, value: str) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO project_metadata(key, value)
                VALUES(?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (key, value),
            )

    def delete_metadata(self, key: str) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM project_metadata WHERE key = ?", (key,))

    def get_metadata(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM project_metadata WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    def upsert_image_index_entry(
        self,
        rel_path: str,
        size_bytes: int,
        mtime_ns: int,
        width: int | None = None,
        height: int | None = None,
    ) -> str:
        """Insert or update one indexed image.

        Returns ``"new"``, ``"updated"``, or ``"unchanged"``.
        """

        indexed_at = _utc_now()
        row = self._conn.execute(
            "SELECT id, size_bytes, mtime_ns, detection_status FROM images WHERE rel_path = ?",
            (rel_path,),
        ).fetchone()

        with self._conn:
            if row is None:
                self._conn.execute(
                    """
                    INSERT INTO images(rel_path, size_bytes, mtime_ns, width, height, missing, indexed_at)
                    VALUES(?, ?, ?, ?, ?, 0, ?)
                    """,
                    (rel_path, size_bytes, mtime_ns, width, height, indexed_at),
                )
                return "new"

            changed = int(row["size_bytes"]) != size_bytes or int(row["mtime_ns"]) != mtime_ns
            detection_status = str(row["detection_status"])
            if changed:
                new_status = "stale" if detection_status in {"detected", "no_markers", "stale"} else detection_status
                modified = 1 if detection_status in {"detected", "no_markers", "stale"} else 0
                self._conn.execute(
                    """
                    UPDATE images
                    SET size_bytes = ?,
                        mtime_ns = ?,
                        width = ?,
                        height = ?,
                        missing = 0,
                        detection_status = ?,
                        modified_since_detection = ?,
                        indexed_at = ?
                    WHERE rel_path = ?
                    """,
                    (size_bytes, mtime_ns, width, height, new_status, modified, indexed_at, rel_path),
                )
                return "updated"

            self._conn.execute(
                """
                UPDATE images
                SET width = COALESCE(?, width),
                    height = COALESCE(?, height),
                    missing = 0,
                    indexed_at = ?
                WHERE rel_path = ?
                """,
                (width, height, indexed_at, rel_path),
            )
            return "unchanged"

    def mark_missing_except(self, rel_paths: Iterable[str]) -> int:
        found = set(rel_paths)
        rows = self._conn.execute("SELECT rel_path FROM images WHERE missing = 0").fetchall()
        missing = [str(row["rel_path"]) for row in rows if str(row["rel_path"]) not in found]
        if not missing:
            return 0
        with self._conn:
            self._conn.executemany("UPDATE images SET missing = 1 WHERE rel_path = ?", [(path,) for path in missing])
        return len(missing)

    def list_images(self, include_missing: bool = True) -> list[ImageRecord]:
        where = "" if include_missing else "WHERE images.missing = 0"
        rows = self._conn.execute(
            f"""
            SELECT images.*,
                   COUNT(detections.id) AS marker_count
            FROM images
            LEFT JOIN detections ON detections.image_id = images.id
            {where}
            GROUP BY images.id
            ORDER BY images.rel_path COLLATE NOCASE
            """
        ).fetchall()
        return [_image_record_from_row(row) for row in rows]

    def get_image(self, image_id: int) -> ImageRecord | None:
        row = self._conn.execute(
            """
            SELECT images.*,
                   COUNT(detections.id) AS marker_count
            FROM images
            LEFT JOIN detections ON detections.image_id = images.id
            WHERE images.id = ?
            GROUP BY images.id
            """,
            (image_id,),
        ).fetchone()
        return None if row is None else _image_record_from_row(row)

    def list_detectable_images(self) -> list[ImageRecord]:
        return [image for image in self.list_images(include_missing=False) if not image.ignored]

    def set_image_ignored(self, image_id: int, ignored: bool) -> None:
        with self._conn:
            self._conn.execute("UPDATE images SET ignored = ? WHERE id = ?", (1 if ignored else 0, image_id))

    def create_detector_run(self, config: DetectorRunConfig, opencv_version: str | None = None) -> int:
        params_json = json.dumps(config.detector_params, sort_keys=True)
        with self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO detector_runs(started_at, detector_type, dictionary_name, detector_params_json, opencv_version)
                VALUES(?, ?, ?, ?, ?)
                """,
                (_utc_now(), config.detector_type, config.dictionary_name, params_json, opencv_version),
            )
        return int(cur.lastrowid)

    def replace_image_detections(
        self,
        image_id: int,
        detector_run_id: int,
        detections: list[MarkerDetection],
    ) -> None:
        status = "detected" if detections else "no_markers"
        with self._conn:
            self._conn.execute("DELETE FROM detections WHERE image_id = ?", (image_id,))
            self._conn.executemany(
                """
                INSERT INTO detections(
                    image_id,
                    detector_run_id,
                    marker_family,
                    dictionary_name,
                    marker_id,
                    corners_json,
                    corner_refinement_method
                )
                VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        image_id,
                        detector_run_id,
                        detection.marker_family,
                        detection.dictionary_name,
                        detection.marker_id,
                        json.dumps(detection.corners),
                        detection.corner_refinement_method,
                    )
                    for detection in detections
                ],
            )
            self._conn.execute(
                """
                UPDATE images
                SET detection_status = ?,
                    modified_since_detection = 0
                WHERE id = ?
                """,
                (status, image_id),
            )

    def get_detections_for_image(self, image_id: int) -> list[MarkerDetection]:
        rows = self._conn.execute(
            """
            SELECT marker_family, dictionary_name, marker_id, corners_json, corner_refinement_method
            FROM detections
            WHERE image_id = ?
            ORDER BY marker_id
            """,
            (image_id,),
        ).fetchall()
        return [
            MarkerDetection(
                marker_family=str(row["marker_family"]),
                dictionary_name=str(row["dictionary_name"]),
                marker_id=int(row["marker_id"]),
                corners=json.loads(str(row["corners_json"])),
                corner_refinement_method=str(row["corner_refinement_method"]),
            )
            for row in rows
        ]

    def get_marker_count_for_image(self, image_id: int) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS count FROM detections WHERE image_id = ?", (image_id,)).fetchone()
        return int(row["count"])

    def list_detection_rows_for_initialization(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT detections.id AS detection_id,
                   detections.image_id,
                   detections.marker_id,
                   detections.corners_json,
                   images.rel_path
            FROM detections
            JOIN images ON images.id = detections.image_id
            WHERE images.ignored = 0
              AND images.missing = 0
            ORDER BY images.rel_path COLLATE NOCASE, detections.marker_id
            """
        ).fetchall()
        return [
            {
                "detection_id": int(row["detection_id"]),
                "image_id": int(row["image_id"]),
                "marker_id": int(row["marker_id"]),
                "corners": json.loads(str(row["corners_json"])),
                "rel_path": str(row["rel_path"]),
            }
            for row in rows
        ]

    def replace_pnp_observations(self, observations: list[PnPObservation]) -> None:
        now = _utc_now()
        with self._conn:
            self._conn.execute("DELETE FROM pnp_observations")
            self._conn.executemany(
                """
                INSERT INTO pnp_observations(
                    image_id,
                    detection_id,
                    marker_id,
                    success,
                    rvec_json,
                    tvec_json,
                    T_C_M_json,
                    reprojection_error_px,
                    error_message,
                    created_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        obs.image_id,
                        obs.detection_id,
                        obs.marker_id,
                        1 if obs.success else 0,
                        None if obs.rvec is None else json.dumps(obs.rvec),
                        None if obs.tvec is None else json.dumps(obs.tvec),
                        None if obs.T_C_M is None else json.dumps(obs.T_C_M),
                        obs.reprojection_error_px,
                        obs.error_message,
                        obs.created_at or now,
                    )
                    for obs in observations
                ],
            )

    def list_pnp_observations(self, success_only: bool = False) -> list[PnPObservation]:
        where = "WHERE success = 1" if success_only else ""
        rows = self._conn.execute(
            f"""
            SELECT *
            FROM pnp_observations
            {where}
            ORDER BY image_id, marker_id, id
            """
        ).fetchall()
        return [_pnp_observation_from_row(row) for row in rows]

    def replace_seed_camera_poses(self, poses: list[SeedCameraPose]) -> None:
        now = _utc_now()
        with self._conn:
            self._conn.execute("DELETE FROM seed_camera_poses")
            self._conn.executemany(
                """
                INSERT INTO seed_camera_poses(
                    image_id,
                    T_W_C_json,
                    source_marker_id,
                    reprojection_error_px,
                    created_at
                )
                VALUES(?, ?, ?, ?, ?)
                """,
                [
                    (
                        pose.image_id,
                        json.dumps(pose.T_W_C),
                        pose.source_marker_id,
                        pose.reprojection_error_px,
                        pose.created_at or now,
                    )
                    for pose in poses
                ],
            )

    def replace_seed_marker_poses(self, poses: list[SeedMarkerPose]) -> None:
        now = _utc_now()
        with self._conn:
            self._conn.execute("DELETE FROM seed_marker_poses")
            self._conn.executemany(
                """
                INSERT INTO seed_marker_poses(
                    marker_id,
                    T_W_M_json,
                    source_image_id,
                    reprojection_error_px,
                    is_anchor,
                    created_at
                )
                VALUES(?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        pose.marker_id,
                        json.dumps(pose.T_W_M),
                        pose.source_image_id,
                        pose.reprojection_error_px,
                        1 if pose.is_anchor else 0,
                        pose.created_at or now,
                    )
                    for pose in poses
                ],
            )

    def get_seed_camera_poses(self) -> list[SeedCameraPose]:
        rows = self._conn.execute("SELECT * FROM seed_camera_poses ORDER BY image_id").fetchall()
        return [
            SeedCameraPose(
                image_id=int(row["image_id"]),
                T_W_C=json.loads(str(row["T_W_C_json"])),
                source_marker_id=None if row["source_marker_id"] is None else int(row["source_marker_id"]),
                reprojection_error_px=None
                if row["reprojection_error_px"] is None
                else float(row["reprojection_error_px"]),
                created_at=str(row["created_at"]),
            )
            for row in rows
        ]

    def get_seed_marker_poses(self) -> list[SeedMarkerPose]:
        rows = self._conn.execute("SELECT * FROM seed_marker_poses ORDER BY marker_id").fetchall()
        return [
            SeedMarkerPose(
                marker_id=int(row["marker_id"]),
                T_W_M=json.loads(str(row["T_W_M_json"])),
                source_image_id=None if row["source_image_id"] is None else int(row["source_image_id"]),
                reprojection_error_px=None
                if row["reprojection_error_px"] is None
                else float(row["reprojection_error_px"]),
                is_anchor=bool(row["is_anchor"]),
                created_at=str(row["created_at"]),
            )
            for row in rows
        ]

    def set_graph_diagnostics(self, values: dict[str, Any]) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM graph_diagnostics")
            self._conn.executemany(
                """
                INSERT INTO graph_diagnostics(key, value)
                VALUES(?, ?)
                """,
                [(key, json.dumps(value)) for key, value in values.items()],
            )

    def get_graph_diagnostics(self) -> dict[str, Any]:
        rows = self._conn.execute("SELECT key, value FROM graph_diagnostics ORDER BY key").fetchall()
        return {str(row["key"]): json.loads(str(row["value"])) for row in rows}

    def create_ba_run(self, config: BAConfig) -> int:
        with self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO ba_runs(
                    started_at,
                    success,
                    backend_name,
                    robust_loss_type,
                    robust_loss_scale_px,
                    corner_outlier_threshold_px,
                    marker_observation_outlier_threshold_px,
                    marker_outlier_threshold_px
                )
                VALUES(?, 0, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _utc_now(),
                    config.backend_name,
                    config.robust_loss_type,
                    config.robust_loss_scale_px,
                    config.corner_outlier_threshold_px,
                    config.marker_observation_outlier_threshold_px,
                    config.marker_observation_outlier_threshold_px,
                ),
            )
        return int(cur.lastrowid)

    def complete_ba_run(
        self,
        ba_run_id: int,
        success: bool,
        num_iterations: int | None = None,
        initial_cost: float | None = None,
        final_cost: float | None = None,
        mean_reprojection_error_px: float | None = None,
        median_reprojection_error_px: float | None = None,
        max_reprojection_error_px: float | None = None,
        num_observations: int = 0,
        num_corners: int = 0,
        num_outlier_observations: int = 0,
        solver_report: str | None = None,
        error_message: str | None = None,
    ) -> None:
        with self._conn:
            self._conn.execute(
                """
                UPDATE ba_runs
                SET completed_at = ?,
                    success = ?,
                    num_iterations = ?,
                    initial_cost = ?,
                    final_cost = ?,
                    mean_reprojection_error_px = ?,
                    median_reprojection_error_px = ?,
                    max_reprojection_error_px = ?,
                    num_observations = ?,
                    num_corners = ?,
                    num_outlier_observations = ?,
                    solver_report = ?,
                    error_message = ?
                WHERE id = ?
                """,
                (
                    _utc_now(),
                    1 if success else 0,
                    num_iterations,
                    initial_cost,
                    final_cost,
                    mean_reprojection_error_px,
                    median_reprojection_error_px,
                    max_reprojection_error_px,
                    num_observations,
                    num_corners,
                    num_outlier_observations,
                    solver_report,
                    error_message,
                    ba_run_id,
                ),
            )

    def get_latest_successful_ba_run_id(self) -> int | None:
        row = self._conn.execute(
            "SELECT id FROM ba_runs WHERE success = 1 ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return None if row is None else int(row["id"])

    def replace_optimized_marker_poses(self, ba_run_id: int, poses: list[OptimizedMarkerPose]) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM optimized_marker_poses WHERE ba_run_id = ?", (ba_run_id,))
            self._conn.executemany(
                """
                INSERT INTO optimized_marker_poses(marker_id, ba_run_id, T_W_M_json, is_anchor)
                VALUES(?, ?, ?, ?)
                """,
                [
                    (pose.marker_id, ba_run_id, json.dumps(pose.T_W_M), 1 if pose.is_anchor else 0)
                    for pose in poses
                ],
            )

    def replace_optimized_camera_poses(self, ba_run_id: int, poses: list[OptimizedCameraPose]) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM optimized_camera_poses WHERE ba_run_id = ?", (ba_run_id,))
            self._conn.executemany(
                """
                INSERT INTO optimized_camera_poses(image_id, ba_run_id, T_W_C_json)
                VALUES(?, ?, ?)
                """,
                [(pose.image_id, ba_run_id, json.dumps(pose.T_W_C)) for pose in poses],
            )

    def get_optimized_marker_poses(self, ba_run_id: int | None = None) -> list[OptimizedMarkerPose]:
        resolved_id = ba_run_id if ba_run_id is not None else self.get_latest_successful_ba_run_id()
        if resolved_id is None:
            return []
        rows = self._conn.execute(
            "SELECT * FROM optimized_marker_poses WHERE ba_run_id = ? ORDER BY marker_id",
            (resolved_id,),
        ).fetchall()
        return [
            OptimizedMarkerPose(
                marker_id=int(row["marker_id"]),
                ba_run_id=int(row["ba_run_id"]),
                T_W_M=json.loads(str(row["T_W_M_json"])),
                is_anchor=bool(row["is_anchor"]),
            )
            for row in rows
        ]

    def get_optimized_camera_poses(self, ba_run_id: int | None = None) -> list[OptimizedCameraPose]:
        resolved_id = ba_run_id if ba_run_id is not None else self.get_latest_successful_ba_run_id()
        if resolved_id is None:
            return []
        rows = self._conn.execute(
            "SELECT * FROM optimized_camera_poses WHERE ba_run_id = ? ORDER BY image_id",
            (resolved_id,),
        ).fetchall()
        return [
            OptimizedCameraPose(
                image_id=int(row["image_id"]),
                ba_run_id=int(row["ba_run_id"]),
                T_W_C=json.loads(str(row["T_W_C_json"])),
            )
            for row in rows
        ]

    def replace_reprojection_errors(self, ba_run_id: int, errors: list[ReprojectionErrorRecord]) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM reprojection_errors WHERE ba_run_id = ?", (ba_run_id,))
            self._conn.executemany(
                """
                INSERT INTO reprojection_errors(
                    ba_run_id,
                    image_id,
                    marker_id,
                    corner_index_detector_order,
                    error_px,
                    residual_x_px,
                    residual_y_px,
                    is_outlier
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        ba_run_id,
                        err.image_id,
                        err.marker_id,
                        err.corner_index_detector_order,
                        err.error_px,
                        err.residual_x_px,
                        err.residual_y_px,
                        1 if err.is_outlier else 0,
                    )
                    for err in errors
                ],
            )

    def get_reprojection_errors(self, ba_run_id: int | None = None) -> list[ReprojectionErrorRecord]:
        resolved_id = ba_run_id if ba_run_id is not None else self.get_latest_successful_ba_run_id()
        if resolved_id is None:
            return []
        rows = self._conn.execute(
            "SELECT * FROM reprojection_errors WHERE ba_run_id = ? ORDER BY image_id, marker_id, corner_index_detector_order",
            (resolved_id,),
        ).fetchall()
        return [_reprojection_error_from_row(row) for row in rows]

    def get_ba_summary(self, ba_run_id: int | None = None) -> BARunSummary | None:
        resolved_id = ba_run_id if ba_run_id is not None else self.get_latest_successful_ba_run_id()
        if resolved_id is None:
            return None
        row = self._conn.execute("SELECT * FROM ba_runs WHERE id = ?", (resolved_id,)).fetchone()
        return None if row is None else _ba_run_summary_from_row(row)


def _image_record_from_row(row: sqlite3.Row) -> ImageRecord:
    return ImageRecord(
        id=int(row["id"]),
        rel_path=str(row["rel_path"]),
        size_bytes=int(row["size_bytes"]),
        mtime_ns=int(row["mtime_ns"]),
        width=None if row["width"] is None else int(row["width"]),
        height=None if row["height"] is None else int(row["height"]),
        ignored=bool(row["ignored"]),
        missing=bool(row["missing"]),
        detection_status=str(row["detection_status"]),
        modified_since_detection=bool(row["modified_since_detection"]),
        indexed_at=str(row["indexed_at"]),
        marker_count=int(row["marker_count"]) if "marker_count" in row.keys() else 0,
    )


def _pnp_observation_from_row(row: sqlite3.Row) -> PnPObservation:
    return PnPObservation(
        id=int(row["id"]),
        image_id=int(row["image_id"]),
        detection_id=int(row["detection_id"]),
        marker_id=int(row["marker_id"]),
        success=bool(row["success"]),
        rvec=None if row["rvec_json"] is None else json.loads(str(row["rvec_json"])),
        tvec=None if row["tvec_json"] is None else json.loads(str(row["tvec_json"])),
        T_C_M=None if row["T_C_M_json"] is None else json.loads(str(row["T_C_M_json"])),
        reprojection_error_px=None if row["reprojection_error_px"] is None else float(row["reprojection_error_px"]),
        error_message=None if row["error_message"] is None else str(row["error_message"]),
        created_at=str(row["created_at"]),
    )


def _ba_run_summary_from_row(row: sqlite3.Row) -> BARunSummary:
    return BARunSummary(
        id=int(row["id"]),
        started_at=str(row["started_at"]),
        completed_at=None if row["completed_at"] is None else str(row["completed_at"]),
        success=bool(row["success"]),
        num_iterations=None if row["num_iterations"] is None else int(row["num_iterations"]),
        initial_cost=None if row["initial_cost"] is None else float(row["initial_cost"]),
        final_cost=None if row["final_cost"] is None else float(row["final_cost"]),
        mean_reprojection_error_px=None
        if row["mean_reprojection_error_px"] is None
        else float(row["mean_reprojection_error_px"]),
        median_reprojection_error_px=None
        if row["median_reprojection_error_px"] is None
        else float(row["median_reprojection_error_px"]),
        max_reprojection_error_px=None
        if row["max_reprojection_error_px"] is None
        else float(row["max_reprojection_error_px"]),
        num_observations=int(row["num_observations"]),
        num_corners=int(row["num_corners"]),
        num_outlier_observations=int(row["num_outlier_observations"]),
        backend_name=None if row["backend_name"] is None else str(row["backend_name"]),
        robust_loss_type=None if row["robust_loss_type"] is None else str(row["robust_loss_type"]),
        robust_loss_scale_px=None if row["robust_loss_scale_px"] is None else float(row["robust_loss_scale_px"]),
        corner_outlier_threshold_px=None
        if row["corner_outlier_threshold_px"] is None
        else float(row["corner_outlier_threshold_px"]),
        marker_observation_outlier_threshold_px=None
        if row["marker_observation_outlier_threshold_px"] is None
        else float(row["marker_observation_outlier_threshold_px"]),
        solver_report=None if row["solver_report"] is None else str(row["solver_report"]),
        error_message=None if row["error_message"] is None else str(row["error_message"]),
    )


def _reprojection_error_from_row(row: sqlite3.Row) -> ReprojectionErrorRecord:
    return ReprojectionErrorRecord(
        id=int(row["id"]),
        ba_run_id=int(row["ba_run_id"]),
        image_id=int(row["image_id"]),
        marker_id=int(row["marker_id"]),
        corner_index_detector_order=int(row["corner_index_detector_order"]),
        error_px=float(row["error_px"]),
        residual_x_px=float(row["residual_x_px"]),
        residual_y_px=float(row["residual_y_px"]),
        is_outlier=bool(row["is_outlier"]),
    )


def _path_to_stored_string(project_folder: Path, path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(project_folder).as_posix()
    except ValueError:
        return str(resolved)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
