"""SQLite-backed project store for the fiducial map builder."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any

from .models import DetectorRunConfig, ImageRecord, MarkerDetection, PnPObservation, SeedCameraPose, SeedMarkerPose


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
                """
            )

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

    def set_anchor_marker_id(self, marker_id: int) -> None:
        self.set_metadata("anchor_marker_id", str(int(marker_id)))

    def get_anchor_marker_id(self) -> int:
        value = self.get_metadata("anchor_marker_id")
        return 0 if value is None else int(value)

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


def _path_to_stored_string(project_folder: Path, path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(project_folder).as_posix()
    except ValueError:
        return str(resolved)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
