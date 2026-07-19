from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Any

import numpy as np

from .models import ARTIFACT_SCHEMA_VERSION, BACKEND_DAV2, MetricDepthArtifact, MetricDepthMetrics


PROJECT_DIR_NAME = ".map_builder"
DB_NAME = "metric_depth.sqlite"
MAP_DIR_NAME = "metric_depth_maps"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MetricDepthStore:
    def __init__(self, folder: Path, conn: sqlite3.Connection):
        self.folder = folder
        self.sidecar = folder / PROJECT_DIR_NAME
        self.db_path = self.sidecar / DB_NAME
        self.conn = conn
        self.conn.row_factory = sqlite3.Row

    @classmethod
    def open(cls, folder: Path) -> "MetricDepthStore":
        root = Path(folder).expanduser().resolve()
        sidecar = root / PROJECT_DIR_NAME
        sidecar.mkdir(parents=True, exist_ok=True)
        store = cls(root, sqlite3.connect(sidecar / DB_NAME))
        store._initialize()
        return store

    def _initialize(self) -> None:
        with self.conn:
            self.conn.executescript(
                """
                PRAGMA foreign_keys = ON;
                CREATE TABLE IF NOT EXISTS depth_runs(
                    id INTEGER PRIMARY KEY,
                    backend TEXT NOT NULL,
                    model_reference TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    source_marker_ba_run_id INTEGER,
                    source_dense_signature TEXT,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    success INTEGER NOT NULL DEFAULT 0,
                    error_message TEXT
                );
                CREATE TABLE IF NOT EXISTS depth_map_records(
                    run_id INTEGER NOT NULL,
                    image_id INTEGER NOT NULL,
                    artifact_rel_path TEXT,
                    status TEXT NOT NULL,
                    width INTEGER,
                    height INTEGER,
                    prompt_count INTEGER NOT NULL DEFAULT 0,
                    prompt_coverage REAL NOT NULL DEFAULT 0,
                    valid_fraction REAL NOT NULL DEFAULT 0,
                    median_anchor_error_m REAL,
                    processing_seconds REAL NOT NULL DEFAULT 0,
                    error_message TEXT,
                    PRIMARY KEY(run_id,image_id),
                    FOREIGN KEY(run_id) REFERENCES depth_runs(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_depth_maps_image ON depth_map_records(image_id,status);
                """
            )

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "MetricDepthStore":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def create_run(self, backend: str, model_reference: str, config: dict[str, Any], marker_ba_run_id: int | None, dense_signature: str) -> int:
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO depth_runs(backend,model_reference,config_json,source_marker_ba_run_id,source_dense_signature,started_at) VALUES(?,?,?,?,?,?)",
                (backend, model_reference, json.dumps(config, sort_keys=True), marker_ba_run_id, dense_signature, _now()),
            )
        return int(cur.lastrowid)

    def complete_run(self, run_id: int, success: bool, error_message: str | None = None) -> None:
        with self.conn:
            self.conn.execute("UPDATE depth_runs SET completed_at=?,success=?,error_message=? WHERE id=?", (_now(), int(success), error_message, run_id))

    def get_run(self, run_id: int) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM depth_runs WHERE id=?", (int(run_id),)).fetchone()

    def record_failure(self, run_id: int, image_id: int, stage: str, error_message: str, metrics: MetricDepthMetrics | None = None, width: int | None = None, height: int | None = None) -> None:
        m = metrics or MetricDepthMetrics(status="failed", error_message=error_message)
        message = f"{stage}: {error_message}"
        with self.conn:
            self.conn.execute(
                """INSERT INTO depth_map_records(run_id,image_id,status,width,height,prompt_count,prompt_coverage,valid_fraction,median_anchor_error_m,processing_seconds,error_message)
                VALUES(?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(run_id,image_id) DO UPDATE SET status=excluded.status,width=excluded.width,height=excluded.height,prompt_count=excluded.prompt_count,prompt_coverage=excluded.prompt_coverage,valid_fraction=excluded.valid_fraction,median_anchor_error_m=excluded.median_anchor_error_m,processing_seconds=excluded.processing_seconds,error_message=excluded.error_message,artifact_rel_path=NULL""",
                (run_id, image_id, "failed", width, height, m.anchor_point_count, m.anchor_spatial_coverage, m.valid_output_fraction, m.median_anchor_absolute_error_m, m.processing_duration_s, message),
            )

    def save_artifact_atomic(self, run_id: int, artifact: MetricDepthArtifact) -> Path:
        target_dir = self.sidecar / MAP_DIR_NAME / str(run_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{artifact.image_id}.npz"
        handle = tempfile.NamedTemporaryFile(prefix=f".{artifact.image_id}.", suffix=".tmp", dir=target_dir, delete=False)
        temp_path = Path(handle.name)
        try:
            with handle:
                metadata = dict(artifact.metadata)
                metadata.setdefault("artifact_schema_version", ARTIFACT_SCHEMA_VERSION)
                arrays: dict[str, Any] = dict(
                    z_depth_m=np.asarray(artifact.z_depth_m, dtype=np.float32),
                    range_m=np.asarray(artifact.range_m, dtype=np.float32),
                    valid_mask=np.asarray(artifact.valid_mask, dtype=np.uint8),
                    confidence=np.asarray(artifact.confidence, dtype=np.float16),
                    metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
                    metrics_json=np.array(json.dumps(artifact.metrics.to_dict(), sort_keys=True)),
                    image_id=np.array(artifact.image_id, dtype=np.int64),
                    backend=np.array(artifact.backend),
                )
                optional = {
                    "global_spline_z_depth_m": (artifact.global_spline_z_depth_m, np.float32),
                    "spatial_log_correction": (artifact.spatial_log_correction, np.float32),
                    "alignment_extrapolation_mask": (artifact.alignment_extrapolation_mask, np.uint8),
                    "anchor_mask": (artifact.anchor_mask, np.uint8),
                    "anchor_residual_m": (artifact.anchor_residual_m, np.float32),
                    "anchor_split": (artifact.anchor_split, np.uint8),
                    "dav2_prediction": (artifact.dav2_prediction, np.float32),
                }
                for name, (value, dtype) in optional.items():
                    if value is not None:
                        arrays[name] = np.asarray(value, dtype=dtype)
                np.savez_compressed(handle, **arrays)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, target)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
        rel = target.relative_to(self.sidecar).as_posix()
        m = artifact.metrics
        with self.conn:
            self.conn.execute(
                """INSERT INTO depth_map_records(run_id,image_id,artifact_rel_path,status,width,height,prompt_count,prompt_coverage,valid_fraction,median_anchor_error_m,processing_seconds,error_message)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,NULL) ON CONFLICT(run_id,image_id) DO UPDATE SET artifact_rel_path=excluded.artifact_rel_path,status='success',width=excluded.width,height=excluded.height,prompt_count=excluded.prompt_count,prompt_coverage=excluded.prompt_coverage,valid_fraction=excluded.valid_fraction,median_anchor_error_m=excluded.median_anchor_error_m,processing_seconds=excluded.processing_seconds,error_message=NULL""",
                (run_id, artifact.image_id, rel, "success", artifact.width, artifact.height, m.anchor_point_count, m.anchor_spatial_coverage, m.valid_output_fraction, m.median_anchor_absolute_error_m, m.processing_duration_s),
            )
        return target

    def load_artifact(self, run_id: int, image_id: int) -> MetricDepthArtifact | None:
        row = self.get_record(run_id, image_id)
        if row is None or row["status"] != "success" or not row["artifact_rel_path"]:
            return None
        path = self.sidecar / str(row["artifact_rel_path"])
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as data:
            names = set(data.files)
            metadata = json.loads(str(data["metadata_json"].item())) if "metadata_json" in names else {"artifact_schema_version": 1}
            metrics = _load_metrics(str(data["metrics_json"].item())) if "metrics_json" in names else MetricDepthMetrics(
                anchor_point_count=int(row["prompt_count"] or 0),
                anchor_spatial_coverage=float(row["prompt_coverage"] or 0.0),
                valid_output_fraction=float(row["valid_fraction"] or 0.0),
                median_anchor_absolute_error_m=row["median_anchor_error_m"],
                processing_duration_s=float(row["processing_seconds"] or 0.0),
                status="success",
            )
            def optional(name: str, dtype: Any) -> np.ndarray | None:
                return None if name not in names else data[name].astype(dtype)
            return MetricDepthArtifact(
                int(data["image_id"]) if "image_id" in names else int(image_id),
                str(data["backend"].item()) if "backend" in names else BACKEND_DAV2,
                int(row["width"] or data["z_depth_m"].shape[1]), int(row["height"] or data["z_depth_m"].shape[0]),
                data["z_depth_m"].astype(np.float32), data["range_m"].astype(np.float32), data["valid_mask"].astype(bool),
                data["confidence"].astype(np.float32), metadata, metrics,
                optional("global_spline_z_depth_m", np.float32), optional("spatial_log_correction", np.float32),
                optional("alignment_extrapolation_mask", bool), optional("anchor_mask", bool),
                optional("anchor_residual_m", np.float32), optional("anchor_split", np.uint8),
                optional("dav2_prediction", np.float32),
            )

    def get_record(self, run_id: int, image_id: int) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM depth_map_records WHERE run_id=? AND image_id=?", (int(run_id), int(image_id))).fetchone()

    def latest_successful_record(self, image_id: int, current_marker_ba_run_id: int | None, backend: str = BACKEND_DAV2) -> sqlite3.Row | None:
        sql = """SELECT r.*,d.backend,d.source_marker_ba_run_id,d.model_reference FROM depth_map_records r JOIN depth_runs d ON d.id=r.run_id
                 WHERE r.image_id=? AND r.status='success' AND d.source_marker_ba_run_id IS ?"""
        args: list[Any] = [int(image_id), current_marker_ba_run_id]
        sql += " AND d.backend=?"
        args.append(backend)
        sql += " ORDER BY r.run_id DESC LIMIT 1"
        return self.conn.execute(sql, tuple(args)).fetchone()

    def latest_run_id(self, backend: str = BACKEND_DAV2) -> int | None:
        row = self.conn.execute("SELECT id FROM depth_runs WHERE backend=? ORDER BY id DESC LIMIT 1", (backend,)).fetchone()
        return None if row is None else int(row["id"])

    def counts(self, current_marker_ba_run_id: int | None) -> dict[str, float | int | None]:
        rows = self.conn.execute(
            """SELECT r.*,d.source_marker_ba_run_id FROM depth_map_records r JOIN depth_runs d ON d.id=r.run_id WHERE d.backend=?""",
            (BACKEND_DAV2,),
        ).fetchall()
        successful = [r for r in rows if r["status"] == "success"]
        current = [r for r in successful if r["source_marker_ba_run_id"] == current_marker_ba_run_id]
        stale = len(successful) - len(current)
        def mean(name: str) -> float:
            values = [float(r[name]) for r in current if r[name] is not None]
            return float(np.mean(values)) if values else 0.0

        errors = [float(r["median_anchor_error_m"]) for r in current if r["median_anchor_error_m"] is not None]
        return {
            "completed": len(current),
            "failed": sum(r["status"] == "failed" for r in rows),
            "stale": stale,
            "mean_anchor_count": mean("prompt_count"),
            "mean_anchor_coverage": mean("prompt_coverage"),
            "mean_valid_fraction": mean("valid_fraction"),
            "median_anchor_error_m": float(np.median(errors)) if errors else None,
            "mean_processing_seconds": mean("processing_seconds"),
        }


def _load_metrics(encoded: str) -> MetricDepthMetrics:
    raw = json.loads(encoded)
    legacy_names = {
        "prompt_point_count": "anchor_point_count",
        "prompt_pixel_count": "anchor_pixel_count",
        "prompt_spatial_coverage": "anchor_spatial_coverage",
    }
    for old, new in legacy_names.items():
        if old in raw and new not in raw:
            raw[new] = raw[old]
    allowed = MetricDepthMetrics.__dataclass_fields__
    return MetricDepthMetrics(**{key: value for key, value in raw.items() if key in allowed})


def dense_state_signature(dense_store: Any) -> str:
    counts = dense_store.dense_counts()
    row = dense_store.conn.execute("SELECT COALESCE(MAX(id),0) AS max_id FROM dense_points WHERE is_active=1").fetchone()
    return f"active_points={counts.get('points', 0)};tracks={counts.get('tracks', 0)};max_active_point_id={int(row['max_id'])}"
