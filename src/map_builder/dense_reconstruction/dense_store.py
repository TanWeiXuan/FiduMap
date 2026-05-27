from __future__ import annotations

from datetime import datetime, timezone
import io
from pathlib import Path
import sqlite3
from typing import Any

import numpy as np

from .models import (
    DenseFeatureRecord,
    DensePointRecord,
    FramePairRecord,
    PairMatchRecord,
    TrackObservationRecord,
    TrackRecord,
)

PROJECT_DIR_NAME = ".map_builder"
DB = "dense_reconstruction.sqlite"
BLOB_FORMAT_VERSION = "np.savez_compressed.v1"


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def numpy_array_to_blob(array: Any) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, array=np.asarray(array))
    return buf.getvalue()


def numpy_array_from_blob(blob: bytes) -> np.ndarray:
    data = np.load(io.BytesIO(blob), allow_pickle=False)
    if isinstance(data, np.lib.npyio.NpzFile):
        return data["array"]
    return data


def numpy_arrays_to_blob(arrays: dict[str, np.ndarray]) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def numpy_arrays_from_blob(blob: bytes) -> dict[str, np.ndarray]:
    data = np.load(io.BytesIO(blob), allow_pickle=False)
    return {k: data[k] for k in data.files}


class DenseReconstructionStore:
    def __init__(self, folder: Path, conn: sqlite3.Connection):
        self.folder = folder
        self.conn = conn
        self.conn.row_factory = sqlite3.Row

    @classmethod
    def open(cls, folder: Path) -> "DenseReconstructionStore":
        project_folder = Path(folder).expanduser().resolve()
        side = project_folder / PROJECT_DIR_NAME
        side.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(side / DB)
        store = cls(project_folder, conn)
        store._init()
        return store

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "DenseReconstructionStore":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _init(self) -> None:
        with self.conn:
            self.conn.executescript(
                """
                PRAGMA foreign_keys = ON;
                CREATE TABLE IF NOT EXISTS dense_metadata(key TEXT PRIMARY KEY,value TEXT);
                CREATE TABLE IF NOT EXISTS dense_images(
                    image_id INTEGER PRIMARY KEY,
                    rel_path TEXT NOT NULL,
                    width INTEGER,
                    height INTEGER,
                    features_status TEXT NOT NULL DEFAULT 'not_run',
                    num_keypoints INTEGER NOT NULL DEFAULT 0,
                    feature_blob BLOB,
                    descriptor_blob BLOB,
                    score_blob BLOB,
                    feature_dtype TEXT,
                    descriptor_dtype TEXT,
                    updated_at TEXT
                );
                CREATE TABLE IF NOT EXISTS frame_pairs(
                    id INTEGER PRIMARY KEY,
                    image_id_a INTEGER NOT NULL,
                    image_id_b INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'candidate',
                    baseline_m REAL,
                    optical_axis_angle_deg REAL,
                    common_marker_count INTEGER,
                    estimated_overlap_score REAL,
                    num_raw_matches INTEGER DEFAULT 0,
                    num_epipolar_inliers INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    UNIQUE(image_id_a,image_id_b)
                );
                CREATE TABLE IF NOT EXISTS pair_matches(
                    id INTEGER PRIMARY KEY,
                    pair_id INTEGER NOT NULL,
                    feature_idx_a INTEGER NOT NULL,
                    feature_idx_b INTEGER NOT NULL,
                    x_a REAL NOT NULL,
                    y_a REAL NOT NULL,
                    x_b REAL NOT NULL,
                    y_b REAL NOT NULL,
                    match_score REAL,
                    epipolar_error REAL,
                    is_epipolar_inlier INTEGER NOT NULL DEFAULT 0,
                    is_used_for_track INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY(pair_id) REFERENCES frame_pairs(id)
                );
                CREATE TABLE IF NOT EXISTS tracks(
                    id INTEGER PRIMARY KEY,
                    status TEXT NOT NULL DEFAULT 'candidate',
                    num_observations INTEGER NOT NULL,
                    num_images INTEGER NOT NULL,
                    x REAL,
                    y REAL,
                    z REAL,
                    mean_reprojection_error_px REAL,
                    max_reprojection_error_px REAL,
                    min_triangulation_angle_deg REAL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS track_observations(
                    id INTEGER PRIMARY KEY,
                    track_id INTEGER NOT NULL,
                    image_id INTEGER NOT NULL,
                    feature_idx INTEGER NOT NULL,
                    x REAL NOT NULL,
                    y REAL NOT NULL,
                    FOREIGN KEY(track_id) REFERENCES tracks(id)
                );
                CREATE TABLE IF NOT EXISTS dense_points(
                    id INTEGER PRIMARY KEY,
                    track_id INTEGER,
                    x REAL NOT NULL,
                    y REAL NOT NULL,
                    z REAL NOT NULL,
                    r REAL,
                    g REAL,
                    b REAL,
                    mean_reprojection_error_px REAL,
                    max_reprojection_error_px REAL,
                    num_observations INTEGER,
                    source TEXT NOT NULL DEFAULT 'triangulated',
                    is_active INTEGER NOT NULL DEFAULT 1
                );
                CREATE TABLE IF NOT EXISTS dense_ba_runs(
                    id INTEGER PRIMARY KEY,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    success INTEGER NOT NULL,
                    backend_name TEXT,
                    mode TEXT,
                    initial_cost REAL,
                    final_cost REAL,
                    mean_reprojection_error_px REAL,
                    max_reprojection_error_px REAL,
                    num_points INTEGER,
                    num_observations INTEGER,
                    error_message TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_dense_images_status ON dense_images(features_status);
                CREATE INDEX IF NOT EXISTS idx_frame_pairs_status ON frame_pairs(status);
                CREATE INDEX IF NOT EXISTS idx_pair_matches_pair ON pair_matches(pair_id);
                CREATE INDEX IF NOT EXISTS idx_pair_matches_inlier ON pair_matches(is_epipolar_inlier);
                CREATE INDEX IF NOT EXISTS idx_tracks_status ON tracks(status);
                CREATE INDEX IF NOT EXISTS idx_track_obs_track ON track_observations(track_id);
                CREATE INDEX IF NOT EXISTS idx_track_obs_image ON track_observations(image_id);
                CREATE INDEX IF NOT EXISTS idx_dense_points_active ON dense_points(is_active);
                """
            )
            self.set_metadata("blob_format_version", BLOB_FORMAT_VERSION)

    def set_metadata(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO dense_metadata(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    def get_metadata(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM dense_metadata WHERE key=?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    def upsert_feature(
        self,
        image_id: int,
        rel_path: str,
        keypoints: Any | None = None,
        descriptors: Any | None = None,
        scores: Any | None = None,
        status: str = "success",
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        kpts = None if keypoints is None else np.asarray(keypoints)
        desc = None if descriptors is None else np.asarray(descriptors)
        scr = None if scores is None else np.asarray(scores)
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO dense_images(
                    image_id,rel_path,width,height,features_status,num_keypoints,
                    feature_blob,descriptor_blob,score_blob,feature_dtype,descriptor_dtype,updated_at
                )
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(image_id) DO UPDATE SET
                    rel_path=excluded.rel_path,
                    width=excluded.width,
                    height=excluded.height,
                    features_status=excluded.features_status,
                    num_keypoints=excluded.num_keypoints,
                    feature_blob=excluded.feature_blob,
                    descriptor_blob=excluded.descriptor_blob,
                    score_blob=excluded.score_blob,
                    feature_dtype=excluded.feature_dtype,
                    descriptor_dtype=excluded.descriptor_dtype,
                    updated_at=excluded.updated_at
                """,
                (
                    int(image_id),
                    rel_path,
                    width,
                    height,
                    status,
                    0 if kpts is None else int(len(kpts)),
                    None if kpts is None else numpy_array_to_blob(kpts),
                    None if desc is None else numpy_array_to_blob(desc),
                    None if scr is None else numpy_array_to_blob(scr),
                    None if kpts is None else str(kpts.dtype),
                    None if desc is None else str(desc.dtype),
                    _ts(),
                ),
            )

    def upsert_feature_record(self, rec: DenseFeatureRecord) -> None:
        self.upsert_feature(
            rec.image_id,
            rec.rel_path,
            rec.keypoints,
            rec.descriptors,
            rec.scores,
            rec.status,
            rec.width,
            rec.height,
        )

    def get_feature(self, image_id: int) -> DenseFeatureRecord | None:
        row = self.conn.execute("SELECT * FROM dense_images WHERE image_id=?", (image_id,)).fetchone()
        if row is None:
            return None
        return _feature_from_row(row)

    def list_features(self, status: str | None = None) -> list[DenseFeatureRecord]:
        if status is None:
            rows = self.conn.execute("SELECT * FROM dense_images ORDER BY image_id").fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM dense_images WHERE features_status=? ORDER BY image_id", (status,)
            ).fetchall()
        return [_feature_from_row(row) for row in rows]

    def upsert_frame_pair(self, rec: FramePairRecord) -> int:
        a, b = sorted((int(rec.image_id_a), int(rec.image_id_b)))
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO frame_pairs(
                    image_id_a,image_id_b,status,baseline_m,optical_axis_angle_deg,
                    common_marker_count,estimated_overlap_score,num_raw_matches,num_epipolar_inliers,created_at
                )
                VALUES(?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(image_id_a,image_id_b) DO UPDATE SET
                    status=excluded.status,
                    baseline_m=excluded.baseline_m,
                    optical_axis_angle_deg=excluded.optical_axis_angle_deg,
                    common_marker_count=excluded.common_marker_count,
                    estimated_overlap_score=excluded.estimated_overlap_score,
                    num_raw_matches=excluded.num_raw_matches,
                    num_epipolar_inliers=excluded.num_epipolar_inliers
                """,
                (
                    a,
                    b,
                    rec.status,
                    rec.baseline_m,
                    rec.optical_axis_angle_deg,
                    rec.common_marker_count,
                    rec.estimated_overlap_score,
                    rec.num_raw_matches,
                    rec.num_epipolar_inliers,
                    _ts(),
                ),
            )
        row = self.conn.execute("SELECT id FROM frame_pairs WHERE image_id_a=? AND image_id_b=?", (a, b)).fetchone()
        return int(row["id"])

    def list_frame_pairs(self, status: str | None = None) -> list[FramePairRecord]:
        if status is None:
            rows = self.conn.execute("SELECT * FROM frame_pairs ORDER BY image_id_a,image_id_b").fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM frame_pairs WHERE status=? ORDER BY image_id_a,image_id_b", (status,)
            ).fetchall()
        return [_frame_pair_from_row(row) for row in rows]

    def get_frame_pair(self, pair_id: int) -> FramePairRecord | None:
        row = self.conn.execute("SELECT * FROM frame_pairs WHERE id=?", (pair_id,)).fetchone()
        return None if row is None else _frame_pair_from_row(row)

    def replace_pair_matches(self, pair_id: int, matches: list[PairMatchRecord]) -> list[PairMatchRecord]:
        with self.conn:
            self.conn.execute("DELETE FROM pair_matches WHERE pair_id=?", (pair_id,))
            for match in matches:
                match.pair_id = int(pair_id)
                cur = self.conn.execute(
                    """
                    INSERT INTO pair_matches(
                        pair_id,feature_idx_a,feature_idx_b,x_a,y_a,x_b,y_b,match_score,
                        epipolar_error,is_epipolar_inlier,is_used_for_track
                    )
                    VALUES(?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        int(pair_id),
                        match.feature_idx_a,
                        match.feature_idx_b,
                        match.x_a,
                        match.y_a,
                        match.x_b,
                        match.y_b,
                        match.match_score,
                        match.epipolar_error,
                        int(match.is_epipolar_inlier),
                        int(match.is_used_for_track),
                    ),
                )
                match.id = int(cur.lastrowid)
            self.conn.execute(
                "UPDATE frame_pairs SET num_raw_matches=?, status=? WHERE id=?",
                (len(matches), "matched", pair_id),
            )
        return matches

    def list_pair_matches(
        self, pair_id: int | None = None, epipolar_inliers_only: bool = False
    ) -> list[PairMatchRecord]:
        clauses: list[str] = []
        args: list[Any] = []
        if pair_id is not None:
            clauses.append("pair_id=?")
            args.append(pair_id)
        if epipolar_inliers_only:
            clauses.append("is_epipolar_inlier=1")
        where = "" if not clauses else "WHERE " + " AND ".join(clauses)
        rows = self.conn.execute(f"SELECT * FROM pair_matches {where} ORDER BY pair_id,id", args).fetchall()
        return [_pair_match_from_row(row) for row in rows]

    def update_pair_epipolar_results(
        self, pair_id: int, match_ids: list[int], errors: np.ndarray, inliers: np.ndarray, min_inliers: int = 8
    ) -> int:
        if len(match_ids) != len(errors) or len(match_ids) != len(inliers):
            raise ValueError("Epipolar update inputs must have matching lengths.")
        if any(int(match_id) <= 0 for match_id in match_ids):
            raise ValueError("Epipolar update requires persisted pair_match row IDs.")
        inlier_count = int(np.count_nonzero(inliers))
        status = "filtered" if inlier_count >= min_inliers else "few_inliers"
        with self.conn:
            for match_id, error, is_inlier in zip(match_ids, errors, inliers):
                self.conn.execute(
                    "UPDATE pair_matches SET epipolar_error=?, is_epipolar_inlier=? WHERE id=?",
                    (float(error), 1 if bool(is_inlier) else 0, int(match_id)),
                )
            self.conn.execute(
                "UPDATE frame_pairs SET num_epipolar_inliers=?, status=? WHERE id=?",
                (inlier_count, status, pair_id),
            )
        return inlier_count

    def replace_tracks_and_points(
        self,
        tracks: list[tuple[TrackRecord, list[TrackObservationRecord], DensePointRecord]],
        clear_existing: bool = True,
    ) -> None:
        with self.conn:
            if clear_existing:
                self.conn.execute("DELETE FROM track_observations")
                self.conn.execute("DELETE FROM dense_points WHERE source='triangulated'")
                self.conn.execute("DELETE FROM tracks")
                self.conn.execute("UPDATE pair_matches SET is_used_for_track=0")
            for track, observations, point in tracks:
                cur = self.conn.execute(
                    """
                    INSERT INTO tracks(
                        status,num_observations,num_images,x,y,z,mean_reprojection_error_px,
                        max_reprojection_error_px,min_triangulation_angle_deg,created_at
                    )
                    VALUES(?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        track.status,
                        track.num_observations,
                        track.num_images,
                        track.x,
                        track.y,
                        track.z,
                        track.mean_reprojection_error_px,
                        track.max_reprojection_error_px,
                        track.min_triangulation_angle_deg,
                        _ts(),
                    ),
                )
                track_id = int(cur.lastrowid)
                self.conn.executemany(
                    """
                    INSERT INTO track_observations(track_id,image_id,feature_idx,x,y)
                    VALUES(?,?,?,?,?)
                    """,
                    [(track_id, o.image_id, o.feature_idx, o.x, o.y) for o in observations],
                )
                self.conn.execute(
                    """
                    INSERT INTO dense_points(
                        track_id,x,y,z,r,g,b,mean_reprojection_error_px,max_reprojection_error_px,
                        num_observations,source,is_active
                    )
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        track_id,
                        point.x,
                        point.y,
                        point.z,
                        point.r,
                        point.g,
                        point.b,
                        point.mean_reprojection_error_px,
                        point.max_reprojection_error_px,
                        point.num_observations,
                        point.source,
                        int(point.is_active),
                    ),
                )

    def list_tracks(self, status: str | None = None) -> list[TrackRecord]:
        if status is None:
            rows = self.conn.execute("SELECT * FROM tracks ORDER BY id").fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM tracks WHERE status=? ORDER BY id", (status,)).fetchall()
        return [_track_from_row(row) for row in rows]

    def list_track_observations(self, track_id: int | None = None) -> list[TrackObservationRecord]:
        if track_id is None:
            rows = self.conn.execute("SELECT * FROM track_observations ORDER BY track_id,id").fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM track_observations WHERE track_id=? ORDER BY id", (track_id,)
            ).fetchall()
        return [_track_obs_from_row(row) for row in rows]

    def insert_dense_point(self, point: DensePointRecord) -> int:
        with self.conn:
            cur = self.conn.execute(
                """
                INSERT INTO dense_points(
                    track_id,x,y,z,r,g,b,mean_reprojection_error_px,max_reprojection_error_px,
                    num_observations,source,is_active
                )
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    point.track_id,
                    point.x,
                    point.y,
                    point.z,
                    point.r,
                    point.g,
                    point.b,
                    point.mean_reprojection_error_px,
                    point.max_reprojection_error_px,
                    point.num_observations,
                    point.source,
                    int(point.is_active),
                ),
            )
        return int(cur.lastrowid)

    def list_active_dense_points(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM dense_points WHERE is_active=1 ORDER BY id").fetchall()

    def replace_active_dense_points(self, points: list[DensePointRecord], source: str = "merged") -> None:
        with self.conn:
            self.conn.execute("UPDATE dense_points SET is_active=0")
            for p in points:
                p.source = source
                p.is_active = 1
                self.insert_dense_point(p)

    def update_dense_point_coordinates(self, point_id: int, xyz: np.ndarray, source: str = "dense_ba") -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE dense_points SET x=?, y=?, z=?, source=? WHERE id=?",
                (float(xyz[0]), float(xyz[1]), float(xyz[2]), source, int(point_id)),
            )

    def create_dense_ba_run(self, mode: str, backend_name: str = "pyceres") -> int:
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO dense_ba_runs(started_at,success,backend_name,mode) VALUES(?,0,?,?)",
                (_ts(), backend_name, mode),
            )
        return int(cur.lastrowid)

    def complete_dense_ba_run(
        self,
        run_id: int,
        success: bool,
        initial_cost: float | None = None,
        final_cost: float | None = None,
        mean_reprojection_error_px: float | None = None,
        max_reprojection_error_px: float | None = None,
        num_points: int | None = None,
        num_observations: int | None = None,
        error_message: str | None = None,
    ) -> None:
        with self.conn:
            self.conn.execute(
                """
                UPDATE dense_ba_runs SET completed_at=?,success=?,initial_cost=?,final_cost=?,
                    mean_reprojection_error_px=?,max_reprojection_error_px=?,num_points=?,
                    num_observations=?,error_message=?
                WHERE id=?
                """,
                (
                    _ts(),
                    1 if success else 0,
                    initial_cost,
                    final_cost,
                    mean_reprojection_error_px,
                    max_reprojection_error_px,
                    num_points,
                    num_observations,
                    error_message,
                    run_id,
                ),
            )

    def dense_counts(self) -> dict[str, int]:
        def count(sql: str, args: tuple[Any, ...] = ()) -> int:
            return int(self.conn.execute(sql, args).fetchone()["c"])

        return {
            "features": count("SELECT COUNT(*) c FROM dense_images WHERE features_status='success'"),
            "pairs": count("SELECT COUNT(*) c FROM frame_pairs"),
            "matches": count("SELECT COUNT(*) c FROM pair_matches"),
            "inliers": count("SELECT COUNT(*) c FROM pair_matches WHERE is_epipolar_inlier=1"),
            "tracks": count("SELECT COUNT(*) c FROM tracks"),
            "points": count("SELECT COUNT(*) c FROM dense_points WHERE is_active=1"),
        }


def _feature_from_row(row: sqlite3.Row) -> DenseFeatureRecord:
    keypoints = None if row["feature_blob"] is None else numpy_array_from_blob(row["feature_blob"])
    descriptors = None if row["descriptor_blob"] is None else numpy_array_from_blob(row["descriptor_blob"])
    scores = None if row["score_blob"] is None else numpy_array_from_blob(row["score_blob"])
    return DenseFeatureRecord(
        image_id=int(row["image_id"]),
        rel_path=str(row["rel_path"]),
        width=None if row["width"] is None else int(row["width"]),
        height=None if row["height"] is None else int(row["height"]),
        keypoints=keypoints,
        descriptors=descriptors,
        scores=scores,
        status=str(row["features_status"]),
        num_keypoints=int(row["num_keypoints"]),
    )


def _frame_pair_from_row(row: sqlite3.Row) -> FramePairRecord:
    return FramePairRecord(
        id=int(row["id"]),
        image_id_a=int(row["image_id_a"]),
        image_id_b=int(row["image_id_b"]),
        status=str(row["status"]),
        baseline_m=None if row["baseline_m"] is None else float(row["baseline_m"]),
        optical_axis_angle_deg=None if row["optical_axis_angle_deg"] is None else float(row["optical_axis_angle_deg"]),
        common_marker_count=0 if row["common_marker_count"] is None else int(row["common_marker_count"]),
        estimated_overlap_score=0.0
        if row["estimated_overlap_score"] is None
        else float(row["estimated_overlap_score"]),
        num_raw_matches=0 if row["num_raw_matches"] is None else int(row["num_raw_matches"]),
        num_epipolar_inliers=0
        if row["num_epipolar_inliers"] is None
        else int(row["num_epipolar_inliers"]),
    )


def _pair_match_from_row(row: sqlite3.Row) -> PairMatchRecord:
    return PairMatchRecord(
        id=int(row["id"]),
        pair_id=int(row["pair_id"]),
        feature_idx_a=int(row["feature_idx_a"]),
        feature_idx_b=int(row["feature_idx_b"]),
        x_a=float(row["x_a"]),
        y_a=float(row["y_a"]),
        x_b=float(row["x_b"]),
        y_b=float(row["y_b"]),
        match_score=None if row["match_score"] is None else float(row["match_score"]),
        epipolar_error=None if row["epipolar_error"] is None else float(row["epipolar_error"]),
        is_epipolar_inlier=int(row["is_epipolar_inlier"]),
        is_used_for_track=int(row["is_used_for_track"]),
    )


def _track_from_row(row: sqlite3.Row) -> TrackRecord:
    return TrackRecord(
        id=int(row["id"]),
        status=str(row["status"]),
        num_observations=int(row["num_observations"]),
        num_images=int(row["num_images"]),
        x=None if row["x"] is None else float(row["x"]),
        y=None if row["y"] is None else float(row["y"]),
        z=None if row["z"] is None else float(row["z"]),
        mean_reprojection_error_px=None
        if row["mean_reprojection_error_px"] is None
        else float(row["mean_reprojection_error_px"]),
        max_reprojection_error_px=None
        if row["max_reprojection_error_px"] is None
        else float(row["max_reprojection_error_px"]),
        min_triangulation_angle_deg=None
        if row["min_triangulation_angle_deg"] is None
        else float(row["min_triangulation_angle_deg"]),
    )


def _track_obs_from_row(row: sqlite3.Row) -> TrackObservationRecord:
    return TrackObservationRecord(
        id=int(row["id"]),
        track_id=int(row["track_id"]),
        image_id=int(row["image_id"]),
        feature_idx=int(row["feature_idx"]),
        x=float(row["x"]),
        y=float(row["y"]),
    )
