from __future__ import annotations
from pathlib import Path
import sqlite3, io
from datetime import datetime, timezone
import numpy as np

PROJECT_DIR_NAME='.map_builder'; DB='dense_reconstruction.sqlite'

def _ts(): return datetime.now(timezone.utc).isoformat()
def numpy_array_to_blob(array):
    buf=io.BytesIO(); np.save(buf,np.asarray(array),allow_pickle=False); return buf.getvalue()
def numpy_array_from_blob(blob):
    return np.load(io.BytesIO(blob),allow_pickle=False)
def numpy_arrays_to_blob(arrays):
    buf=io.BytesIO(); np.savez_compressed(buf, **arrays); return buf.getvalue()
def numpy_arrays_from_blob(blob):
    data=np.load(io.BytesIO(blob)); return {k:data[k] for k in data.files}

class DenseReconstructionStore:
    def __init__(self,folder:Path,conn:sqlite3.Connection): self.folder=folder; self.conn=conn; conn.row_factory=sqlite3.Row
    @classmethod
    def open(cls, folder:Path):
        side=Path(folder)/PROJECT_DIR_NAME; side.mkdir(parents=True,exist_ok=True); c=sqlite3.connect(side/DB); s=cls(Path(folder),c); s._init(); return s
    def close(self): self.conn.close()
    def _init(self):
        with self.conn:
            self.conn.executescript('''
CREATE TABLE IF NOT EXISTS dense_metadata(key TEXT PRIMARY KEY,value TEXT);
CREATE TABLE IF NOT EXISTS dense_images(image_id INTEGER PRIMARY KEY,rel_path TEXT NOT NULL,width INTEGER,height INTEGER,features_status TEXT NOT NULL DEFAULT 'not_run',num_keypoints INTEGER NOT NULL DEFAULT 0,feature_blob BLOB,descriptor_blob BLOB,score_blob BLOB,feature_dtype TEXT,descriptor_dtype TEXT,updated_at TEXT);
CREATE TABLE IF NOT EXISTS frame_pairs(id INTEGER PRIMARY KEY,image_id_a INTEGER NOT NULL,image_id_b INTEGER NOT NULL,status TEXT NOT NULL DEFAULT 'candidate',baseline_m REAL,optical_axis_angle_deg REAL,common_marker_count INTEGER,estimated_overlap_score REAL,num_raw_matches INTEGER DEFAULT 0,num_epipolar_inliers INTEGER DEFAULT 0,created_at TEXT NOT NULL,UNIQUE(image_id_a,image_id_b));
CREATE TABLE IF NOT EXISTS pair_matches(id INTEGER PRIMARY KEY,pair_id INTEGER NOT NULL,feature_idx_a INTEGER NOT NULL,feature_idx_b INTEGER NOT NULL,x_a REAL NOT NULL,y_a REAL NOT NULL,x_b REAL NOT NULL,y_b REAL NOT NULL,match_score REAL,epipolar_error REAL,is_epipolar_inlier INTEGER NOT NULL DEFAULT 0,is_used_for_track INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS tracks(id INTEGER PRIMARY KEY,status TEXT NOT NULL DEFAULT 'candidate',num_observations INTEGER NOT NULL,num_images INTEGER NOT NULL,x REAL,y REAL,z REAL,mean_reprojection_error_px REAL,max_reprojection_error_px REAL,min_triangulation_angle_deg REAL,created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS track_observations(id INTEGER PRIMARY KEY,track_id INTEGER NOT NULL,image_id INTEGER NOT NULL,feature_idx INTEGER NOT NULL,x REAL NOT NULL,y REAL NOT NULL);
CREATE TABLE IF NOT EXISTS dense_points(id INTEGER PRIMARY KEY,track_id INTEGER,x REAL NOT NULL,y REAL NOT NULL,z REAL NOT NULL,r REAL,g REAL,b REAL,mean_reprojection_error_px REAL,max_reprojection_error_px REAL,num_observations INTEGER,source TEXT NOT NULL DEFAULT 'triangulated',is_active INTEGER NOT NULL DEFAULT 1);
CREATE TABLE IF NOT EXISTS dense_ba_runs(id INTEGER PRIMARY KEY,started_at TEXT NOT NULL,completed_at TEXT,success INTEGER NOT NULL,backend_name TEXT,mode TEXT,initial_cost REAL,final_cost REAL,mean_reprojection_error_px REAL,max_reprojection_error_px REAL,num_points INTEGER,num_observations INTEGER,error_message TEXT);
CREATE INDEX IF NOT EXISTS idx_dense_images_status ON dense_images(features_status);
CREATE INDEX IF NOT EXISTS idx_frame_pairs_status ON frame_pairs(status);
CREATE INDEX IF NOT EXISTS idx_pair_matches_pair ON pair_matches(pair_id);
CREATE INDEX IF NOT EXISTS idx_pair_matches_inlier ON pair_matches(is_epipolar_inlier);
CREATE INDEX IF NOT EXISTS idx_tracks_status ON tracks(status);
CREATE INDEX IF NOT EXISTS idx_track_obs_track ON track_observations(track_id);
CREATE INDEX IF NOT EXISTS idx_track_obs_image ON track_observations(image_id);
CREATE INDEX IF NOT EXISTS idx_dense_points_active ON dense_points(is_active);
''')
    def upsert_feature(self,image_id,rel_path,keypoints=None,descriptors=None,scores=None,status='success'):
        with self.conn:
            self.conn.execute('''INSERT INTO dense_images(image_id,rel_path,features_status,num_keypoints,feature_blob,descriptor_blob,score_blob,feature_dtype,descriptor_dtype,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?) ON CONFLICT(image_id) DO UPDATE SET rel_path=excluded.rel_path,features_status=excluded.features_status,num_keypoints=excluded.num_keypoints,feature_blob=excluded.feature_blob,descriptor_blob=excluded.descriptor_blob,score_blob=excluded.score_blob,updated_at=excluded.updated_at''', (image_id,rel_path,status,0 if keypoints is None else len(keypoints),None if keypoints is None else numpy_array_to_blob(keypoints),None if descriptors is None else numpy_array_to_blob(descriptors),None if scores is None else numpy_array_to_blob(scores),None if keypoints is None else str(np.asarray(keypoints).dtype),None if descriptors is None else str(np.asarray(descriptors).dtype),_ts()))
    def list_active_dense_points(self): return self.conn.execute('SELECT x,y,z FROM dense_points WHERE is_active=1').fetchall()
