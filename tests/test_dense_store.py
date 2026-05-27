import numpy as np
from map_builder.dense_reconstruction.dense_store import DenseReconstructionStore, numpy_array_to_blob, numpy_array_from_blob

def test_blob_roundtrip(tmp_path):
    a=np.array([[1,2],[3,4]],dtype=np.float32)
    assert np.allclose(numpy_array_from_blob(numpy_array_to_blob(a)),a)

def test_store_persist(tmp_path):
    s=DenseReconstructionStore.open(tmp_path)
    s.upsert_feature(1,'a.jpg',np.zeros((2,2),dtype=np.float32),np.zeros((2,16),dtype=np.float32),None)
    s.conn.execute("INSERT INTO dense_points(x,y,z,is_active) VALUES (1,2,3,1)")
    s.conn.commit(); s.close()
    s2=DenseReconstructionStore.open(tmp_path)
    rows=s2.list_active_dense_points(); assert len(rows)==1
