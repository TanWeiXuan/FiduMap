from map_builder.dense_reconstruction.dense_store import DenseReconstructionStore
from map_builder.dense_reconstruction.point_cloud_export import export_dense_point_cloud_csv

def test_export(tmp_path):
    s=DenseReconstructionStore.open(tmp_path)
    s.conn.execute("INSERT INTO dense_points(x,y,z,is_active) VALUES (1,2,3,1)"); s.conn.commit()
    out=tmp_path/'dense_point_cloud.csv'
    n=export_dense_point_cloud_csv(s,out)
    assert n==1
    assert out.read_text().splitlines()[0]=='x,y,z'
