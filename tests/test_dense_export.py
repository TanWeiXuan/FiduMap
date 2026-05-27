from map_builder.dense_reconstruction.dense_store import DenseReconstructionStore
from map_builder.dense_reconstruction.point_cloud_export import (
    DEFAULT_DENSE_POINT_CLOUD_FILENAME,
    export_dense_point_cloud_csv,
)

def test_export(tmp_path):
    s=DenseReconstructionStore.open(tmp_path)
    s.conn.execute("INSERT INTO dense_points(x,y,z,is_active) VALUES (1,2,3,1)"); s.conn.commit()
    s.conn.execute("INSERT INTO dense_points(x,y,z,is_active) VALUES (4.5,5.25,6.125,1)"); s.conn.commit()
    s.conn.execute("INSERT INTO dense_points(x,y,z,is_active) VALUES (7,8,9,0)"); s.conn.commit()
    out=tmp_path/DEFAULT_DENSE_POINT_CLOUD_FILENAME
    n=export_dense_point_cloud_csv(s,out)
    assert n==2
    lines = out.read_text().splitlines()
    assert lines[0]=='x,y,z'
    assert "marker" not in lines[0].lower()
    assert "corner" not in lines[0].lower()
    assert lines[1].split(",") == ["1.0", "2.0", "3.0"]
    assert lines[2].split(",") == ["4.5", "5.25", "6.125"]
    assert all(len(line.split(",")) == 3 for line in lines)
