import numpy as np
from map_builder.dense_reconstruction.triangulation import triangulate_two_view

def test_triangulate_shape():
    C1=np.array([0.,0,0]); C2=np.array([1.,0,0])
    d1=np.array([[0,0,1.]]); d2=np.array([[-0.1,0,1.]])
    d1=d1/np.linalg.norm(d1,axis=1,keepdims=True); d2=d2/np.linalg.norm(d2,axis=1,keepdims=True)
    X,g=triangulate_two_view(C1,d1,C2,d2)
    assert X.shape==(1,3); assert g.shape==(1,)
