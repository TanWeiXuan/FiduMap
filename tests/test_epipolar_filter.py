import numpy as np
from map_builder.dense_reconstruction.epipolar_filter import compute_ray_angular_errors

def test_epipolar_zero_for_consistent_case():
    f1=np.array([[0,0,1.0],[0.1,0,0.99]])
    f1=f1/np.linalg.norm(f1,axis=1,keepdims=True)
    R=np.eye(3); t=np.array([1.,0,0])
    f2=f1.copy()
    e=compute_ray_angular_errors(f1,f2,R,t)
    assert np.all(e<1.0)
