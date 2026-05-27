from __future__ import annotations
import numpy as np

def compute_ray_angular_errors(f1:np.ndarray,f2:np.ndarray,R21:np.ndarray,t21:np.ndarray,eps:float=1e-12)->np.ndarray:
    rf1=(R21@f1.T).T
    n2=np.cross(np.broadcast_to(t21,(rf1.shape[0],3)),rf1)
    return np.abs(np.sum(f2*n2,axis=1))/np.maximum(np.linalg.norm(n2,axis=1),eps)
