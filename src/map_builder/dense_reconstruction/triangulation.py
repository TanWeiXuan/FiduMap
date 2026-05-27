from __future__ import annotations
import numpy as np

def triangulate_two_view(C1,d1,C2,d2):
    w0=C1-C2
    a=np.sum(d1*d1,axis=1); b=np.sum(d1*d2,axis=1); c=np.sum(d2*d2,axis=1)
    d=np.sum(d1*w0,axis=1); e=np.sum(d2*w0,axis=1)
    den=a*c-b*b+1e-12
    s=(b*e-c*d)/den; t=(a*e-b*d)/den
    p1=C1+s[:,None]*d1; p2=C2+t[:,None]*d2
    return 0.5*(p1+p2), np.linalg.norm(p1-p2,axis=1)
