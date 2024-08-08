import numpy as np
from scipy.spatial import cKDTree

def chamfer_dist(m1_v, m2_v):
    t1, t2 = cKDTree(m1_v), cKDTree(m2_v)
    d1, _ = t1.query(m1_v)
    d2, _ = t2.query(m2_v)
    return np.mean(d1) + np.mean(d2)

fitness = lambda m1_v, m2_v: 1 / (1 + chamfer_dist(m1_v, m2_v))