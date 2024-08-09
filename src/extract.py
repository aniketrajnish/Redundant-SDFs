import numpy as np
import gpytoolbox as gpy
from tqdm import tqdm
from enum import Enum

class SampleMethod(Enum):
    RANDOM = 1
    PROXIMITY = 2

def extract_sdf(method: SampleMethod, mesh, num_samples=12500, batch_size=500, sigma=0.1):
    print('extracting sdf...')

    v, f = gpy.read_mesh(mesh)
    v = gpy.normalize_points(v)

    bbox_min = np.min(v, axis=0)
    bbox_max = np.max(v, axis=0)
    bbox_range = bbox_max - bbox_min    
    pad = 0.1 * bbox_range
    q_min = bbox_min - pad
    q_max = bbox_max + pad

    if method == SampleMethod.RANDOM:
        P = np.random.uniform(q_min, q_max, (num_samples, 3))
    elif method == SampleMethod.PROXIMITY:
        P = gpy.random_points_on_mesh(v, f, num_samples)
        noise = np.random.normal(0, sigma * np.mean(bbox_range), P.shape)
        P += noise
    
    sdf = np.zeros(num_samples)

    for i in tqdm(range(0, num_samples, batch_size), desc='computing sdf'):
        end = min(i + batch_size, num_samples)
        sdf[i:end], _, _ = gpy.signed_distance(P[i:end], v, f)

    print('extracted sdf!')

    return P, sdf