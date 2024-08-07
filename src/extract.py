import numpy as np
import gpytoolbox as gpy
from tqdm import tqdm
from enum import Enum

class SampleMethod(Enum):
    RANDOM = 1
    PROXIMITY = 2

def extract_sdf(method : SampleMethod, mesh, num_samples=3500000, 
                batch_size=10000, sigma = .1):
    print('extracting sdf..')

    v, f = gpy.read_mesh(mesh)
    v = gpy.normalize_points(v)

    bbox_min = np.min(v, axis=0)
    bbox_max = np.max(v, axis=0)
    bbox_range = bbox_max - bbox_min    
    pad = .1 * bbox_range
    q_min = bbox_min - pad
    q_max = bbox_max + pad

    if method == SampleMethod.RANDOM:
        P = np.random.uniform(q_min, q_max, (num_samples, 3))

    elif method == SampleMethod.PROXIMITY:
        P = gpy.random_points_on_mesh(v, f, num_samples)
        noise = np.random.normal(0, sigma * np.mean(bbox_range), P.shape)
        P += noise
        
    sqr_sdf = np.zeros(num_samples)

    for i in tqdm(range(0, num_samples, batch_size), desc='computing sdf'):
        end = min(i + batch_size, num_samples)
        sqr_sdf[i:end], _, _ = gpy.squared_distance(P[i:end], v, F=f, use_aabb=True)
    
    sdf = np.sqrt(sqr_sdf)

    print('determining sign of sdf..')
    inside = gpy.winding_number(v, f, P) > 0.5

    if inside.shape[0] != num_samples:
        inside = np.resize(inside, num_samples)

    sdf[inside] *= -1

    print('extracted sdf')

    return P, sdf