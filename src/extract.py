import numpy as np
import gpytoolbox as gpy
from tqdm import tqdm

def extract_sdf(mesh, num_samples=500000, batch_size=100):
    print('extracting sdf..')

    v, f = gpy.read_mesh(mesh)

    v = gpy.normalize_points(v)

    bbox_min = np.min(v, axis=0)
    bbox_max = np.max(v, axis=0)
    bbox_range = bbox_max - bbox_min    
    pad = .1 * bbox_range

    q_min = bbox_min - pad
    q_max = bbox_max + pad
    P = np.random.uniform(q_min, q_max, (num_samples, 3))

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