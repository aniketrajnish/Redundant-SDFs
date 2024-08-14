import numpy as np
import gpytoolbox as gpy
from enums import SampleMethod, SDFMethod

def extract_sdf(sample_method: SampleMethod, mesh, num_samples=1250000, batch_size=50000, sigma=0.1, sdf_method=SDFMethod.SAMPLE, grid_res=50):
    print('extracting sdf...')

    v, f = gpy.read_mesh(mesh)
    v = gpy.normalize_points(v)

    if sdf_method == SDFMethod.GRID:
        x = np.linspace(-1, 1, grid_res)
        y = np.linspace(-1, 1, grid_res)
        z = np.linspace(-1, 1, grid_res)
        X, Y, Z = np.meshgrid(x, y, z)
        P = np.c_[X.flatten(), Y.flatten(), Z.flatten()]
        num_samples = P.shape[0]
    elif sdf_method == SDFMethod.SAMPLE:
        bbox_min = np.min(v, axis=0)
        bbox_max = np.max(v, axis=0)
        bbox_range = bbox_max - bbox_min    
        pad = 0.1 * bbox_range
        q_min = bbox_min - pad
        q_max = bbox_max + pad

        if sample_method == SampleMethod.RANDOM:
            P = np.random.uniform(q_min, q_max, (num_samples, 3))
        elif sample_method == SampleMethod.PROXIMITY:
            P = gpy.random_points_on_mesh(v, f, num_samples)
            noise = np.random.normal(0, sigma * np.mean(bbox_range), P.shape)
            P += noise   
    
    sdf, _, _ = gpy.signed_distance(P, v, f)

    print('extracted sdf!')

    return P, sdf