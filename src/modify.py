import numpy as np
import gpytoolbox as gpy

def modify_mesh(mesh, percentage):
    v, f = gpy.read_mesh(mesh)
    num_faces = f.shape[0]
    num_to_rmv = int(num_faces * percentage / 100)

    keep_mask = np.ones(num_faces, dtype=bool)
    rmv_indices = np.random.choice(num_faces, num_to_rmv, replace=False)
    keep_mask[rmv_indices] = False

    f_mod = f[keep_mask]
    new_mesh = mesh.rsplit('.', 1)[0] + f'_mod_{percentage}percent.obj'

    gpy.write_mesh(new_mesh, v, f_mod)
    return new_mesh