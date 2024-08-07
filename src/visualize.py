import numpy as np
import polyscope as ps
from scipy.interpolate import griddata
import gpytoolbox as gpy

def viz_sdf(pts, sdf, mesh, grid_res=128):
    ps.init()

    v_orig, f_orig = gpy.read_mesh(mesh)
    v_orig = gpy.normalize_points(v_orig)  
  
    v_orig[:, 2] += 1.0  # move the mesh right a bit
    
    original_mesh = ps.register_surface_mesh("original mesh", v_orig, f_orig, smooth_shade=True)

    bbox_min = np.min(pts, axis=0)
    bbox_max = np.max(pts, axis=0)
    bbox_range = bbox_max - bbox_min
    bbox_min -= 0.1 * bbox_range
    bbox_max += 0.1 * bbox_range

    dims = (grid_res, grid_res, grid_res)
    ps_grid = ps.register_volume_grid("sdf grid", dims, bbox_min, bbox_max)

    x = np.linspace(bbox_min[0], bbox_max[0], dims[0])
    y = np.linspace(bbox_min[1], bbox_max[1], dims[1])
    z = np.linspace(bbox_min[2], bbox_max[2], dims[2])
    x, y, z = np.meshgrid(x, y, z, indexing='ij')

    grid_pts = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    grid_sdf = griddata(pts, sdf, grid_pts, method='linear', fill_value=np.max(sdf))
    grid_sdf = grid_sdf.reshape(dims)
 
    grid_sdf[0, :, :] = grid_sdf[-1, :, :] = grid_sdf[:, 0, :] = grid_sdf[:, -1, :] = grid_sdf[:, :, 0] = grid_sdf[:, :, -1] = np.max(sdf)
    
    ps_grid.add_scalar_quantity('sdf', grid_sdf, defined_on='nodes', 
                                enabled=True,
                                enable_isosurface_viz=True, 
                                isosurface_level=3.9e-2,  # 3.9e-2 is a good value for the bunny
                                isosurface_color=(0.2, 0.6, 0.8), 
                                enable_gridcube_viz=False)
   
    slice_plane = ps.add_scene_slice_plane()    
    original_mesh.set_ignore_slice_plane(slice_plane.get_name(), True) 

    ps.show()