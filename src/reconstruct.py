import numpy as np
import polyscope as ps
import gpytoolbox as gpy
from enum import Enum
from tqdm import tqdm
from scipy.interpolate import griddata

class ReconstructionMethod(Enum):
    MARCHING_CUBES = 1
    REACH_FOR_THE_SPHERES = 2
    ALL = 3

class Reconstructor:
    def __init__(self, pts, sdf, mesh_path):
        self.pts = pts
        self.sdf = sdf
        self.mesh_path = mesh_path
        self.grid_sdf = None
        self.ps_grid = None
        self.reconstructed_v_mc = None
        self.reconstructed_f_mc = None
        self.reconstructed_v_rfs = None
        self.reconstructed_f_rfs = None

    def marching_cubes(self, grid_res=128):
        print('running marching cubes...')
        bbox_min = np.min(self.pts, axis=0)
        bbox_max = np.max(self.pts, axis=0)
        bbox_range = bbox_max - bbox_min
        bbox_min -= 0.1 * bbox_range
        bbox_max += 0.1 * bbox_range

        dims = (grid_res, grid_res, grid_res)
        self.ps_grid = ps.register_volume_grid('sdf grid', dims, bbox_min, bbox_max)

        x = np.linspace(bbox_min[0], bbox_max[0], dims[0])
        y = np.linspace(bbox_min[1], bbox_max[1], dims[1])
        z = np.linspace(bbox_min[2], bbox_max[2], dims[2])
        x, y, z = np.meshgrid(x, y, z, indexing='ij')

        grid_pts = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
        self.grid_sdf = griddata(self.pts, self.sdf, grid_pts, method='linear', fill_value=np.max(self.sdf))
        self.grid_sdf = self.grid_sdf.reshape(dims)
 
        self.grid_sdf[0, :, :] = self.grid_sdf[-1, :, :] = self.grid_sdf[:, 0, :] = self.grid_sdf[:, -1, :] = self.grid_sdf[:, :, 0] = self.grid_sdf[:, :, -1] = np.max(self.sdf)

    def reach_for_spheres(self, num_iters=5):
        v_init, f_init = gpy.icosphere(2)
        v_init = gpy.normalize_points(v_init)

        sdf_func = lambda x: griddata(self.pts, self.sdf, x, method='linear', fill_value=np.max(self.sdf))

        for _ in tqdm(range(num_iters), desc='reach for the spheres iterations'):
            self.reconstructed_v_rfs, self.reconstructed_f_rfs = gpy.reach_for_the_spheres(
                self.pts, sdf_func, v_init, f_init
            )
            v_init, f_init = self.reconstructed_v_rfs, self.reconstructed_f_rfs

    def reconstruct(self, method: ReconstructionMethod, **kwargs):
        ps.init()

        v_orig, f_orig = gpy.read_mesh(self.mesh_path)
        v_orig = gpy.normalize_points(v_orig)

        if method == ReconstructionMethod.ALL:
            original_mesh = ps.register_surface_mesh('original mesh', v_orig, f_orig, smooth_shade=True)
            original_mesh.translate([-2, 0, 0])

            num_iters = kwargs.get('num_iters', 5)
            self.reach_for_spheres(num_iters)
            rfs_mesh = ps.register_surface_mesh('rfs mesh', 
                                                self.reconstructed_v_rfs, 
                                                self.reconstructed_f_rfs, 
                                                smooth_shade=True)
            rfs_mesh.translate([2, 0, 0])

            grid_res = kwargs.get('grid_res', 128)
            isosurface_level = kwargs.get('isosurface_level', 2.5e-2)
            self.marching_cubes(grid_res)
            self.ps_grid.add_scalar_quantity('sdf', self.grid_sdf, defined_on='nodes', 
                                        enabled=True,
                                        enable_isosurface_viz=True, 
                                        isosurface_level=isosurface_level,
                                        isosurface_color=(0.2, 0.6, 0.8), 
                                        enable_gridcube_viz=False)
            self.ps_grid.translate([0, 0, 0])            

        elif method == ReconstructionMethod.MARCHING_CUBES:
            original_mesh = ps.register_surface_mesh('original mesh', v_orig, f_orig, smooth_shade=True)
            original_mesh.translate([0, 0, 1])

            grid_res = kwargs.get('grid_res', 128)
            isosurface_level = kwargs.get('isosurface_level', 1e-6)
            
            self.marching_cubes(grid_res)
            self.ps_grid.add_scalar_quantity('sdf', self.grid_sdf, defined_on='nodes', 
                                        enabled=True,
                                        enable_isosurface_viz=True, 
                                        isosurface_level=isosurface_level,
                                        isosurface_color=(0.2, 0.6, 0.8), 
                                        enable_gridcube_viz=False)

        elif method == ReconstructionMethod.REACH_FOR_THE_SPHERES:
            original_mesh = ps.register_surface_mesh('original mesh', v_orig, f_orig, smooth_shade=True)
            original_mesh.translate([0, 0, 1])

            num_iters = kwargs.get('num_iters', 5)
            
            self.reach_for_spheres(num_iters)
            rfs_mesh = ps.register_surface_mesh('rfs mesh', 
                                                 self.reconstructed_v_rfs, 
                                                 self.reconstructed_f_rfs, 
                                                 smooth_shade=True)
            rfs_mesh.translate([0, 0, -1]) 

        slice_plane = ps.add_scene_slice_plane()
        
        original_mesh.set_ignore_slice_plane(slice_plane.get_name(), True)

        if method == ReconstructionMethod.REACH_FOR_THE_SPHERES or method == ReconstructionMethod.ALL:
            rfs_mesh.set_ignore_slice_plane(slice_plane.get_name(), True)

        ps.show()