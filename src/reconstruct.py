import numpy as np
import polyscope as ps
import gpytoolbox as gpy
from enum import Enum
from tqdm import tqdm
from scipy.interpolate import griddata
from skimage import measure
from fitness import fitness

class ReconstructionMethod(Enum):
    MARCHING_CUBES = 1
    REACH_FOR_THE_SPHERES = 2
    REACH_FOR_THE_ARCS = 3
    ALL = 4

class Reconstructor:
    def __init__(self, pts, sdf, mesh_path):
        self.pts = pts
        self.sdf = sdf
        self.mesh_path = mesh_path
        self.grid_sdf = None
        self.init_recon_data()

    def init_recon_data(self):
        self.reconstructed_v_mc = None
        self.reconstructed_f_mc = None
        self.reconstructed_v_rfs = None
        self.reconstructed_f_rfs = None
        self.reconstructed_v_rfa = None
        self.reconstructed_f_rfa = None

    def marching_cubes(self, grid_res=128):
        print('performing marching cubes...')
        bbox_min = np.min(self.pts, axis=0)
        bbox_max = np.max(self.pts, axis=0)
        bbox_range = bbox_max - bbox_min
        bbox_min -= 0.1 * bbox_range
        bbox_max += 0.1 * bbox_range

        dims = (grid_res, grid_res, grid_res)
        x = np.linspace(bbox_min[0], bbox_max[0], dims[0])
        y = np.linspace(bbox_min[1], bbox_max[1], dims[1])
        z = np.linspace(bbox_min[2], bbox_max[2], dims[2])
        x, y, z = np.meshgrid(x, y, z, indexing='ij')

        grid_pts = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
        self.grid_sdf = griddata(self.pts, self.sdf, grid_pts, method='linear', fill_value=np.max(self.sdf))
        self.grid_sdf = self.grid_sdf.reshape(dims)

        verts, faces, _, _ = measure.marching_cubes(self.grid_sdf, level=0)
        
        verts = verts * (bbox_max - bbox_min) / (np.array(dims) - 1) + bbox_min
        
        self.reconstructed_v_mc, self.reconstructed_f_mc = verts, faces

    def reach_for_spheres(self, num_iters=5):
        print('performing reach for the spheres...')
        v_init, f_init = gpy.icosphere(2)
        v_init = gpy.normalize_points(v_init)

        sdf_func = lambda x: griddata(self.pts, self.sdf, x, method='linear', fill_value=np.max(self.sdf))

        for _ in tqdm(range(num_iters), desc='rfs iterations', unit='iter'):
            self.reconstructed_v_rfs, self.reconstructed_f_rfs = gpy.reach_for_the_spheres(
                self.pts, sdf_func, v_init, f_init
            )
            v_init, f_init = self.reconstructed_v_rfs, self.reconstructed_f_rfs

    def reach_for_arcs(self, **kwargs):
        print('performing reach for the arcs...')
        self.reconstructed_v_rfa, self.reconstructed_f_rfa = gpy.reach_for_the_arcs(
            self.pts, self.sdf,
            rng_seed=kwargs.get('rng_seed', 3452),
            fine_tune_iters=kwargs.get('fine_tune_iters', 3),
            batch_size=kwargs.get('batch_size', 10000),
            num_rasterization_spheres=kwargs.get('num_rasterization_spheres', 0),
            screening_weight=kwargs.get('screening_weight', 10.0),
            rasterization_resolution=kwargs.get('rasterization_resolution', None),
            max_points_per_sphere=kwargs.get('max_points_per_sphere', 3),
            n_local_searches=kwargs.get('n_local_searches', None),
            local_search_iters=kwargs.get('local_search_iters', 20),
            local_search_t=kwargs.get('local_search_t', 0.01),
            tol=kwargs.get('tol', 0.0001),
            clamp_value=kwargs.get('clamp_value', np.Inf),
            force_cpu=kwargs.get('force_cpu', False),
            parallel=kwargs.get('parallel', False),
            verbose=kwargs.get('verbose', False)
        )

    def remesh(self, v, f, target_len=None, num_iters=10):
        print('remeshing...')
        if target_len is None:
            target_len = np.mean(gpy.halfedge_lengths(v, f))
        return gpy.remesh_botsch(v, f, i=num_iters, h=target_len, project=True)

    def reconstruct(self, method: ReconstructionMethod, **kwargs):
        ps.init()

        v_orig, f_orig = gpy.read_mesh(self.mesh_path)
        v_orig = gpy.normalize_points(v_orig)

        original_mesh = ps.register_surface_mesh('ground truth', v_orig, f_orig, smooth_shade=True)
        original_mesh.translate([-3, 0, 0])

        if method in [ReconstructionMethod.ALL, ReconstructionMethod.MARCHING_CUBES]:
            grid_res = kwargs.get('grid_res', 128)
            self.marching_cubes(grid_res)
            v_mc_remeshed, f_mc_remeshed = self.remesh(self.reconstructed_v_mc, self.reconstructed_f_mc)
            mc_remeshed = ps.register_surface_mesh('mc', v_mc_remeshed, f_mc_remeshed, smooth_shade=True)
            mc_remeshed.translate([-1, 0, 0])
            mc_fitness = fitness(v_orig, v_mc_remeshed)
            print(f'Marching Cubes fitness: {mc_fitness}')

        if method in [ReconstructionMethod.ALL, ReconstructionMethod.REACH_FOR_THE_SPHERES]:
            num_iters = kwargs.get('num_iters', 5)
            self.reach_for_spheres(num_iters)
            v_rfs_remeshed, f_rfs_remeshed = self.remesh(self.reconstructed_v_rfs, self.reconstructed_f_rfs)
            rfs_remeshed = ps.register_surface_mesh('rfs', v_rfs_remeshed, f_rfs_remeshed, smooth_shade=True)
            rfs_remeshed.translate([1, 0, 0])
            rfs_fitness = fitness(v_orig, v_rfs_remeshed)
            print(f'Reach for the Spheres fitness: {rfs_fitness}')

        if method in [ReconstructionMethod.ALL, ReconstructionMethod.REACH_FOR_THE_ARCS]:
            self.reach_for_arcs(**kwargs)
            v_rfa_remeshed, f_rfa_remeshed = self.remesh(self.reconstructed_v_rfa, self.reconstructed_f_rfa)
            rfa_remeshed = ps.register_surface_mesh('rfa', v_rfa_remeshed, f_rfa_remeshed, smooth_shade=True)
            rfa_remeshed.translate([3, 0, 0])
            rfa_fitness = fitness(v_orig, v_rfa_remeshed)
            print(f'Reach for the Arcs fitness: {rfa_fitness}')

        slice_plane = ps.add_scene_slice_plane()
        original_mesh.set_ignore_slice_plane(slice_plane.get_name(), True)

        for mesh in [mc_remeshed, rfs_remeshed, rfa_remeshed]:
            if mesh:
                mesh.set_ignore_slice_plane(slice_plane.get_name(), True)

        ps.show()