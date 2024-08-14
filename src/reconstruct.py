import numpy as np
import polyscope as ps
import gpytoolbox as gpy
from tqdm import tqdm
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from skimage import measure
from fitness import fitness
from render import render_imgs
from enums import SDFReconstructionMethod, VDFReconstructionMethod

class SDFReconstructor:
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

    def reconstruct(self, method: SDFReconstructionMethod, render=False, **kwargs):
        ps.init()

        v_orig, f_orig = gpy.read_mesh(self.mesh_path)
        v_orig = gpy.normalize_points(v_orig)

        original_mesh = ps.register_surface_mesh('ground truth', v_orig, f_orig, smooth_shade=True)
        original_mesh.translate([-3, 0, 0])

        meshes = [original_mesh]

        if method in [SDFReconstructionMethod.ALL, SDFReconstructionMethod.MARCHING_CUBES]:
            grid_res = kwargs.get('grid_res', 128)
            self.marching_cubes(grid_res)
            v_mc_remeshed, f_mc_remeshed = self.remesh(self.reconstructed_v_mc, self.reconstructed_f_mc)
            mc_remeshed = ps.register_surface_mesh('mc', v_mc_remeshed, f_mc_remeshed, smooth_shade=True)
            mc_remeshed.translate([-1, 0, 0])
            mc_fitness = fitness(v_orig, v_mc_remeshed)
            print(f'marching cubes fitness: {mc_fitness}')
            meshes.append(mc_remeshed)

        if method in [SDFReconstructionMethod.ALL, SDFReconstructionMethod.REACH_FOR_THE_SPHERES]:
            num_iters = kwargs.get('num_iters', 5)
            self.reach_for_spheres(num_iters)
            v_rfs_remeshed, f_rfs_remeshed = self.remesh(self.reconstructed_v_rfs, self.reconstructed_f_rfs)
            rfs_remeshed = ps.register_surface_mesh('rfs', v_rfs_remeshed, f_rfs_remeshed, smooth_shade=True)
            rfs_remeshed.translate([1, 0, 0])
            rfs_fitness = fitness(v_orig, v_rfs_remeshed)
            print(f'reach for the spheres fitness: {rfs_fitness}')
            meshes.append(rfs_remeshed)

        if method in [SDFReconstructionMethod.ALL, SDFReconstructionMethod.REACH_FOR_THE_ARCS]:
            self.reach_for_arcs(**kwargs)
            v_rfa_remeshed, f_rfa_remeshed = self.remesh(self.reconstructed_v_rfa, self.reconstructed_f_rfa)
            rfa_remeshed = ps.register_surface_mesh('rfa', v_rfa_remeshed, f_rfa_remeshed, smooth_shade=True)
            rfa_remeshed.translate([3, 0, 0])
            rfa_fitness = fitness(v_orig, v_rfa_remeshed)
            print(f'reach for the arcs fitness: {rfa_fitness}')
            meshes.append(rfa_remeshed)

        slice_plane = ps.add_scene_slice_plane()
        for mesh in meshes:
            mesh.set_ignore_slice_plane(slice_plane.get_name(), True)

        if render:
            render_imgs(meshes)
        
        ps.show()
        
class VDFReconstructor:
    def __init__(self, mesh_path, grid_res=50, sdf=None, pts=None):
        self.mesh_path = mesh_path
        self.grid_res = grid_res
        self.grid_shape = (grid_res, grid_res, grid_res)
        self.grid_vertices = pts
        self.signed_distance = sdf

        self.init_recon_data()

    def init_recon_data(self):        
        self.V = None
        self.F = None
        self.vector_distance = None
        self.vdf_pts = None
        self.face_indices = None
        self.barycentric_coords = None

    def load_mesh(self):
        self.V, self.F = gpy.read_mesh(self.mesh_path)
        self.V = gpy.normalize_points(self.V)

    def compute_gradient(self):
        def finite_difference(f, axis):
            f_pos = np.roll(f, shift=-1, axis=axis)
            f_neg = np.roll(f, shift=1, axis=axis)
            f_pos[0] = f[0]  # neumann boundary condition
            f_neg[-1] = f[-1]
            return (f_pos - f_neg) / 2.0
        
        distance_grid = self.signed_distance.reshape(self.grid_shape)
        gradients = np.zeros((np.prod(self.grid_shape), 3))
        for dim in range(3):
            grad = finite_difference(distance_grid, axis=dim).flatten()
            gradients[:, dim] = grad
        
        magnitudes = np.linalg.norm(gradients, axis=1, keepdims=True)
        magnitudes = np.clip(magnitudes, a_min=1e-10, a_max=None)
        normalized_gradients = gradients / magnitudes

        signed_distance_reshaped = self.signed_distance.reshape(-1, 1)
        self.vector_distance = -1 * normalized_gradients * signed_distance_reshaped
        self.vector_distance[:, [0, 1, 2]] = self.vector_distance[:, [1, 0, 2]]

    def compute_barycentric(self):
        self.signed_distance, self.face_indices, self.barycentric_coords = gpy.signed_distance(self.grid_vertices, self.V, self.F)
        self.vdf_pts = self.barycentric_to_cartesian()
        self.vector_distance = self.grid_vertices - self.vdf_pts
        magnitudes = np.linalg.norm(self.vector_distance, axis=1, keepdims=True)
        self.vector_distance = self.vector_distance / magnitudes

    def barycentric_to_cartesian(self):
        cartesian_coords = np.zeros((self.barycentric_coords.shape[0], 3))
        face_vertex_coordinates = self.V[self.F[self.face_indices]]
        for i, (bary_coords, face_vertices) in enumerate(zip(self.barycentric_coords, face_vertex_coordinates)):
            cartesian_coords[i] = np.dot(bary_coords, face_vertices)
        return cartesian_coords

    def compute_surface_points(self):
        self.vdf_pts = self.grid_vertices + self.vector_distance * self.signed_distance.reshape(-1, 1)

    def reconstruct(self, method: VDFReconstructionMethod, render=False):
        self.load_mesh()

        if method == VDFReconstructionMethod.GRADIENT:            
            self.compute_gradient()
            self.compute_surface_points()
        elif method == VDFReconstructionMethod.BARYCENTRIC:
            self.compute_barycentric()

        self.visualize(method, render)

    def visualize(self, method: VDFReconstructionMethod, render=False):
        ps.init()
        
        original_mesh = ps.register_surface_mesh("original_mesh", self.V, self.F, smooth_shade=True)
        original_mesh.translate([-1.5, 0, 0])
        
        meshes = [original_mesh]

        if method == VDFReconstructionMethod.GRADIENT:
            ps_net = ps.register_point_cloud("vdf", self.grid_vertices, enabled=False)
            ps_net.add_vector_quantity("vectors", self.vector_distance, "ambient")
            vdf_cloud = ps.register_point_cloud("vdf_pts", self.vdf_pts, radius=0.01)
            vdf_cloud.translate([1.5, 0, 0])

        elif method == VDFReconstructionMethod.BARYCENTRIC:
            cartesian_cloud = ps.register_point_cloud("vdf_pts", self.vdf_pts, radius=0.01)
            cartesian_cloud.translate([1.5, 0, 0])

        slice_plane = ps.add_scene_slice_plane()
        for mesh in meshes:
            mesh.set_ignore_slice_plane(slice_plane.get_name(), True)

        if render:
            render_imgs(meshes)
        
        ps.show()

class RayReconstructor:
    def __init__(self, mesh_path, num_rays=100000, bounds=(-1, 1)):
        self.mesh_path = mesh_path
        self.num_rays = num_rays
        self.bounds = bounds
        
        self.V = None
        self.F = None
        self.intersection_points = None
        self.intersection_normals = None

    def load_mesh(self):
        self.V, self.F = gpy.read_mesh(self.mesh_path)
        self.V = gpy.normalize_points(self.V)

    def shoot_random_rays(self):
        print('Shooting random rays...')
        origins = np.random.uniform(self.bounds[0], self.bounds[1], (self.num_rays, 3))
        directions = np.random.randn(self.num_rays, 3)
        directions /= np.linalg.norm(directions, axis=1)[:, np.newaxis]

        intersection_points = []
        intersection_normals = []

        for origin, direction in tqdm(zip(origins, directions), total=self.num_rays, desc="Shooting rays"):
            t = 0
            point = origin
            while t < 2:  # Limit the ray length to avoid infinite loops
                sdf_value, _, normal = gpy.signed_distance(point.reshape(1, -1), self.V, self.F)
                if abs(sdf_value[0]) < 1e-4:  # We've hit the surface
                    intersection_points.append(point)
                    intersection_normals.append(normal[0])
                    break
                point += direction * sdf_value[0]
                t += abs(sdf_value[0])

        self.intersection_points = np.array(intersection_points)
        self.intersection_normals = np.array(intersection_normals)

        print(f"Found {len(self.intersection_points)} intersection points.")

    def reconstruct(self, render=False):
        self.load_mesh()
        self.shoot_random_rays()
        self.visualize(render)

    def visualize(self, render=False):
        ps.init()
        
        original_mesh = ps.register_surface_mesh("original_mesh", self.V, self.F, smooth_shade=True)
        original_mesh.translate([-1.5, 0, 0])

        meshes = [original_mesh]
        
        intersection_cloud = ps.register_point_cloud("intersection_points", self.intersection_points, enabled=True)
        intersection_cloud.add_vector_quantity("normals", self.intersection_normals, enabled=True)
        intersection_cloud.translate([1.5, 0, 0])        

        if render:
            render_imgs(meshes)
        
        ps.show()