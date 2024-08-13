from extract import *
from reconstruct import *
from modify import *
from enums import *

def main():
    mesh_path = 'src/data/model/bunny.obj'
    # mod_mesh_path = modify_mesh(mesh_path, 25)

    # pts_sdf, sdf_sdf = extract_sdf(SampleMethod.RANDOM, mesh_path, 
    #                                num_samples=125000, batch_size=5000, sigma=0.1,
    #                                sdf_method=SDFMethod.SAMPLE)
    
    # sdf_reconstructor = SDFReconstructor(pts_sdf, sdf_sdf, mesh_path)
    # sdf_reconstructor.reconstruct(ReconstructionMethod.ALL, num_iters=2, render=True)

    # grid_res = 50
    # pts_vdf, sdf_vdf = extract_sdf(SampleMethod.RANDOM, mesh_path, 
    #                                sdf_method=SDFMethod.GRID, grid_res=grid_res)

    pts_vdf, sdf_vdf = extract_sdf(SampleMethod.PROXIMITY, mesh_path, 
                                   sdf_method=SDFMethod.GRID, grid_res=50)

    vdf_reconstructor = VDFReconstructor(mesh_path, grid_res=50, sdf=sdf_vdf, pts=pts_vdf)
    vdf_reconstructor.reconstruct()

if __name__ == '__main__':
    main()