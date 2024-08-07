from extract import *
from reconstruct import *
from modify import *

def main():
    mesh_path = 'src/data/bunny_mid.obj'
    # mod_mesh_path = modify_mesh(mesh_path, 75)

    pts, sdf = extract_sdf(SampleMethod.PROXIMITY, mesh_path, 
                           num_samples=12500, batch_size=500, sigma=0.1)
    
    reconstructor = Reconstructor(pts, sdf, mesh_path)

    reconstructor.reconstruct(ReconstructionMethod.REACH_FOR_THE_SPHERES, 
                              max_iter=1000000000000, tol=1e-2)

if __name__ == '__main__':
    main()