from extract import *
from reconstruct import *
from modify import *

def main():
    mesh_path = 'src/data/bunny_mid.obj'
    # mod_mesh_path = modify_mesh(mesh_path, 75)

    pts, sdf = extract_sdf(SampleMethod.PROXIMITY, mesh_path, 
                           num_samples=1250000, batch_size=50000, sigma=0.1)
    
    reconstructor = Reconstructor(pts, sdf, mesh_path)

    reconstructor.reconstruct(ReconstructionMethod.ALL, num_iters=2)

if __name__ == '__main__':
    main()