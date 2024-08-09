from extract import *
from reconstruct import *
from modify import *
from enums import ReconstructionMethod, SampleMethod

def main():
    mesh_path = 'src/data/model/bunny_mid.obj'
    # mod_mesh_path = modify_mesh(mesh_path, 25)
    pts, sdf = extract_sdf(SampleMethod.PROXIMITY, mesh_path, 
                           num_samples=125000, batch_size=5000, sigma=0.1)
    
    reconstructor = Reconstructor(pts, sdf, mesh_path)
    reconstructor.reconstruct(ReconstructionMethod.ALL, num_iters=2, render=True)

if __name__ == '__main__':
    main()