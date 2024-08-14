from extract import extract_sdf
from reconstruct import *
from enums import *

def run_sdf_reconstruction(mesh_path, sdf_sample_method = SampleMethod.PROXIMITY, render=False,
                           sdf_reconstruction_method=SDFReconstructionMethod.ALL,
                           grid_res=128, num_samples=125000, batch_size=5000, sigma=0.1):
    
    pts_sdf, sdf_sdf = extract_sdf(sdf_sample_method, mesh_path, 
                                   num_samples=num_samples, batch_size=batch_size, sigma=sigma,
                                   sdf_method=SDFMethod.SAMPLE)
    
    sdf_reconstructor = SDFReconstructor(pts_sdf, sdf_sdf, mesh_path)

    sdf_reconstructor.reconstruct(sdf_reconstruction_method, 
                                  num_iters=2, render=render, 
                                  grid_res=grid_res)

def run_vdf_reconstruction(mesh_path, sdf_sample_method = SampleMethod.PROXIMITY, grid_res=50, 
                           vdf_reconstruction_method=VDFReconstructionMethod.GRADIENT, render=False):
    
    pts_vdf, sdf_vdf = extract_sdf(sdf_sample_method, mesh_path, 
                                   sdf_method=SDFMethod.GRID, grid_res=grid_res)

    vdf_reconstructor = VDFReconstructor(mesh_path, grid_res=grid_res, 
                                         sdf=sdf_vdf, pts=pts_vdf)

    vdf_reconstructor.reconstruct(method=vdf_reconstruction_method, render=render)

def main():
    mesh_path = 'src/data/model/bunny.obj'

    render = True # set to True to render the results

    sdf_sample_method = SampleMethod.PROXIMITY # SampleMethod.RANDOM for random sampling
                                               # PROXIMITY samples points near the mesh surface and generally gives better results

    # uncomment the following lines to run SDF reconstruction
    # run_sdf_reconstruction(mesh_path, grid_res=128, 
    #                        num_samples=125000, batch_size=5000, render = render,
    #                        sigma=0.1, sdf_sample_method=sdf_sample_method,
    #                        sdf_reconstruction_method=SDFReconstructionMethod.ALL) # change sdf_reconstruction_method to view results of individual methods

    run_vdf_reconstruction(mesh_path, grid_res=50, sdf_sample_method=sdf_sample_method, render = render,
                           vdf_reconstruction_method=VDFReconstructionMethod.BARYCENTRIC) # change vdf_reconstruction_method to GRADIENT to view results of gradient method   
                                                                                          # BAYCENTRIC generally gives better results                         

if __name__ == '__main__':
    main()