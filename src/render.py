import polyscope as ps
import numpy as np

def render_models(meshes, output_prefix='render'):
    ps.set_ground_plane_mode('shadow_only') 
    ps.set_view_projection_mode('orthographic')
    ps.set_shadow_darkness(.6)

    intrinsics = ps.CameraIntrinsics(
        fov_vertical_deg=75., 
        aspect=2.
    )

    camera_position = np.array([0, 2.5, 5.])
    look_at_point = np.array([0, 0., 0.])
    up_direction = np.array([0, 1., 0.])

    extrinsics = ps.CameraExtrinsics(
        root=camera_position,
        look_dir=look_at_point - camera_position,
        up_dir=up_direction
    )
    
    ps.set_view_camera_parameters(
        ps.CameraParameters(
            intrinsics=intrinsics, 
            extrinsics=extrinsics
        )
    )

    for edge_width in [0, .75]:
        for mesh in meshes:
            if mesh is not None:
                mesh.set_enabled(True)
                mesh.set_edge_width(edge_width)

        rand_id = np.random.randint(0, 1e6)

        ps.screenshot(f'src/data/img/{output_prefix}_{edge_width}_{rand_id}.png')

def cleanup(meshes):
    ps.set_ground_plane_mode('tile_reflection')
    ps.set_view_projection_mode('perspective')
    ps.set_shadow_darkness(0.25)

    for mesh in meshes:
        if mesh is not None:
            mesh.set_edge_width(0)

def render_imgs(meshes, output_prefix='render'):
    render_models(meshes, output_prefix)
    cleanup(meshes)