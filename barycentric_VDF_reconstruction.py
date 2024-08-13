###this is the code for an implementation of the vector distance function method with poisson surface reconstruction
###it uses the barycentric coordinates from signed_distance instead of gradient
###you can change the resolution n (now at 20) and the object (now bunny.obj) and the filename (at bottom) if you'd like
###this code outputs a download link to your reconstructed mesh, so it can be visualized later

import gpytoolbox as gpy
import polyscope as ps
import numpy as np
import scipy as sp

###obtain the sdf at a set of points (the grid, from which the sdf is sampled)
#building a list of vertices distributed grid-wise in a space
n=20
x=np.linspace(-1,1,n)
y=np.linspace(-1,1,n)
z=np.linspace(-1,1,n)
X,Y,Z=np.meshgrid(x,y,z)
grid_vertices=np.c_[X.flatten(),Y.flatten(),Z.flatten()]
#reading and scaling mesh
mesh = trimesh.load_mesh("bunny.obj")
V, F = mesh.vertices, mesh.faces
V = gpy.normalize_points(V)
#obtaining SDF
signed_distance,ind,b = gpy.signed_distance(grid_vertices,V,F)

def barycentric_to_cartesian(barycentric_coords, faces, vertices):
    cartesian_coords = np.zeros((barycentric_coords.shape[0], 3))
    # Function to convert face indices to vertex coordinates
    def faces_to_vertex_coordinates(faces, F, vertices):
        # Extract vertex indices for each face from F using faces array
        face_vertex_indices = F[faces]
        # Retrieve the Cartesian coordinates for each vertex of each face
        return vertices[face_vertex_indices]
    
    # Convert faces to vertex coordinates
    face_vertex_coordinates = faces_to_vertex_coordinates(faces, F, V)
    for i, (bary_coords, face) in enumerate(zip(barycentric_coords, faces)):

        # Get the vertices of the current face
        v0, v1, v2 = face_vertex_coordinates[i]
        # Compute Cartesian coordinates
        cartesian_coords[i] = (bary_coords[0] * v0 +
                               bary_coords[1] * v1 +
                               bary_coords[2] * v2)
    return cartesian_coords
cartesian_coordinates = barycentric_to_cartesian(b,ind,V)

#visualize
#ps.init()
#show the original mesh
#ps.register_surface_mesh("original_mesh",V,F)
# Register the point cloud
#ps_net=ps.register_point_cloud("ind in cartesian cooradinates", cartesian_coordinates)
# visualize the VDF vectors
#ps_net.add_vector_quantity("vectors", vector_distance, "ambient", enabled=True)
#ps.show()

normals = grid_vertices - cartesian_coordinates

# Calculate the magnitudes of each normal vector
magnitudes = np.linalg.norm(normals, axis=1)

# Normalize the normals by dividing each normal by its magnitude
normals = normals / magnitudes[:, np.newaxis]

new_V, new_F = gpy.point_cloud_to_mesh(cartesian_coordinates, normals, method='PSR')

def save_obj(filename, vertices, faces):
    """
    Save vertices and faces to an OBJ file.
    
    Parameters:
    - filename: Path to the output OBJ file.
    - vertices: (n, 3) numpy array of vertex positions.
    - faces: (m, 3) numpy array of face indices (1-based).
    """
    with open(filename, 'w') as file:
        # Write vertices
        for v in vertices:
            file.write(f"v {v[0]} {v[1]} {v[2]}\n")
        
        # Write faces
        for face in faces:
            # OBJ file format is 1-based indexing, adjust from 0-based
            face_str = ' '.join(str(idx + 1) for idx in face)
            file.write(f"f {face_str}\n")


# Save to an OBJ file
save_obj('my_VDF_reconstructed_mesh.obj', new_V, new_F)


from IPython.display import FileLink

# Create a link to download the file
FileLink('my_VDF_reconstructed_mesh.obj')
