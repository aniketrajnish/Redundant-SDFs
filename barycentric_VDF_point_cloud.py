###best method so far

###this one uses the barycentric coordinates from signed_distance instead of gradient

###this is the code code for an implementation of the vector distance function method (this part gives the resultant point cloud with visualiztion, but not surface reconstruction)

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
V,F = gpy.read_mesh("bunny.obj")
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
ps.init()
#show the original mesh
ps.register_surface_mesh("original_mesh",V,F)
# Register the point cloud
ps_net=ps.register_point_cloud("ind in cartesian cooradinates", cartesian_coordinates)
# visualize the VDF vectors
#ps_net.add_vector_quantity("vectors", vector_distance, "ambient", enabled=True)
ps.show()
