# Redundant-SDFs

Incorporating redundancies into SDFs for better surface representations and evolution of it. Additionally, exploring how to better train neural networks to encode such redundant representations.
<br>See the details on our blog here: [https://summergeometry.org/sgi2024/redundant-sdfs/](https://summergeometry.org/sgi2024/redundant-sdfs/)

### SDF Based Reconstruction
We extract the SDF of a mesh, then perform existing algorithms for reconstruction over it.
<div align =center>
  <img src="https://github.com/user-attachments/assets/44538c85-8fa0-4d4f-b914-5018f546dbd6" alt="plt" style="width: 80%;"/>
<br> <b>From left to right:</b> Ground Truth, Marching Cubes, Reach for the Spheres, Reach for the Arcs
</div>

### PSR
Shooting random rays and finding their intersections with the surface using raymarching, then  performing Poisson Surface Reconstruction to create the final mesh.
<div align =center>
  <img src="https://github.com/user-attachments/assets/6a2bf581-70e7-4c0d-85a7-f4d0913fa465" alt="plt" style="width: 80%;"/>
<br> <b>From left to right:</b> Ground Truth, PSR Reconstruction
</div>

### VDF Based Reconstruction (Ours)
Reconstructions using Vector Distance Functions (VDFs) are similar to those using SDFs, except we have added a directional component. After creating a grid of points in a space, we can construct a vector stemming from each point that is both pointing in the direction that minimizes the distance to the mesh, as well as has the magnitude of the SDF at that point. In this way, the tip of each vector will lie on the real surface. From here, we can extract a point cloud of points on the surface and implement a reconstruction method.

<div align =center>
  <img src="https://github.com/user-attachments/assets/6330fb61-02af-4ab0-9166-6f8ada3d4427" alt="plt" style="width: 80%;"/>
<br> <b>From left to right:</b> Ground Truth, Gradient VDF Reconstruction, Barycentric VDF Reconstruction, One-Point-Per-Face VDF Reconstruction
</div>

# Acknowledgements
**Fellows:** Aniket Rajnish, Ashlyn Lee, Megan Grosse, Taylor
<br>**Mentors:** Ishit Mehta, Sina Nabizadeh
