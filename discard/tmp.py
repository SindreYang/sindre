import open3d as o3d
import numpy as np
print(o3d)
mesh1 = o3d.t.geometry.TriangleMesh.create_torus()
#print(mesh1.cuda())
grid_coords = np.stack(np.meshgrid(*3*[np.linspace(-2,2,num=64, dtype=np.float32)], indexing='ij'), axis=-1)

scene = o3d.t.geometry.RaycastingScene()
scene.add_triangles(mesh1)
sdf = scene.compute_signed_distance(grid_coords)
mesh2 = o3d.t.geometry.TriangleMesh.create_isosurfaces(sdf.cuda())

# Flip the triangle orientation for SDFs with negative values as "inside" and positive values as "outside"
mesh2.triangle.indices = mesh2.triangle.indices[:,[2,1,0]]

o3d.visualization.draw(mesh2)