# %%
import ipyenv as uu
uu.enpy("test")
uu.chdir(__file__)

import open3d as o3d

# %% pcd点云显示
pcd = o3d.io.read_point_cloud(uu.rpath("img/bunny.pcd"))
print(pcd)
o3d.visualization.draw_geometries([pcd])

# %% obj面片显示
textured_mesh = o3d.io.read_triangle_mesh('xxx.obj')
print(textured_mesh)
textured_mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([textured_mesh], window_name="Open3D1")

# %% obj顶点显示
pcobj = o3d.geometry.PointCloud()
pcobj.points = o3d.utility.Vector3dVector(textured_mesh.vertices)
o3d.visualization.draw_geometries([pcobj], window_name="Open3D2")

# %% obj顶点转array
import numpy as np

textured_pc = np.asarray(textured_mesh.vertices)
print(textured_pc)
