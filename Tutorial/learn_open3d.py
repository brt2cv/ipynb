
# %%
import ipyenv as uu
uu.enpy("test")
uu.chdir(__file__)

import numpy as np
import open3d as o3d

# %% 载入pcd模型 & 显示
# pcd = o3d.io.read_point_cloud(uu.rpath("open3d_/bunny_simp.pcd"))
pcd = o3d.io.read_point_cloud(uu.rpath("open3d_/bunny.ply"))
print(pcd)

# %% 在独立的窗口中显示模型
o3d.visualization.draw_geometries([pcd])

# %% 在jupyter中显示模型
from open3d import JVisualizer
# from open3d.j_visualizer import JVisualizer

visualizer = JVisualizer()
visualizer.add_geometry(pcd)
visualizer.show()

# %% 上色
pcd.paint_uniform_color([255,0,0])
o3d.visualization.draw_geometries([pcd])

# %% 获取pcd模型的点坐标
arr_pcd = np.asarray(pcd.points)
print(">>>", type(arr_pcd), arr_pcd.dtype)

# %% 体素降采样
# 1. 把点云装进体素网格
# 2. 把每个被占据的体素中的点做平均，取一个精确的点

downpcd = pcd.voxel_down_sample(voxel_size=0.05)  # voxel_size越大，保留点越少
o3d.visualization.draw_geometries([downpcd])

# %% 检索估计的顶点法线
print(">>>", np.asarray(pcd.normals))

# %%
aabb = pcd.get_axis_aligned_bounding_box()
aabb.color = (1,0,0)

obb = pcd.get_oriented_bounding_box()
obb.color = (0,1,0)

o3d.visualization.draw_geometries([pcd, aabb, obb])

print("corner_points:", np.asarray(aabb.get_box_points()))  # bounding的8个角点坐标
print("volume:", aabb.volume())

# %% 凸包（三角网格）
mesh = o3d.io.read_triangle_mesh(uu.rpath("open3d_/bunny.ply"))

pcl = mesh.sample_points_poisson_disk(number_of_points=2000)
hull, _ = pcl.compute_convex_hull()
hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
hull_ls.paint_uniform_color((1, 0, 0))
o3d.visualization.draw_geometries([pcl, hull_ls])

#####################################################################

# %% 体素化
# 点云和三角网格是一种十分灵活的，但是不规则的几何类型。
# 体素网格是通过规则的3D网格来表示的另一种3D几何类型，并且它可以看作是2D像素在3D上的对照物。
# Open3d中的VoxelGrid几何类型能够被用来处理体素网格数据。

# %% 从三角网格体素化
# 它返回一个体素网格，其中所有与三角形相交的网格被设置为1，其余的设置为0。其中voxel_zie参数是用来设置网格分辨率。
mesh = o3d.io.read_triangle_mesh(uu.rpath("open3d_/bunny.ply"))
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh])
# %%
mesh.scale(1 / np.max(mesh.get_max_bound() - mesh.get_min_bound()), center=mesh.get_center())
o3d.visualization.draw_geometries([mesh])

voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=0.05)
o3d.visualization.draw_geometries([voxel_grid])

# %% 从点云模型体素化
# 如果点云中至少有一个点在体素网格内，则该网格被占用。颜色表示的是该体素中点的平均值。参数voxel_size用来定义网格分辨率。
pcd.scale(1 / np.max(pcd.get_max_bound() - pcd.get_min_bound()), center=pcd.get_center())
N = np.asarray(pcd.points).shape[0]
pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0,1,size=(N,3)))
o3d.visualization.draw_geometries([pcd])

voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.05)
o3d.visualization.draw_geometries([voxel_grid])

# %% 另存模型
o3d.io.write_point_cloud(uu.rpath("open3d_/bunny_simp.ply"), pcd)
o3d.io.write_triangle_mesh("copy_of_pcd.ply", mesh)
