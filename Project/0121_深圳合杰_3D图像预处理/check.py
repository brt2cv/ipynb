
# %%
import ipyenv as uu
uu.enpy("test")
uu.chdir(__file__)

import numpy as np
import open3d as o3d

# %% 工业检测
path_ply = "/d/Home/workspace/MCAD/vision3D/hv3D/12.ply"
pcd = o3d.io.read_point_cloud(path_ply)
o3d.visualization.draw_geometries([pcd])
print(pcd)  # 分析当前模型的参数信息

# %% 下采样
downpcd = pcd.voxel_down_sample(voxel_size=0.05)
print(downpcd)  # 分析当前模型的参数信息
o3d.visualization.draw_geometries([downpcd], window_name='Open3D downSample', width=1920, height=1080, left=50, top=50,
                                  point_show_normal=False, mesh_show_wireframe=False, mesh_show_back_face=False)

# %% 重新计算平面法线
downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
o3d.visualization.draw_geometries([downpcd], window_name='Open3D downSample Normals',
                                  point_show_normal=True,
                                  mesh_show_wireframe=False,
                                  mesh_show_back_face=False)

# %% 离群点去除
def display_inlier_outlier(pcd, idx):
    inlier_pcd = pcd.select_by_index(idx)
    outlier_pcd = pcd.select_by_index(idx, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_pcd.paint_uniform_color([1, 0, 0])
    inlier_pcd.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_pcd, outlier_pcd], window_name='Open3D Removal Outlier', width=1920,
                                      height=1080, left=50, top=50, point_show_normal=False, mesh_show_wireframe=False,
                                      mesh_show_back_face=False)

cl, idx = downpcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
display_inlier_outlier(downpcd, idx)

downpcd_inlier_cloud = downpcd.select_by_index(idx)
print(downpcd_inlier_cloud)
# o3d.visualization.draw_geometries([downpcd_inlier_cloud])

# %% 地平面提取: Plane Segmentation
plane_model, inliers = downpcd_inlier_cloud.segment_plane(distance_threshold=0.01,
                                             ransac_n=5,
                                             num_iterations=10000)
[a, b, c, d] = plane_model
print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
# Plane equation: 0.02x + 0.00y + 1.00z + 0.10 = 0

# %% 区分显示地平面与其他区域
inlier_cloud = downpcd_inlier_cloud.select_by_index(inliers)
print('----inlier_cloud: ', inlier_cloud.points)
inlier_cloud.paint_uniform_color([1, 0, 0])

outlier_cloud = downpcd_inlier_cloud.select_by_index(inliers, invert=True)
print('----outlier_cloud: ', outlier_cloud.points)
# o3d.visualization.draw_geometries([inlier_cloud])  # 单独显示地平面
o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud], window_name='Open3D Plane Model', width=1920,
                                  height=1080, left=50, top=50, point_show_normal=False, mesh_show_wireframe=False,
                                  mesh_show_back_face=False)

# %% ~~ DBSCAN 聚类群集 ~~
import matplotlib.pyplot as plt

with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(
        downpcd.cluster_dbscan(eps=0.01, min_points=10, print_progress=True))

max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
downpcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
o3d.visualization.draw_geometries([downpcd],
                                  zoom=0.455,
                                  front=[-0.4999, -0.1659, -0.8499],
                                  lookat=[2.1813, 2.0619, 2.0999],
                                  up=[0.1204, -0.9852, 0.1215])
