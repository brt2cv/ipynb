
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
try:
    from open3d import JVisualizer
except ImportError:
    from open3d.j_visualizer import JVisualizer

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

# %% 用户交互：裁剪

# 1) Press 'Y' twice to align geometry with negative direction of y-axis
# 2) Press 'K' to lock screen and to switch to selection mode
# 3) Drag for rectangle selection, or use ctrl + left click for polygon selection
# 4) Press 'C' to get a selected geometry and to save it
# 5) Press 'F' to switch to freeview mode

pcd = o3d.io.read_point_cloud(uu.rpath("open3d_/bunny.ply"))
o3d.visualization.draw_geometries_with_editing([pcd])

# %% 用户交互：手动ICP配准

# 1) Please pick at least three correspondences using [shift + left click]
#    Press [shift + right click] to undo point picking
# 2) After picking points, press 'Q' to close the window

import numpy as np
import copy

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def pick_points(pcd):
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    return vis.get_picked_points()

print("Demo for manual ICP")
source = o3d.io.read_point_cloud("/d/Home/workspace/MCAD/vision3D//ICP/cloud_bin_0.pcd")
target = o3d.io.read_point_cloud("/d/Home/workspace/MCAD/vision3D/ICP/cloud_bin_2.pcd")
print("Visualization of two point clouds before manual alignment")
draw_registration_result(source, target, np.identity(4))

# pick points from two point clouds and builds correspondences
picked_id_source = pick_points(source)
picked_id_target = pick_points(target)
assert (len(picked_id_source) >= 3 and len(picked_id_target) >= 3)
assert (len(picked_id_source) == len(picked_id_target))
corr = np.zeros((len(picked_id_source), 2))
corr[:, 0] = picked_id_source
corr[:, 1] = picked_id_target

# estimate rough transformation using correspondences
print("Compute a rough transform using the correspondences given by user")
p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
trans_init = p2p.compute_transformation(source, target,
                                        o3d.utility.Vector2iVector(corr))

# point-to-point ICP for refinement
print("Perform point-to-point ICP refinement")
threshold = 0.03  # 3cm distance threshold
reg_p2p = o3d.pipelines.registration.registration_icp(
    source, target, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint())
draw_registration_result(source, target, reg_p2p.transformation)

# %% 深度图
import ipyenv as uu
uu.enpy("img")

import cv2
import numpy as np
import pycv.opencv as cv

uu.chdir(__file__)

# %%
img_dep = cv.imread(uu.rpath("open3d_/box.png"))

# %% 深度图转点云
def depth_to_pcloud(img_dep, tab_x, tab_y):
    pc = np.zeros((np.size(img_dep), 3))
    pc[:, 0] = img_dep.flatten() * tab_x
    pc[:, 1] = img_dep.flatten() * tab_y
    pc[:, 2] = img_dep.flatten()
    return pc

# 防止出现除数为零的情况
ToF_CAM_EPS = 1.0e-16
## 计算速算表
# tab_x[u,v]=(u-u0)*fx
# tab_y[u,v]=(v-v0)*fy
# 通过速算表，计算像素位置(u,v)对应的物理坐标(x,y,z)
# x=tab_x[u,v]*z, y=tab_y[u,v]*z
# 注意：为了方便使用，tab_x和tab_y矩阵被拉直成向量存放
def gen_tab(cx,cy,fx,fy,img_hgt,img_wid):
    u = (np.arange(IMG_WID) - cx) / fx
    v = (np.arange(img_hgt) - cy) / fy
    tab_x = np.tile(u, img_hgt)
    tab_y = np.repeat(v, img_wid)
    return tab_x,tab_y

# 深度图转点云
tab_x,tab_y = gen_tab(160,120,180,180,240,320)
pointcloud = depth_to_pcloud(img_dep, tab_x, tab_y)

# %% 保存pcd文件
with open(uu.rpath("open3d_/box2pcd2.pcd"), "w") as handle:
    # 得到点云点数
    point_num = pointcloud.shape[0]
    # pcd头部（重要）
    handle.write(
f'# .PCD v0.7 - Point Cloud Data file format\n\
VERSION 0.7\n\
FIELDS x y z\n\
SIZE 4 4 4\n\
TYPE F F F\n\
COUNT 1 1 1\n\
WIDTH {point_num}\n\
HEIGHT 1\n\
VIEWPOINT 0 0 0 1 0 0 0\n\
POINTS {point_num}\n\
DATA ascii')
    # 依次写入点
    for i in range(point_num):
        handle.write(f"\n{pointcloud[i, 0]} {pointcloud[i, 1]} {str(pointcloud[i, 2]}")

# %% 点云转深度图
def pcloud_to_depth(pc,cx,cy,fx,fy,img_hgt,img_wid):
    # 计算点云投影到传感器的像素坐标
    x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
    u = (np.round(x * fx / (z + ToF_CAM_EPS) + cx)).astype(int)
    v = (np.round(y * fy / (z + ToF_CAM_EPS) + cy)).astype(int)

    valid = np.bitwise_and(np.bitwise_and((u >= 0), (u <img_wid)), np.bitwise_and((v >= 0), (v <img_hgt)))
    u_valid = u[valid]
    v_valid = v[valid]
    z_valid = z[valid]

    img_dep = np.zeros((img_hgt, img_wid))
    for ui, vi, zi in zip(u_valid, v_valid, z_valid):
        # 替代0像素值点
        img_dep[vi][ui] = max(img_dep[vi][ui], zi)
    return img_dep

img_cvt = pcloud_to_depth(pointcloud, 160,120,180,180,240,320)
img_cvt = img_cvt.astype(np.uint8)
uu.imshow(img_cvt)

# %% 深度图时域滤波
list_img = []
for path in ["path_img_0", "path_img_1", "path_img_2"]:
    img = cv.imread(path)
    list_img.append(img)

count = len(list_img)
buf_dep = np.zeros((*list_img[0].shape, count))
for idx in range(count):
    buf_dep[:, :, idx] = list_img[idx]

img_sum = np.sum(buf_dep,axis=2)
img_max = np.max(buf_dep,axis=2)
img_min = np.min(buf_dep,axis=2)

img_filter = img_sum - img_max - img_min
uu.imshow(img_filter)

# %%
