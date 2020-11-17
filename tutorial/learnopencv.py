# http://codec.wang/#/opencv/
# https://github.com/CodecWang/Blog/tree/master/docs/opencv
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html
# https://www.learnopencv.com/blob-detection-using-opencv-python-c/

# %%
import util as uu
uu.enpy("opencv")
import ocv as cv

import numpy as np
import math

# %%
uu.reload(cv)

#####################################################################
# %% 手动实现canny边缘检测
# http://codec.wang/#/opencv/basic/11-edge-detection

path_shapes = "tutorial/img/handwriting.jpg"
im_word = cv.imread(path_shapes)
uu.imshow(im_word)

# %% 高斯模糊处理
im_gau = cv.gaussian(im_word, 3)
uu.imshow(im_gau)

# %%
im_bin = cv.threshold_otsu(im_gau)
uu.imshow(im_bin)

# %% 图像梯度、梯度幅值、梯度方向计算
g_x = cv.sobel(im_bin, 1, 0)
g_y = cv.sobel(im_bin, 0, 1)

# %% 计算梯度方向
theta = np.arctan2(g_y, g_x)
print(">>> gradient theta:", theta)

# %%
g_ = cv.merge(g_x, g_y)
uu.imshow(g_)

# %% NMS（非极大值抑制）
"""
非最大值抑制是一种边缘细化方法。
通常得出来的梯度边缘不止一个像素宽，而是多个像素宽。
就像我们所说Sobel算子得出来的边缘粗大而明亮，因此这样的梯度图还是很“模糊”。
非最大值抑制能帮助保留局部最大梯度而抑制所有其他梯度值。这意味着只保留了梯度变化中最锐利的位置。

原理很简单：遍历梯度矩阵上的所有点，并保留边缘方向上具有极大值的像素。就像下面这幅图一样。

![](https://ai-chen.github.io/assets/images/canny/9.JPG)

算法如下：

1. 比较当前点的梯度强度和正负梯度方向点的梯度强度。
1. 如果当前点的梯度强度和同方向的其他点的梯度强度相比较是最大，保留其值。否则抑制，即设为0。
    比如当前点的方向指向正上方90°方向，那它需要和垂直方向，它的正上方和正下方的像素比较。
"""

# %% 双阈值的边界选取
"""
在施加非极大值抑制之后，剩余的像素可以更准确地表示图像中的实际边缘。
然而，仍然存在由于噪声和颜色变化引起的一些边缘像素。为了解决这些杂散响应，必须用弱梯度值过滤边缘像素，
并保留具有高梯度值的边缘像素，可以通过选择高低阈值（minVal,maxVal）来实现。

1. 如果边缘像素的梯度值高于高阈值，则将其标记为强边缘像素；
1. 如果边缘像素的梯度值小于高阈值并且大于低阈值，则将其标记为弱边缘像素；
1. 如果边缘像素的梯度值小于低阈值，则会被抑制。

![](http://cos.codec.wang/cv2_understand_canny_max_min_val.jpg)
"""

# %% 抑制孤立弱边缘完成边缘检测
"""
到目前为止，被划分为强边缘的像素点已经被确定为边缘，因为它们是从图像中的真实边缘中提取出来的。
然而，对于弱边缘像素，将会有一些争论，因为这些像素可以从真实边缘提取也可以是因噪声或颜色变化引起的。
为了获得准确的结果，应该抑制由后者引起的弱边缘。
通常，由真实边缘引起的弱边缘像素将连接到强边缘像素，而噪声响应未连接。
为了跟踪边缘连接，通过查看弱边缘像素及其[公式]个邻域像素，只要其中一个为强边缘像素，
则该弱边缘点就可以保留为真实的边缘。
"""

#####################################################################
# %% 直方图
# http://codec.wang/#/opencv/basic/15-histograms

path_shapes = "tutorial/img/handwriting.jpg"
img = cv.imread(path_shapes)

# %% 比较cv2与numpy的算法效率，推荐: np.bincount()
# hist = cv2.calcHist([img], [0], None, [256], [0, 256])  # 性能：0.025288 s
# hist2, bins = np.histogram(img.ravel(), 256, [0, 256])  # 性能：0.020628 s
hist3 = np.bincount(img.ravel(), minlength=256)  # 性能：0.003163 s
plt.hist(img.ravel(), 256, [0, 256])

# %% 直方图均衡化
equ = cv2.equalizeHist(img)
uu.imshow(equ)

# %% 自适应均衡化
# 不难看出来，直方图均衡化是应用于整幅图片的。因为全局调整亮度和对比度的原因，可能导致局部太亮，丢失细节。
# 自适应均衡化就是用来解决这一问题的：它在每一个小区域内（默认8×8）进行直方图均衡化。
# 当然，如果有噪点的话，噪点会被放大，需要对小区域内的对比度进行了限制。
# 所以这个算法全称叫：对比度受限的自适应直方图均衡化CLAHE(Contrast Limited Adaptive Histogram Equalization)。
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cl1 = clahe.apply(img)
uu.imshow(cl1)

#####################################################################
# %% 查找直线与圆
path_shapes = "tutorial/img/shapes.jpg"
im_shapes = cv.imread(path_shapes)
uu.imshow(im_shapes)

# %% find_lines or circles
im_edges = cv.canny(im_shapes, [50,100])
uu.imshow(im_edges)
# %%
lines = cv.find_lines(im_edges, 0.8, 90, min_length=50)
im_dr = np.zeros(im_shapes.shape)
for line in lines:
    cv.draw_line(im_dr, line.a, line.b, 255, 1)
    print(cv.distance(*line))
uu.imshow(im_dr)

# %% 查找圆
# 因为霍夫圆检测对噪声比较敏感,所以首先要对图像做中值滤波！
# 基于效率考虑, OpenCV中实现的霍夫变换圆检测是基于图像梯度的实现,分为两步:
# 1. 检测边缘,发现可能的圆心
# 2. 基于第一步的基础上从候选圆心开始计算最佳半径大小
circles = cv.find_circles(im_edges, im_edges.shape[1]*0.5, 30)
im_dr = np.zeros(im_shapes.shape)
for cc in circles:
    cv.draw_circle(im_dr, cc.cx, cc.cy, cc.r, 255, 1)
uu.imshow(im_dr)

#####################################################################
# %% 凸包及更多轮廓特征
path_convex = "tutorial/img/convex.jpg"
im_convex = cv.imread(path_convex)
uu.imshow(im_convex)

# %%
blobs = cv.find_blobs(im_convex, 111)
assert len(blobs) == 1
blob = blobs[0]
hull = cv.approx_convex(blob._cnt)
# cv.draw_polygon(im_convex, hull, 255)
# uu.imshow(im_convex)

# %%
defects = cv.convex_defects(blob._cnt)
im_dr = im_convex.copy()
# cv.draw_points(im_dr, defects, (255,0,0), 5)
for p, dist in defects:
    print(">>> 距离轮廓：", dist)
    cv.draw_circle(im_dr, *p, 5, (255,0,0), 5)
uu.imshow(im_dr)
