# %%
from utils import expy
expy.enpy("opencv")
import numpy as np

import util as uu
import ocv as cv

uu.chdir(__file__)

# %%
uu.reload(uu)

# %%
path_img = uu.rpath("img/usb_cam_4.jpg")
im = cv.imread(path_img)
uu.imshow(im)

# %% 图像裁剪
roi = cv.crop2(im, (500,0), (1900, 1350))
uu.imshow(roi)

# %% 灰度图滤波
im_blur = cv.bilateral(roi, 15, 120, 10)
uu.imshow(im_blur)

# %% ! 对比度增强
from matplotlib import pyplot as plt
plt.hist(roi.ravel(), 256, [0, 256])

# %%
im_bin = cv.threshold(roi, 150, 1)
uu.imshow(im_bin)

# %% 中值滤波：去除二值图里的噪点
im_mooth = cv.median(im_bin, 5)
uu.imshow(im_mooth, 1)

# %% ! 找轮廓
cnts = cv.find_cnts(im_mooth)
list_cnts = []
for cnt in cnts:
    if cv.cnt_area(cnt) > 100:
        list_cnts.append(cnt)
print(">>> Len(cnts) =", len(list_cnts))

im_dr = np.zeros(im_mooth.shape)
cv.draw_contours(im_dr, list_cnts)
uu.imshow(im_dr)

# %% 通过顶帽，获取边缘凸起的增强
im_close = cv.tophat(im_mooth, (15,15))
uu.imshow(im_close)

# %%
list_blobs = cv.find_blobs(im_close)
list_burr = []
for b in list_blobs:
    if b.area() < 10:
        continue
    if b.elongation() > 0.1:
        list_burr.append(b.center())
print(list_burr)

# %%
for p in list_burr:
    cv.draw_cross(im_dr, *p, (255, 0, 0), 5)
    cv.draw_circle(im_dr, *p, 5, (255, 0, 0), 5, 5)
    cv.draw_string(im_dr, *p, "NG")

cv.draw_string(im_dr, 100, 100, f"NG: number of burr: {len(list_burr)}", thickness=2)
uu.imshow(im_dr)
