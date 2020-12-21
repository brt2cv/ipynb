#!/usr/bin/env python3
# @Date    : 2020-12-17
# @Author  : Bright Li (brt2@qq.com)
# @Link    : https://gitee.com/brt2
# @Version : 0.2.1

# %%
import ipyenv as uu
uu.chdir(__file__)
uu.enpy("opencv")

import pycv.opencv as cv
import numpy as np

# %%
uu.reload(cv)

# %%
im = cv.imread(uu.rpath("image/01.jpg"))
uu.imshow(im)

# %%
im_next = cv.binary(im, 188, 1)

# %%
im_next = cv.closing(im_next, (3,3))
uu.imshow(im_next)

# %%
list_cnts = cv.find_blobs(im_next)
max_area = 0
max_blob = None
for b in list_cnts:
    if b.area() > max_area:
        max_area = b.area()
        max_blob = b

# %% 透视拉正
# x = cv.perspect2rect(im_next, cv.bounding_rect(max_blob.cnt), tuple(int(i) for i in max_blob.size))
# uu.imshow(x)

# %% 2D放射变换
im_core = cv.crop(im, max_blob.bounding())
im_deg = cv.rotate(im_core, max_blob.rotation_deg)
uu.imshow(im_deg)

# %% 文字区域
im_roi = cv.crop2(im_deg, (25, 10), (300, 45))
uu.imshow(im_roi)
