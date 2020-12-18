# %%
import ipyenv as uu
uu.chdir(__file__)
uu.enpy("opencv")

import pycv.opencv as cv
import numpy as np

# %%
uu.reload(cv)

# %%
path_tpl = uu.rpath("image/cap_1.jpg")
im_tpl = cv.imread(path_tpl)
uu.imshow(im_tpl)

# # %% 增强对比度
# tpl_con = cv.contrast(im_tpl, 14, 90)
# uu.imshow(tpl_con)

# %% 去除边界
margin = 60
h, w = im_tpl.shape[:2]
tpl_no_margin = cv.crop2(im_tpl, (margin, margin), (w-margin,h))
uu.imshow(tpl_no_margin)

# %%
""" 尝试使用角点检测
x = cv.find_corners(im, 0.8, 1)
print(x)
center = x[0]
im_dr = im.copy()
cv.draw_circle(im_dr, *center, 20)
uu.imshow(im_dr)
"""

def pos_corner(im, thresh, margin_ratio=0.1):
    im_bin = cv.binary(im, thresh)
    im_proc = cv.median(im_bin, 5)
    # im_open = cv.opening(im_bin, (9,9))
    # im_gau = cv.gaussian(im_open, 3)

    list_blobs = cv.find_cnts(im_proc)
    # assert len(list_blobs) == 1
    if len(list_blobs) == 1:
        polygon = list_blobs[0]
    else:
        max_area = 0
        for b in list_blobs:
            b_area = cv.cnt_area(b)
            if b_area > max_area:
                max_area = b_area
                polygon = b

    list_pnts = cv.approx_polygon(polygon, epsilon=11)
    w, h = im.shape[:2]
    w_margin = w * margin_ratio
    h_margin = h * margin_ratio
    w_range = (w_margin, w-w_margin)
    h_range = (h_margin, h-h_margin)

    corners = []
    for p in list_pnts:
        if w_range[0] <= p[0] <= w_range[1] and h_range[0] <= p[1] <= h_range[1]:
            corners.append(p)

    nCor = len(corners)
    if nCor == 1:
        return corners[0]
    elif nCor > 1:
        x = round(sum([p[0] for p in corners]) / nCor)
        y = round(sum([p[1] for p in corners]) / nCor)
        return [int(x),int(y)]
    else:
        print(">>>", list_pnts)
        uu.imshow(im_proc)

cor_len = 160
h, w = tpl_no_margin.shape[:2]
list_corners = []
for x,y in [(0,0), (w-cor_len, 0), (w-cor_len, h-cor_len), (0, h-cor_len)]:
    im_corner = cv.crop(tpl_no_margin, (x,y,cor_len,cor_len))
    relpos = pos_corner(im_corner, thresh=111)
    abspos = (x+relpos[0], y+relpos[1])
    list_corners.append(abspos)

print(list_corners)

# %% 透视变换
tpl_psp = cv.perspect2rect(tpl_no_margin, list_corners)
uu.imshow(tpl_psp, 1)

# %% 滤波，二值化，多边形拟合
tpl_bin = cv.binary(tpl_psp, 111, 1)
tpl_med = cv.median(tpl_bin, 3)
uu.imshow(tpl_med)

#####################################################################

# %% 载入
path_check = uu.rpath("image/cap_2.jpg")
im_check = cv.imread(path_check)
uu.imshow(im_check)

# %% 去除边界
margin = 60
h, w = im_check.shape[:2]
check_no_margin = cv.crop2(im_check, (margin, margin), (w-margin,h-margin))
uu.imshow(check_no_margin)

# %%
cor_len = 160
h, w = check_no_margin.shape[:2]
list_corners2 = []
for x,y in [(0,0), (w-cor_len, 0), (w-cor_len, h-cor_len), (0, h-cor_len)]:
    im_corner = cv.crop(check_no_margin, (x,y,cor_len,cor_len))
    relpos = pos_corner(im_corner, thresh=111)
    if relpos is None:
        continue
    abspos = (x+relpos[0], y+relpos[1])
    list_corners2.append(abspos)

print(list_corners2)

# %% 透视为模板的尺寸
check_psp = cv.perspect2rect(check_no_margin, list_corners2, (2472, 1884))
check_bin = cv.binary(check_psp, 111, 1)
check_med = cv.median(check_bin, 3)
uu.imshow(check_med)

# %% 比对： diff
im_diff = tpl_med - check_med
uu.imshow(im_diff)
