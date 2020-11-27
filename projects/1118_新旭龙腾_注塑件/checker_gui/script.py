#!/usr/bin/env python3
# @Date    : 2020-11-18
# @Author  : Bright Li (brt2@qq.com)
# @Link    : https://gitee.com/brt2
# @Version : 0.2.1

import numpy as np
import opencv as cv

class UtilFaker:
    def imshow(self, *args, **kwargs):
        """ 不显示图像 """

uu = UtilFaker()

def improc2(im):
    # return cv.canny(im, 30)
    return cv.threshold(im, 99)

def improc(im, outputs=None):
    # %% 图像裁剪
    print(">>> 图像尺寸：", im.shape[:2])
    w, h = max(im.shape[:2]), min(im.shape[:2])
    w_start = (w - h) // 2
    roi = cv.crop(im, (w_start, 0, w, h))
    uu.imshow(roi)

    # # %% 灰度图滤波
    # im_blur = cv.bilateral(roi, 15, 120, 10)
    # uu.imshow(im_blur)

    # %%
    im_bin = cv.threshold(roi, 130, 1)
    uu.imshow(im_bin)

    im_dil = cv.dilate(im_bin, (3,3))

    # %% 中值滤波：去除二值图里的噪点
    im_med = cv.median(im_dil, 5)
    uu.imshow(im_med, 1)
    # return im_med

    # %% ! 找轮廓
    edges = cv.canny(im_dil, 30)
    """
    cnts = cv.find_cnts(cv.canny(im_dil, 30))
    list_cnts = []
    for cnt in edges:
        print(cv.cnt_area(cnt))
        if cv.cnt_area(cnt) < 100:
            continue
        list_cnts.append(cnt)
    print(">>> Len(cnts) =", len(list_cnts))

    im_dr = np.zeros(im_dil.shape)
    cv.draw_contours(im_dr, cnts)
    uu.imshow(im_dr)
    """

    # %% 通过顶帽，获取边缘凸起的增强
    im_close = cv.tophat(im_med, (9,9))
    uu.imshow(im_close)
    # return im_close

    # %%
    list_blobs = cv.find_blobs(im_close)
    list_burr = []
    for b in list_blobs:
        if b.area() < 15:
            continue
        if b.elongation() > 0.1:
            list_burr.append(b.center())

    # %%
    im_dr = cv.gray2rgb(edges)
    for p in list_burr:
        cv.draw_cross(im_dr, *p, (255, 0, 0), 5)
        cv.draw_circle(im_dr, *p, 5, (255, 0, 0), 5, 5)
        cv.draw_string(im_dr, *p, "NG")

    cv.draw_string(im_dr, 100, 100, f"NG: number of burr: {len(list_burr)}", thickness=2)
    uu.imshow(im_dr)

    return im_dr

