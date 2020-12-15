#!/usr/bin/env python3
# @Date    : 2020-12-04
# @Author  : Bright Li (brt2@qq.com)
# @Link    : https://gitee.com/brt2
# @Version : 0.2.2

import numpy as np
import pycv.opencv as cv

class UtilFaker:
    def imshow(self, *args, **kwargs):
        """ 不显示图像 """
uu = UtilFaker()

def improc_origin(im, *args):
    return cv.resize(im, (800,600))

def improc_right(im, *args):
    return cv.threshold(im, args[0])

def improc_roi(im, *args):
    im_h, im_w = im.shape[:2]
    w, h = 352, 100
    left, right = int(args[1]/255 * im_w), int(args[2]/255 * im_h)
    ROI = [left, right, w, h]
    cv.draw_rect(im, *ROI, thickness=2)
    im_roi = cv.crop(im, ROI)

    # im_gau = cv.gaussian(im_roi, 3)
    # im_bin = cv.threshold(im_roi, args[0])
    # im_med = cv.median(im_bin, 5)
    return im_roi
