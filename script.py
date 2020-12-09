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

def improc(im, *args):
    return cv.threshold(im, args[0])

def improc_ocr(im, *args):
    # ROI = [250, 200, 300, 100]
    w, h = 300, 200
    left, right_off = args[1], args[2]
    if left and right_off:
        ROI = [left, h, w+right_off, 60]
        cv.draw_rect(im, *ROI, thickness=2)
        im_roi = cv.crop(im, ROI)
    else:
        im_roi = im
    # im_gau = cv.gaussian(im_roi, 3)
    im_bin = cv.threshold(im_roi, args[0])
    im_med = cv.median(im_bin, 3)
    return im_med


