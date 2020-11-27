#!/usr/bin/env python3
# @Date    : 2020-11-27
# @Author  : Bright Li (brt2@qq.com)
# @Link    : https://gitee.com/brt2
# @Version : 0.2.1

import numpy as np
import opencv as cv

class UtilFaker:
    def imshow(self, *args, **kwargs):
        """ 不显示图像 """
uu = UtilFaker()

def improc(im):
    # return cv.canny(im, 30)
    return cv.threshold(im, 99)
