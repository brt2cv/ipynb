#!/usr/bin/env python3
# @Date    : 2020-12-17
# @Author  : Bright Li (brt2@qq.com)
# @Link    : https://gitee.com/brt2
# @Version : 0.2.10

import os
import platform
import math
import numpy as np

try:
    from PIL import Image as PilImageModule, ImageDraw
    ENABLE_MODULE_PILLOW = True
except ImportError:
    ENABLE_MODULE_PILLOW = True

try:
    import cv2
    ENABLE_MODULE_OPENCV = True
except ImportError:
    ENABLE_MODULE_OPENCV = True

try:
    import imageio
    ENABLE_MODULE_IMAGEIO = True
except ImportError:
    ENABLE_MODULE_IMAGEIO = False

DEBUG_MODE = True

#####################################################################
# Math
#####################################################################

def distance(pnt1, pnt2):
    # return np.square(np.sum(np.square(vec1 - vec2)))
    return np.linalg.norm(pnt1 - pnt2)

def nonzero_zone(im):
    # return im[x > 0] = 255
    # return np.nonzero(mask)
    return cv2.findNonZero(im)

#####################################################################
# IO module
#####################################################################

# from cv2 import imread
# from cv2 import imwrite

def isAscii(s):
    return all(ord(c) < 128 for c in s)

def imread_byImageio(uri, **kwargs):
    if kwargs.get("as_gray"):
        pilmode = kwargs.get("pilmode", "L")
        assert pilmode != "L", f"pilmode设置【{pilmode}】与 as_gray->【L】不匹配，请验证参数"
        kwargs["pilmode"] = "L"
        kwargs["as_gray"] = False
    return imageio.imread(uri, **kwargs)

def imread(uri, as_gray=True):
    if DEBUG_MODE:
        assert os.path.exists(uri)

    if platform.system() == "Windows":
        if ENABLE_MODULE_IMAGEIO:
            return imread_byImageio(uri)
        # assert isAscii(uri), "Windows::OpenCV不支持中文路径"

    mode = cv2.IMREAD_GRAYSCALE
    if not as_gray:
        mode = cv2.IMREAD_COLOR  # cv2.IMREAD_UNCHANGED
    return cv2.imread(uri, mode)

from cv2 import imwrite

def imsave(im_arr, path_save):
    """ 注意：参数顺序不同于cv2.imwrite(path_save, im_arr) """
    return cv2.imwrite(path_save, im_arr)

def shape2size(shape):
    """ im_arr.shape: {h, w, c}
        PIL.Image.size: {w, h}
    """
    size = (shape[1], shape[0])
    return size

def shape2mode(shape):
    if len(shape) < 3:
        return "L"
    elif shape[2] == 3:
        return "RGB"  # 无法区分BGR (OpenCV)
    elif shape[2] == 4:
        return "RGBA"
    else:
        raise Exception("未知的图像类型")

def guess_mode(im_arr):
    """ 一种预测图像mode的简单方式 """
    if im_arr.ndim < 3:
        return "L"
    else:
        return shape2mode(im_arr.shape)

class _ImageConverter:
    """ 内部使用numpy::array存储数据
        mode: "1", "L", "P", "RGB", "BGR", "RGBA", "YUV", "LAB"
    """
    def mode(self):
        return self._img.mode

    def width(self):
        return self._img.width()

    def height(self):
        return self._img.height()

    #####################################################################
    def convert(self, mode):
        if mode == self._img.mode:
            return
        self._img = self._img.convert(mode)

    # IO
    def open(self, path_file):
        self._img = PilImageModule.open(path_file, "r")

    def save(self, path_file):
        # with open(path_file, "wb") as fp:
        self._img.save(path_file)

    # def load(self, arg=None, **kwargs):
    #     if arg is None:
    #         self._img = None  # PIL::Image
    #     elif len(kwargs) == 0 and isPath(arg):
    #         self.open(arg)
    #     elif len(kwargs) == 1 and isinstance(arg, np.ndarray):
    #         self.from_numpy(arg, mode=kwargs["mode"])
    #     elif len(kwargs) == 2 and isinstance(arg, (bytes, bytearray)):
    #         self.from_bytes(arg, kwargs["mode"], kwargs["size"])
    #     else:
    #         raise Exception(f"Unkown arguments to load: 【arg:{arg}, kwargs:{kwargs}】")

    def from_bytes(self, data, mode, **kwargs):
        """ mode: "1", "L", "P", "RGB", "RGBA", "CMYK"...
        """
        if "size" in kwargs:
            size = kwargs["size"]
        elif "shape" in kwargs:
            size = shape2size(kwargs["shape"])
        else:
            raise Exception("必须传入size或shape参数")

        self._img = PilImageModule.frombytes(mode, size, data)

    def to_bytes(self):
        return self._img.tobytes()

    def from_numpy(self, im_arr, mode):
        """ 这里设定mode为显式参数，因为无法通过channel完全确定mode：
            * 2dim: "1", "L", "P", "I", "F"
            * 3dim: "RGB", "BGR"
            * 4dim: "RGBA", "CMYK", "YCbCr"
        """
        if mode == "BGR":
            self.from_opencv(im_arr)
            return
        self._img = PilImageModule.fromarray(im_arr, mode)

    def to_numpy(self):
        im = np.asarray(self._img)
        return im

    def from_qtimg(self, qt_img, type="QPixmap"):
        if type == "QPixmap":
            self._img = PilImageModule.fromqpixmap(qt_img)
        elif type == "QImage":
            self._img = PilImageModule.fromqimage(qt_img)
        else:
            raise Exception(f"Unkown type 【{type}】")

    def to_qtimg(self, type="QPixmap"):
        if type == "QPixmap":
            return self._img.toqpixmap()
        elif type == "QImage":
            return self._img.toqimage()
        else:
            raise Exception(f"Unkown type 【{type}】")

    # def from_opencv(self, im_arr, cv_mode="BGR"):
    #     if len(im_arr.shape) < 3:
    #         assert len(cv_mode) == 1, f"当前nadrray的维度与参数cv_mode【{cv_mode}】不匹配"
    #     elif cv_mode == "RGB":
    #         pass
    #     elif cv_mode == "BGR":
    #         im_arr = bgr2rgb(im_arr)
    #         cv_mode = "RGB"
    #     else:
    #         raise Exception("敬请期待")
    #     self.from_numpy(im_arr, cv_mode)

    # def to_opencv(self):
    #     # opencv 不处理 ndim>3 的图像
    #     im_arr = self.to_numpy()
    #     if len(im_arr.shape) >= 3:  # 保存为3维数据
    #         if self.mode != "RGB":
    #             # im_arr = np.delete(im_arr, -1, axis=1)
    #             self._img.convert("RGB")
    #         im_arr = rgb2bgr(im_arr)
    #     return im_arr

    def from_pillow(self, pil_img):
        self._img = pil_img

    def to_pillow(self):
        return self._img

_converter = _ImageConverter()  # 全局转换器对象，用于图片格式转换

def ndarray2pixmap(im_arr, mode=None):
    if mode is None:
        mode = guess_mode(im_arr)

    _converter.from_numpy(im_arr, mode)
    pixmap = _converter.to_qtimg()
    return pixmap

def pixmap2ndarray(pixmap):
    _converter.from_qtimg(pixmap)
    im_arr = _converter.to_numpy()
    return im_arr

def ndarray2bytes(im_arr, mode=None):
    if mode is None:
        mode = guess_mode(im_arr)

    _converter.from_numpy(im_arr, mode)
    bytes_ = _converter.to_bytes()
    return bytes_

def bytes2ndarray(data, mode, **kwargs):
    """ kwargs:
            - size: (w, h)
            - shape: (h, w, ...)
    """
    if "size" in kwargs:
        size = kwargs["size"]
    elif "shape" in kwargs:
        size = shape2size(kwargs["shape"])
    else:
        raise Exception("必须传入size或shape参数")

    _converter.from_bytes(data, mode, size=size)
    im_arr = _converter.to_numpy()
    return im_arr

def pillow2ndarray(pil_img):
    im_arr = np.asarray(pil_img)
    return im_arr

    # _converter.from_pillow(pil_img)
    # im_arr = _converter.to_numpy()
    # return im_arr

def ndarray2pillow(im_arr, mode=None):
    if mode is None:
        mode = guess_mode(im_arr)

    _converter.from_numpy(im_arr, mode)
    return _converter._img

def convert_mode(im_arr, mode_to, mode_from=None):
    if mode_from is None:
        mode_from = guess_mode(im_arr)

    if mode_to == mode_from:
        return im_arr
    else:
        _converter.from_numpy(im_arr, mode_from)
        _converter.convert(mode_to)
        im_arr2 = _converter.to_numpy()
        return im_arr2

#####################################################################
# Transform
#####################################################################

def float2uint8(im):
    # return (im * 255).astype(np.uint8)
    return cv2.convertScaleAbs(im)

def split(im):
    return cv2.split(im)

def merge(bands):
    return cv2.merge(bands)

# 色彩空间
def rgb2gray(im):
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

def gray2rgb(im):
    return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

def rgb2bgr(im):
    return cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

def bgr2rgb(im):
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

def rgb2hsv(im):
    return cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

def hsv2rgb(im):
    return cv2.cvtColor(im, cv2.COLOR_HSV2BGR)

def rgb2lab(im):
    return cv2.cvtColor(im, cv2.COLOR_BGR2LAB)

def lab2rgb(im):
    return cv2.cvtColor(im, cv2.COLOR_Lab2BGR)

def rgb2yuv(im):
    return cv2.cvtColor(im, cv2.COLOR_BGR2YUV)

def yuv2rgb(im):
    return cv2.cvtColor(im, cv2.COLOR_YUV2BGR)

#####################################################################
# Image Transfrom
#####################################################################

resize = lambda im, size: cv2.resize(im, dsize=size)

def rescale(im, scale):
    """ scale could be float or tuple(row, colulmn) like [0.2, 0.3] """
    shape = list(im.shape)
    try:
        # fx, fy = scale
        nLen = len(scale)
        assert nLen <= len(shape), f"当前图像的shape为【{shape}】，不支持{nLen}个缩放参数【{scale}】"
        list_scale = list(reversed(scale))  # 调换为(column, row)
    except TypeError:  # scale 为 float
        list_scale = [scale, scale]
    list_scale += [1] * (len(shape) -2)
    return cv2.resize(im, None, fx=list_scale[0], fy=list_scale[1])

def rotate(im, angle):
    """ angle: 逆时针角度 """
    # cv2.flip(im, flipCode)  # 翻转
    cols, rows = im.shape[:2]
    # cols-1 and rows-1 are the coordinate limits.
    M = cv2.getRotationMatrix2D(((rows-1)/2, (cols-1)/2), angle, 1)
    return cv2.warpAffine(im, M, (rows, cols))

def perspect_transform(im, src_pnts, dst_pnts, new_size):
    M = cv2.getPerspectiveTransform(src_pnts, dst_pnts)
    return cv2.warpPerspective(im, M, new_size)

def perspect2rect(im, src_pnts, new_size=None):
    if new_size is None:
        h, w = im.shape[:2]
        new_size = (w,h)
    w, h = new_size
    return perspect_transform(im, np.float32(src_pnts),
           np.float32([(0,0), (w,0), (w,h), (0,h)]), new_size)

def crop(im, roi):
    """ roi: (x, y, w, h) """
    x, y, w, h = roi
    return im[y:y+h, x:x+w].copy()

def crop2(im, top_left, bottom_right):
    x, y = top_left
    x2, y2 = bottom_right
    return crop(im, (x, y, x2-x, y2-y))

# https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#gacea54f142e81b6758cb6f375ce782c8d
def sobel(im, dx=1, dy=1, ksize=3):
    """
    dx,dy: 求导阶数，0表示不求导。
    ksize: size of the extended Sobel kernel; it must be 1, 3, 5, or 7.
    """
    return cv2.Sobel(im, -1, dx, dy, ksize=ksize)

def scharr(im, dx=1, dy=1):
    """
    dx,dy: 求导阶数，0表示不求导。
    """
    # 虽然Sobel算子可以有效的提取图像边缘，但是对图像中较弱的边缘提取效果较差。
    # 因此为了能够有效的提取出较弱的边缘，需要将像素值间的差距增大，因此引入Scharr算子。
    # Scharr算子是对Sobel算子差异性的增强，因此两者之间的在检测图像边缘的原理和使用方式上相同。

    # Kernal = [-3 0 3 -10 0 10 -3 0 3]
    # Scharr算子的边缘检测滤波的尺寸为3×3，因此也有称其为Scharr滤波器。
    return cv2.Scharr(im, -1, dx, dy)

def laplacian(im, ksize=1):
    """ 二阶导数计算梯度 """
    return cv2.Laplacian(im, -1, ksize=ksize)

#####################################################################
# Pixel
#####################################################################

# def bitwise_not(im_arr)
from cv2 import bitwise_not
invert = bitwise_not

# def bitwise_xor(im_arr, mask)
from cv2 import bitwise_xor  # 取取不重叠的区域:

# def bitwise_or(im_arr, mask)
from cv2 import bitwise_or  # 取并集

def bitwise_diff(im_arr, mask):
    mask_not = bitwise_not(mask)
    return cv2.bitwise_and(im_arr, im_arr, mask_not)

# from np import bitwise_and
def bitwise_and(im_arr, mask):
    return cv2.bitwise_and(im_arr, im_arr, mask)

# def add(src1, src2, dst=None, mask=None, dtype=None)
# def multiply(src1, src2, dst=None, scale=None, dtype=None)
# def subtract(src1, src2, dst=None, mask=None, dtype=None)
# def divide(src1, src2, dst=None, scale=None, dtype=None)

from cv2 import add  # 饱和操作（不同于np.add）
# from cv2 import addWeighted  # 透明感
# def addWeighted(img1, 0.7, img2, 0.3, 0)

def add2(img1, img2, weights=0.5):
    """ weights: float(0.7) for img1, or list([7,5] for weights) """
    if isinstance(weights, float):
        w1, w2 = weights, 1-weights
    else:
        w1, w2 = weights
    return cv2.addWeighted(img1, w1, img2, w2, 0)

def _split_thresholds(thresholds):
    return [thresholds, 255] if isinstance(thresholds, int) else thresholds

def binary(im, thresholds, invert=False):
    """
    thresholds: int or list(thresh, maxval), or 'otsu'/'triangle'
    type_:
        cv2.THRESH_BINARY
        cv2.THRESH_BINARY_INV
        cv2.THRESH_TRUNC
        cv2.THRESH_TOZERO
        cv2.THRESH_TOZERO_INV
    """
    type_ = cv2.THRESH_BINARY  # 0
    if isinstance(thresholds, str):
        thresh, maxval = 0, 255
        if thresholds == "otsu":
            type_ += cv2.THRESH_OTSU  # 8
        elif thresholds == "triangle":
            type_ += cv2.THRESH_TRIANGLE
        else:
            raise KeyError(f"未知的关键词【{thresholds}】")
    else:
        thresh, maxval = _split_thresholds(thresholds)

    if invert:
        type_ += 1
    _, im2 = cv2.threshold(im, thresh, maxval, type_)
    return im2

threshold = binary

def threshold_otsu(im, invert=False):
    return binary(im, "otsu", invert)

def threshold_triangle(im, invert=False):
    return binary(im, "triangle", invert)

def graystairs(im, low_val, high_val):
    im2 = np.subtract(im, low_val, casting="unsafe")
    k = 255 / max(high_val - low_val, 1e-10)
    im2 = np.multiply(im2, k, casting='unsafe').astype("uint8")
    im2[im < low_val] = 0
    im2[im > high_val] = 255
    return im2

def contrast(im, bright, contrast):
    mid = 128 - bright
    length = 255 / np.tan(contrast / 180 * np.pi)

    im2 = im.copy()
    if mid - length / 2 > 0:
        np.subtract(im2, mid-length / 2, out=im2, casting='unsafe')
        np.multiply(im2, 255 / length, out=im2, casting='unsafe')
    else:
        np.multiply(im2, 255 / length, out=im2, casting='unsafe')
        np.subtract(im2, (mid - length / 2) / length * 255, out=im2, casting='unsafe')
    im2[im < mid-length/2] = 0
    im2[im > mid+length/2] = 255
    return im2

#####################################################################
# Filters
# https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html
#####################################################################

def gaussian(im, k, sigma=0):
    """
    ksize: scalar or tuple of scalars
    参数3: σx值越大，模糊效果越明显。
    """
    # 高斯滤波相比均值滤波效率要慢，但可以有效消除高斯噪声，能保留更多的图像细节。
    if isinstance(k, int):
        k = (k, k)
    elif DEBUG_MODE:
        assert isinstance(k, tuple)
    # ksize.width和ksize.height必须为正奇数，也可以为零，然后根据sigma计算得出
    # sigma_x可以为零，则分别从ksize.width和ksize.height计算得出
    return cv2.GaussianBlur(im, k, sigma)

def median(im, ksize: int):
    """ k: 必须为奇数 """
    # 中值滤波就是用区域内的中值来代替本像素值，所以那种孤立的斑点，如0或255很容易消除掉，
    # 适用于去除椒盐噪声和斑点噪声。
    # 中值是一种非线性操作，效率相比前面几种线性滤波要慢。
    assert ksize % 2, "k值必须为奇数"
    return cv2.medianBlur(im, ksize)

def mean(im, kernal: tuple):
    """ 均值滤波 """
    return cv2.blur(im, kernal)

def bilateral(im, d, sigma_color, sigma_space):
    """ 双边滤波
    d: -1，则从sigmaSpace中计算得到。常见的d取值为15或者20，如果过大会导致运算时间较长。
    sigmaColor: 表示高斯核中颜色值标准方差，如: 120
    sigmaSpace: 表示高斯核中空间的标准方差，如: 10
    """
    # 模糊操作基本都会损失掉图像细节信息，尤其前面介绍的线性滤波器，图像的边缘信息很难保留下来。
    # 然而，边缘（edge）信息是图像中很重要的一个特征，所以这才有了双边滤波。
    return cv2.bilateralFilter(im, d, sigma_color, sigma_space)

def sharpening(im, sigma=5):
    """ Laplacian Sharpening（拉普拉斯锐化） """
    # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    kernel = np.array([[0, -1, 0], [-1, sigma, -1], [0, -1, 0]])
    im_laplacian = cv2.filter2D(im, -1, kernel)
    # return cv2.addWeighted(im, 1, im_laplacian, 0.5, 0)
    return im_laplacian

def shapening_USM(im, sigma=5):
    blur = cv2.GaussianBlur(im, (0, 0), sigma)
    usm = cv2.addWeighted(im, 1.5, blur, -0.5, 0)
    return usm

#####################################################################
# Morpholopy 形态学操作
# https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html
#####################################################################

KERNEL_SHAPE_OPENCV = {
    "rect": 0,
    "cross": 1,
    "ellipse": 2
}

def kernal(size, shape="rect"):
    """ return a np.ndarray as kernal """
    if isinstance(size, np.ndarray):
        return size
    # elif isinstance(size, int):
    #     size = (size, size)
    # elif DEBUG_MODE and not isinstance(size, tuple):
    #     size = tuple(size)
    nShape = KERNEL_SHAPE_OPENCV[shape]
    return cv2.getStructuringElement(nShape, size)

# def erosion(im, k):
erode = lambda im, k: cv2.erode(im, kernal(k))

# def dilation(im, k):
dilate = lambda im, k: cv2.dilate(im, kernal(k))

# 先腐蚀再膨胀，消除小物体或小斑块
opening = lambda im, k: cv2.morphologyEx(im, cv2.MORPH_OPEN, kernal(k))

# 先膨胀再腐蚀，填充孔洞
closing = lambda im, k: cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernal(k))

# 梯度：图像的膨胀和腐蚀之间的差异，结果看起来像目标的轮廓
gradient = lambda im, k: cv2.morphologyEx(im, cv2.MORPH_GRADIENT, kernal(k))

# 顶帽：原图像减去它的开运算值，突出原图像中比周围亮的区域
tophat = lambda im, k: cv2.morphologyEx(im, cv2.MORPH_TOPHAT, kernal(k))

# 黑帽：原图像减去它的闭运算值，突出原图像中比周围暗的区域
blackhat = lambda im, k: cv2.morphologyEx(im, cv2.MORPH_BLACKHAT, kernal(k))

#####################################################################
# Feature 特征处理
#####################################################################

def canny(im, thresholds):
    """ thresholds: int or list(thresh, maxval) """
    thresholds = _split_thresholds(thresholds)
    return cv2.Canny(im, *thresholds)

edges = canny

def find_corners(im, quality, max_num=0, min_dis=0):
    """
    max_num: 最大返回关键点数目
    min_dis: 两个关键点之间的最短距离
    quality: float from [0,1], 越大则对检测角点的质量要求越严格
    """
    corners = cv2.goodFeaturesToTrack(im, max_num, qualityLevel=quality, minDistance=min_dis)
    if corners is not None:
        if DEBUG_MODE:
            print(f">>> 共计检测到【{len(corners)}】个Corner对象")
        return corners[:, 0]

from collections import namedtuple
# Point = namedtuple("Point", ["x", "y"])
Line = namedtuple("Line", ["a", "b"])
Circle = namedtuple("Circle", ["cx","cy","r"])

def find_lines(im, rho, threshold, theta=np.pi/180, min_length=0, max_gap=0):
    """ 统计概率霍夫直线变换 """
    lines = cv2.HoughLinesP(im, rho, theta, threshold,
                        minLineLength=min_length, maxLineGap=max_gap)
    # im:  要检测的二值图（一般是阈值分割或边缘检测后的图）
    # rho: 距离r的精度，值越大，考虑越多的线（控制所监测的直线的合并）
    # theta:  角度θ的精度，值越小，考虑越多的线
    # thresh: 累加数阈值，值越小，考虑越多的线
    # minLineLength: 最短长度阈值，比这个长度短的线会被排除
    # maxLineGap:    直线间的最大距离
    # Example: lines = cv2.HoughLinesP(edges, 0.8, np.pi/180, 90, minLineLength=50, maxLineGap=10)
    return None if lines is None else [Line(p[:2], p[2:]) for p in lines[:,0]]

def find_circles(im, r_dist, threshold=100, canny_level=100, r_min=0, r_max=0):
    """
    return a list of circle info: [(cx,cy,r), ...]
    threshold:   param2, 数值越大，圆度要求越高
    canny_level: param1，参数越大，匹配成功度越高，且拟合程度更优
    """
    # https://docs.opencv.org/master/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d
    if DEBUG_MODE:
        assert im.dtype == "uint8"
    try:
        method = cv2.HOUGH_GRADIENT_ALT
    except AttributeError:
        method = cv2.HOUGH_GRADIENT
    circles = cv2.HoughCircles(im, method, dp=1, minDist=r_dist,
                            param1=canny_level, param2=threshold,
                            minRadius=r_min, maxRadius=r_max)
    if circles is None:
        return
    if DEBUG_MODE:
        print(f">>> 共计检测到【{len(circles[0,:])}】个Circle对象")
    return [Circle(*i) for i in circles[0,:]]

# find_circles(im, 200, 300, 50)

def match_template(im, template, threshold):
    """ 匹配符合最低阈值的全部点信息
    method:
        * cv2.TM_SQDIFF: 平方差匹配法：该方法采用平方差来进行匹配；最好的匹配值为0；匹配越差，匹配值越大。
        * cv2.TM_CCORR: 相关匹配法：该方法采用乘法操作；数值越大表明匹配程度越好。
        * cv2.TM_CCOEFF: 相关系数匹配法：1表示完美的匹配；-1表示最差的匹配。
        * cv2.TM_SQDIFF_NORMED: 归一化平方差匹配法
        * cv2.TM_CCORR_NORMED: 归一化相关匹配法
        * cv2.TM_CCOEFF_NORMED: 归一化相关系数匹配法
    """
    result = cv2.matchTemplate(im, template, cv2.TM_SQDIFF_NORMED)
    max_distance = min(template.shape[:2])
    # return simple_cluster(result, max_distance)

    # assert method == cv2.TM_SQDIFF_NORMED:
    loc = np.where(result < threshold)

    list_matches = []
    get_result = lambda tuple_pnt: result[tuple_pnt[0]][tuple_pnt[1]]

    for point in zip(*loc):
        # y, x = point
        new_group = True
        for index, k in enumerate(list_matches):
            p2p = np.linalg.norm(np.array([point,]) - np.array([k,]))
            if p2p < max_distance:
                # 基于cv2.TM_SQDIFF_NORMED，越小越准
                if get_result(point) < get_result(k):
                    list_matches[index] = point
                new_group = False
                break
        if new_group:
            list_matches.append(point)
    return list_matches

def find_template(im, template):
    """ 获取最优匹配度的信息 """
    result = cv2.matchTemplate(im, template, cv2.TM_CCOEFF_NORMED)
    pos = np.unravel_index(np.argmax(result), result.shape)  # 相似度最高的顶点位置
    similarity = result[pos[0]][pos[1]]
    return pos, similarity

def find_cnts(im, mode=0, method=1):
    """
    return: list of ndarray(dtype=int32)
        - one contours is like this: [[[234, 123]], [[345, 789]], ...]

    mode, 轮廓的检索模式:
        - cv2.RETR_EXTERNAL 表示只检测外轮廓
        - cv2.RETR_LIST 检测的轮廓不建立等级关系
        - cv2.RETR_CCOMP 建立两个等级的轮廓，上面的一层为外边界，里面的一层为内孔的边界信息。如果内孔内还有一个连通物体，这个物体的边界也在顶层。
        - cv2.RETR_TREE 建立一个等级树结构的轮廓【推荐】

    method, 轮廓的近似办法:
        - cv2.CHAIN_APPROX_NONE 存储所有的轮廓点，相邻的两个点的像素位置差不超过1，即max（abs（x1-x2），abs（y2-y1））==1
        - cv2.CHAIN_APPROX_SIMPLE 压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息
        - cv2.CHAIN_APPROX_TC89_L1，CV_CHAIN_APPROX_TC89_KCOS使用teh-Chinl chain 近似算法
    """
    if DEBUG_MODE:
        assert im.dtype == "uint8"  # np.uint8
        assert im.ndim == 2
    list_ret = cv2.findContours(im, mode, method)
    # hierarchy: 各层轮廓的索引
    # _, cnts, hierarchy = list_ret
    cnts = list_ret[-2]
    if DEBUG_MODE:
        print(f">>> 共计检测到【{len(cnts)}】个Contours对象")
    return cnts

def list2cnts(list_pnts):
    """ list_pnts: [QPointF(x,y), ...] or [(x,y), ...] """
    isTuple = True
    try:
        x, y = list_pnts[0]
    except TypeError:
        isTuple = False

    list_cnts = []  # 维度为3 --> (n,1,2)
    for point in list_pnts:
        if isTuple:
            x, y = point
        else:
            x = point.x()
            y = point.y()

        point = [[x, y]]  # 这个结构很特殊，多一层[]
        list_cnts.append(point)

    cnts = np.asarray(list_cnts, dtype="float32")  # 必须为np.float32
    return cnts

def bounding_box(cnt):
    """ 拟合矩形边框 """
    x, y, w, h = cv2.boundingRect(cnt)
    return (x, y, w, h)

def bounding_rect(cnt, extend_result=False):
    """ 最小外接矩形
        result_ext: ["center_pos", "shape", "angle"]
    """
    result = cv2.minAreaRect(cnt)  # return: [[中心坐标]、[宽度, 高度]、[旋转角度]]，其中，角度是度数形式，不是弧度数
    box = cv2.boxPoints(result)  # 获得角点坐标
    box = np.int0(box)  # np.uint8(box)... 额，uint8错误！
    # [[0,0], [0,1], [1,1], [1,0]]
    return (box, result) if extend_result else box

def get_box_sides(box):
    # 计算高和宽
    height = round(np.linalg.norm(box[0] - box[1]), 3)
    width = round(np.linalg.norm(box[0] - box[3]), 3)
    if height < width:
        height, width = width, height
    return (width, height)

def bounding_circle(cnt):
    """ 最小外接圆 """
    center, radius = cv2.minEnclosingCircle(cnt)  # (x,y) = center
    center = tuple(np.int0(center))  # 转换为整型
    radius = np.int0(radius)
    return (center, radius)

def approx_ellipse(cnt):
    """ 最小外接椭圆 """
    ellipse = cv2.fitEllipse(cnt)
    return ellipse

def approx_polygon(cnt, epsilon=0):  # 多边形拟合
    """
    void approxPolyDP(InputArray curve, OutputArray approxCurve, double epsilon, bool closed)
    参数:
        - InputArray curve:        一般是由图像的轮廓点组成的点集
        - OutputArray approxCurve: 表示输出的多边形点集
        - double epsilon:          表示逼近曲率，越小表示逼近精度越高
        - bool closed:             表示输出的多边形是否封闭
    """
    if epsilon <= 0:
        epsilon = 0.1 * cnt_perimeter(cnt)
    polygon = cv2.approxPolyDP(cnt, epsilon, True)
    return polygon[:,0]

def isConvex(cnt):  # 检测轮廓的凸性
    isConvex = cv2.isContourConvex(cnt)
    return isConvex

def approx_convex(cnt):  # 凸包
    """
    hull = cv2.convexHull(points, hull, clockwise, returnPoints)
    参数：
        - points:       轮廓
        - hull:         输出，通常不需要
        - clockwise:    方向标志，如果设置为True，输出的凸包是顺时针方向的，否则为逆时针方向
        - returnPoints: 默认值为True，它会返回凸包上点的坐标，如果设置为False，就会返回与凸包点对应的轮廓上的点
    """
    hull = cv2.convexHull(cnt)
    return hull[:,0]

def convex_defects(cnt):
    hull = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull)
    defect_pts = []
    for i in range(defects.shape[0]):
        # 特征向量：
        # * 起始点（startPoint）
        # * 结束点(endPoint)
        # * 距离convexity hull最远点(farPoint)
        # * 最远点到convexity hull的距离(depth)
        s,e,f,d = defects[i,0]
        # start = tuple(cnt[s][0])
        # end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        # cv2.line(img,start,end,[0,255,0],2)
        # cv2.circle(img,far,5,[0,0,255],-1)
        defect_pts.append((far, d))
    return defect_pts

#####################################################################
# 图形处理
#####################################################################

def cnt_area(cnt):
    area = cv2.contourArea(cnt)  # 获取面积
    return area

def cnt_perimeter(cnt):
    perimeter = cv2.arcLength(cnt, True)  # 闭合的形状
    return perimeter

def cnt_roundness(cnt):
    area = cnt_area(cnt)
    perimeter = cnt_perimeter(cnt)
    result = 4 * math.pi * area / math.pow(perimeter, 2)
    return result

def cnt_elongation(cnt):
    """ 比例范围从 0.1-10 映射为整数 """
    _, size, _ = cv2.minAreaRect(cnt)  # 最小外接矩形
    w, h = size
    if w < h:
        w, h = h, w
    ratio = h / w  # range: [0,1]
    return ratio

def dist_to_cnt(point, cnt):
    # If False, it finds whether the point is inside or outside
    # or on the contour (it returns +1, -1, 0 respectively).
    return cv2.pointPolygonTest(cnt, point, True)

def inside_cnt(point, cnt):
    return cv2.pointPolygonTest(cnt, point, False)

#####################################################################

class Shape:
    @property
    def cx(self):
        return self._cx
    @property
    def cy(self):
        return self._cy
    def center(self):
        return [self.cx, self.cy]

class Blob(Shape):
    def __init__(self, cnt):
        self._cnt = cnt
        # return: [[中心坐标]、[宽度, 高度]、[旋转角度]]，其中，角度是度数形式，不是弧度数
        pos, self.size, self.rotation_deg = cv2.minAreaRect(cnt)
        self._cx, self._cy = pos
        w, h = self.size
        if w < h:
            w, h = h, w
        self.elongation_ratio = h / w  # range: [0,1]

    @property
    def cnt(self):
        return self._cnt

    def rotation(self):
        """ deprecated: 并不保证准确 """
        return self.rotation_deg

    def rotation_rad(self):
        """ deprecated: 并不保证准确 """
        return math.radians(self.rotation_deg)

    def corners(self):
        list_pnts = approx_polygon(self._cnt)
        return list_pnts

    def area(self):
        return cnt_area(self._cnt)

    def perimeter(self):
        return cnt_perimeter(self._cnt)

    def roundness(self):
        """ 圆形接近1 """
        return cnt_roundness(self._cnt)

    def elongation(self):
        """ deprecated: 并不保证准确
            圆形接近1，直线接近0
        """
        return self.elongation_ratio

    def bounding(self):
        return bounding_box(self._cnt)

def find_blobs(im, thresholds=None, invert=False):
    """ thresholds: int or list(thresh, maxval) """
    if thresholds:
        im2 = binary(im, thresholds, invert)
    elif invert:
        im2 = bitwise_not(im)
    else:
        im2 = im
    list_cnts = find_cnts(im2, 1, 1)
    list_blobs = []
    for cnt in list_cnts:
        try:
            list_blobs.append(Blob(cnt))
        except ZeroDivisionError:
            pass
    if DEBUG_MODE:
        print(f">>> 共计检测到【{len(list_blobs)}】个Blob对象")
    return list_blobs

#####################################################################
# Drawing
# https://docs.opencv.org/master/d6/d6e/group__imgproc__draw.html
#####################################################################

"""
常用参数说明：
+ color:
    是用于灰度或RGB565图像的RGB888元组。默认为白色。
    也可以传递灰度图像的基础像素值(0-255)或RGB565图像的字节反转RGB565值。
+ thickness: 控制线的粗细像素。
"""

def draw_polygon(img, list_pnts, color=None, thickness=3, fill=False):
    # if DEBUG_MODE:
    #     list_pnts.dtype == ""
    if color is None:
        color = (255,0,0) if img.ndim == 3 else 255
    list_pnts = np.array([(int(p[0]), int(p[1])) for p in list_pnts])
    cv2.polylines(img, [list_pnts], True, color, thickness, lineType=cv2.LINE_AA)

def draw_line(img, a, b, color=None, thickness=3):
    if color is None:
        color = (255,0,0) if img.ndim == 3 else 255
    cv2.line(img, tuple(int(i) for i in a), tuple(int(i) for i in b),
            color, thickness, lineType=cv2.LINE_AA)  # 抗锯齿线型

def draw_lines(img, list_pnts, color=None, thickness=3):
    if color is None:
        color = (255,0,0) if img.ndim == 3 else 255
    list_pnts = np.array([(int(p[0]), int(p[1])) for p in list_pnts])
    cv2.polylines(img, [list_pnts], False, color, thickness, lineType=cv2.LINE_AA)

def draw_rect2(img, top_left, bottom_right, color=None, thickness=3, fill=False):
    if color is None:
        color = (255,0,0) if img.ndim == 3 else 255
    cv2.rectangle(img, tuple(int(i) for i in top_left),
            tuple(int(i) for i in bottom_right), color, thickness, lineType=cv2.LINE_AA)

def draw_rect(img, x, y, w, h, color=None, thickness=3, fill=False):
    draw_rect2(img, (x,y), (x+w,y+h), color, thickness, fill)

def draw_circle(img, x, y, radius, color=None, thickness=3, fill=False):
    if color is None:
        color = (255,0,0) if img.ndim == 3 else 255
    cv2.circle(img, (int(x),int(y)), int(radius), color, thickness, lineType=cv2.LINE_AA)

def draw_ellipse(img, center, axes, rotation: float, color=None, thickness=3, fill=False):
    """ center: tuple of point(x,y)
        axes:   tuple of (dx, dy) """
    if color is None:
        color = (255,0,0) if img.ndim == 3 else 255
    startAngle, endAngle = 0, 360
    cx, cy = [round(i) for i in center]
    rx, ry = [round(i/2) for i in axes]
    cv2.ellipse(img, (cx, cy), (rx, ry), rotation, startAngle, endAngle,
        color, thickness, lineType=cv2.LINE_AA)

def draw_string(img, x, y, text, scale=1, color=None, thickness=3, font=cv2.FONT_HERSHEY_SIMPLEX):
    if color is None:
        color = (255,0,0) if img.ndim == 3 else 255
    cv2.putText(img, text, (int(x),int(y)), font, scale, color, thickness)

draw_text = draw_string

def draw_contours(img, list_cnts, color=None, thickness=3, fill=False):
    if color is None:
        color = (255,0,0) if img.ndim == 3 else 255
    cv2.drawContours(img, list_cnts, -1, color, thickness, lineType=cv2.LINE_AA)

def draw_points(img, list_points, color=None):
    if color is None:
        color = (255,0,0) if img.ndim == 3 else 255
    cv2.drawKeypoints(img, list_points, None, color)

def draw_cross(img, x, y, color=None, size=5, thickness=3):
    if color is None:
        color = (255,0,0) if img.ndim == 3 else 255
    x,y = int(x), int(y)
    h_0, h_1 = (x-size,y), (x+size,y)
    v_0, v_1 = (x,y-size), (x,y+size)
    cv2.line(img, h_0, h_1, color, thickness, lineType=cv2.LINE_8)
    cv2.line(img, v_0, v_1, color, thickness, lineType=cv2.LINE_8)

#####################################################################

HSV_COLOR = {  # ["hmin", "hmax", "smin", "smax", "vmin", "vmax"]
    "black": [0, 180, 0, 255, 0, 46],
    "gray": [0, 180, 0, 43, 46, 220],
    "white": [0, 180, 0, 30, 221, 255],
    "red": [156, 10, 43, 255, 46, 255],
    "orange": [11, 25, 43, 255, 46, 255],
    "yellow": [26, 34, 43, 255, 46, 255],
    "green": [35, 77, 43, 255, 46, 255],
    "cyan": [78, 99, 43, 255, 46, 255],
    "blue": [100, 124, 43, 255, 46, 255],
    "purple": [125, 155, 43, 255, 46, 255],
}
