#!/usr/bin/env python3
# @Date    : 2020-11-16
# @Author  : Bright Li (brt2@qq.com)
# @Link    : https://gitee.com/brt2
# @Version : 0.0.4

import os
import math
import numpy as np
import cv2

try:
    from utils.log import getLogger
    print("[+] {}: 启动调试Logger".format(__file__))
    logger = getLogger(30)
except ImportError:
    from logging import getLogger
    print("[!] {}: 调用系统logging模块".format(__file__))
    logger = getLogger(__file__)

DEBUG_MODE = True

#####################################################################
# Math
#####################################################################

def distance(pnt1, pnt2):
    # return np.square(numpy.sum(numpy.square(vec1 - vec2)))
    return np.linalg.norm(pnt1 - pnt2)

#####################################################################
# IO module
#####################################################################

# from cv2 import imread
def imread(uri, as_gray=True):
    if DEBUG_MODE:
        assert os.path.exists(uri)

    mode = cv2.IMREAD_GRAYSCALE
    if not as_gray:
        mode = cv2.IMREAD_COLOR  # cv2.IMREAD_UNCHANGED
    return cv2.imread(uri, mode)

from cv2 import imwrite as imsave

def float2ubyte(im):
    return (im * 255).astype(np.uint8)

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

def crop(im, roi):
    """ roi: (x, y, w, h) """
    x, y, w, h = roi
    return im[y:y+h, x:x+w]

def crop2(im, top_left, bottom_right):
    x, y = top_left
    x2, y2 = bottom_right
    return crop(im, (x, y, x2-x, y2-y))

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
from cv2 import addWeighted  # 透明感
# def addWeighted(img1, 0.7, img2, 0.3, 0)

def _split_thresholds(thresholds):
    return [thresholds, 255] if isinstance(thresholds, int) else thresholds

def binary(im, thresholds, invert=False):
    """
    thresholds: int or list(thresh, maxval)
    type_:
        cv2.THRESH_BINARY
        cv2.THRESH_BINARY_INV
        cv2.THRESH_TRUNC
        cv2.THRESH_TOZERO
        cv2.THRESH_TOZERO_INV
    """
    type_ = cv2.THRESH_BINARY  # 0
    if invert:
        type_ += 1
    if thresholds == "otsu":
        type_ += cv2.THRESH_OTSU  # 8
        thresholds = [0,255]
    else:
        thresholds = _split_thresholds(thresholds)

    thresh, maxval = thresholds
    _, im2 = cv2.threshold(im, thresh, maxval, type_)
    return im2

threshold = binary

def threshold_otsu(im, invert=False):
    return binary(im, "otsu", invert)

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
#####################################################################

def gaussian(im, sigma):
    """
    sigma: scalar or sequence of scalars
    ksize.width和ksize.height必须为正奇数，也可以为零，然后根据sigma计算得出
    sigmas可以为零，则分别从ksize.width和ksize.height计算得出
    """
    if isinstance(sigma, int):
        sigma_x = sigma_y = sigma
    else:
        sigma_x, sigma_y = sigma
    return cv2.GaussianBlur(im, None, sigma_x, sigmaY=sigma_y)

def median(im, k: int):
    """ k: 必须为奇数 """
    assert k % 2, "k值必须为奇数"
    return cv2.medianBlur(im, k)

def mean(im, k: tuple):
    """ 均值滤波 """
    return cv2.blur(im, k)

#####################################################################
# Morpholopy 形态学操作
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
    nShape = KERNEL_SHAPE_OPENCV[shape]
    return cv2.getStructuringElement(nShape, size)

# def erosion(im, k):
from cv2 import erode

# def dilation(im, k):
from cv2 import dilate

# 先腐蚀再膨胀，消除小物体或小斑块
opening = lambda im, k: cv2.morphologyEx(im, cv2.MORPH_OPEN, k)

# 先膨胀再腐蚀，填充孔洞
closing = lambda im, k: cv2.morphologyEx(im, cv2.MORPH_CLOSE, k)

# 梯度：图像的膨胀和腐蚀之间的差异，结果看起来像目标的轮廓
gradient = lambda im, k: cv2.morphologyEx(im, cv2.MORPH_GRADIENT, k)

# 顶帽：原图像减去它的开运算值，突出原图像中比周围亮的区域
tophat = lambda im, k: cv2.morphologyEx(im, cv2.MORPH_TOPHAT, k)

# 黑帽：原图像减去它的闭运算值，突出原图像中比周围暗的区域
blackhat = lambda im, k: cv2.morphologyEx(im, cv2.MORPH_BLACKHAT, k)

#####################################################################
# Feature 特征处理
#####################################################################

def canny(im, thresholds):
    """ thresholds: int or list(thresh, maxval) """
    thresholds = _split_thresholds(thresholds)
    return cv2.Canny(im, *thresholds)

edges = canny

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
    canny_level: param1
    threshold:   param2
    """
    # https://docs.opencv.org/master/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d
    if DEBUG_MODE:
        assert im.dtype == "uint8"
    circles = cv2.HoughCircles(im, cv2.HOUGH_GRADIENT, dp=1, minDist=r_dist,
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
        * CV_TM_SQDIFF: 平方差匹配法：该方法采用平方差来进行匹配；最好的匹配值为0；匹配越差，匹配值越大。
        * CV_TM_CCORR: 相关匹配法：该方法采用乘法操作；数值越大表明匹配程度越好。
        * CV_TM_CCOEFF: 相关系数匹配法：1表示完美的匹配；-1表示最差的匹配。
        * CV_TM_SQDIFF_NORMED: 归一化平方差匹配法
        * CV_TM_CCORR_NORMED: 归一化相关匹配法
        * CV_TM_CCOEFF_NORMED: 归一化相关系数匹配法
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
    result = cv2.matchTemplate(im, template, cv2.TM_SQDIFF_NORMED)
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
        - cv2.RETR_TREE 建立一个等级树结构的轮廓。

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
    except:
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

def approx_bounding(cnt):
    """ 拟合矩形边框 """
    x, y, w, h = cv2.boundingRect(cnt)
    return (x, y, w, h)

def approx_rect(cnt, extend_result=False):
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

def approx_circle(cnt):
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
        - double epsilon:          主要表示输出的精度，就是另个轮廓点之间最大距离数
        - bool closed:             表示输出的多边形是否封闭
    """
    if epsilon <= 0:
        epsilon = 0.1 * cnt_perimeter(cnt)
    polygon = cv2.approxPolyDP(cnt, epsilon, True)
    return polygon

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

class Blob:
    def __init__(self, cnt):
        self._cnt = cnt
        # return: [[中心坐标]、[宽度, 高度]、[旋转角度]]，其中，角度是度数形式，不是弧度数
        pos, size, self.rotation_deg = cv2.minAreaRect(cnt)
        self._cx, self._cy = pos
        w, h = size
        if w < h:
            w, h = h, w
        self.elongation_ratio = h / w  # range: [0,1]

    def center(self):
        return [self.cx, self.cy]

    @property
    def cx(self):
        return self._cx

    @property
    def cy(self):
        return self._cy

    @property
    def cnt(self):
        return self._cnt

    def rotation(self):
        """ deprecated: 并不保证准确 """
        logger.warning("Blob.rotation() 并不保证准确，请谨慎使用")
        return self.rotation_deg

    def rotation_rad(self):
        """ deprecated: 并不保证准确 """
        logger.warning("Blob.rotation() 并不保证准确，请谨慎使用")
        return math.radians(self.rotation_deg)

    def corners(self):
        raise NotImplemented()
        return approx_convex(self._cnt)

    def area(self):
        return cnt_area(self._cnt)

    def perimeter(self):
        return cnt_perimeter(self._cnt)

    def roundness(self):
        return cnt_roundness(self._cnt)

    def elongation(self):
        """ deprecated: 并不保证准确 """
        logger.warning("Blob.rotation() 并不保证准确，请谨慎使用")
        return self.elongation_ratio

    def bounding(self):
        return approx_bounding(self._cnt)

    def rect(self, extend_result=False):
        return approx_rect(self._cnt, extend_result)

    def circle(self):
        return approx_circle(self._cnt)

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

def draw_polygon(img, list_pnts, color=None, thickness=1, fill=False):
    # if DEBUG_MODE:
    #     list_pnts.dtype == ""
    if color is None:
        color = (255,0,0) if img.ndim == 3 else 255
    list_pnts = np.array([(int(p[0]), int(p[1])) for p in list_pnts])
    cv2.polylines(img, [list_pnts], True, color, thickness, lineType=cv2.LINE_AA)

def draw_line(img, a, b, color=None, thickness=1):
    if color is None:
        color = (255,0,0) if img.ndim == 3 else 255
    cv2.line(img, tuple(int(i) for i in a), tuple(int(i) for i in b),
            color, thickness, lineType=cv2.LINE_AA)  # 抗锯齿线型

def draw_lines(img, list_pnts, color=None, thickness=1, fill=False):
    if color is None:
        color = (255,0,0) if img.ndim == 3 else 255
    list_pnts = np.array([(int(p[0]), int(p[1])) for p in list_pnts])
    cv2.polylines(img, [list_pnts], False, color, thickness, lineType=cv2.LINE_AA)

def draw_rectangle2(img, top_left, bottom_right, color=None, thickness=1, fill=False):
    if color is None:
        color = (255,0,0) if img.ndim == 3 else 255
    cv2.rectangle(img, tuple(int(i) for i in top_left),
            tuple(int(i) for i in bottom_right), color, thickness, lineType=cv2.LINE_AA)

def draw_rectangle(img, x, y, w, h, color=None, thickness=1, fill=False):
    draw_rectangle2(img, (x,y), (x+w,y+h), color, thickness, fill)

def draw_circle(img, x, y, radius, color=None, thickness=1, fill=False):
    if color is None:
        color = (255,0,0) if img.ndim == 3 else 255
    cv2.circle(img, (int(x),int(y)), int(radius), color, thickness, lineType=cv2.LINE_AA)

def draw_ellipse(img, cx, cy, rx, ry, rotation, color=None, thickness=1, fill=False):
    if color is None:
        color = (255,0,0) if img.ndim == 3 else 255
    startAngle, endAngle = 0, 360
    cv2.ellipse(img, (int(cx),int(cy)), (int(rx),int(ry)), rotation, startAngle, endAngle,
        color, thickness, lineType=cv2.LINE_AA)

def draw_string(img, x, y, text, scale=1, color=None, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    if color is None:
        color = (255,0,0) if img.ndim == 3 else 255
    cv2.putText(img, text, (int(x),int(y)), font, scale, color, thickness)

def draw_contours(img, list_cnts, color=None, thickness=1, fill=False):
    if color is None:
        color = (255,0,0) if img.ndim == 3 else 255
    cv2.drawContours(img, list_cnts, -1, color, thickness, lineType=cv2.LINE_AA)

def draw_points(img, list_points, color=None, thickness=1, fill=False):
    if color is None:
        color = (255,0,0) if img.ndim == 3 else 255
    cv2.drawKeypoints(img, list_points, None, color)

def draw_cross(img, x, y, color=None, size=5, thickness=1):
    if color is None:
        color = (255,0,0) if img.ndim == 3 else 255
    x,y = int(x), int(y)
    h_0, h_1 = (x-size,y), (x+size,y)
    v_0, v_1 = (x,y-size), (x,y+size)
    cv2.line(img, h_0, h_1, color, thickness, lineType=cv2.LINE_8)
    cv2.line(img, v_0, v_1, color, thickness, lineType=cv2.LINE_8)


#####################################################################

COLOR_RANGES_HSV = {
    "red": [(0, 50, 10), (10, 255, 255)],
    "orange": [(10, 50, 10), (25, 255, 255)],
    "yellow": [(25, 50, 10), (35, 255, 255)],
    "green": [(35, 50, 10), (80, 255, 255)],
    "cyan": [(80, 50, 10), (100, 255, 255)],
    "blue": [(100, 50, 10), (130, 255, 255)],
    "purple": [(130, 50, 10), (170, 255, 255)],
    "red ": [(170, 50, 10), (180, 255, 255)]
}