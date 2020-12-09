#!/usr/bin/env python3
# @Date    : 2020-12-08
# @Author  : Bright Li (brt2@qq.com)
# @Link    : https://gitee.com/brt2
# @Version : 0.2.1

from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
# https://pillow.readthedocs.io/en/stable/reference/ImageDraw.html
from PIL.ImageOps import scale, invert, flip, mirror, grayscale, colorize, solarize

def imread(uri, as_gray=True):
    im = Image.open(uri)
    if as_gray:
        im = im.convert("L")  # grayscale
    return im

def imsave(im, path_save):
    return im.save(path_save)

def crop(im, roi):
    left, top, w, h = roi
    return im.crop([left, top, left+w, top+h])

def crop2(im, top_left, bottom_right):
    roi = [*top_left, *bottom_right]
    return im.crop(roi)

# def bitwise_not(im):
#     raise NotImplementedError()

bitwise_not = invert

def binary(im, thresholds, invert=False):
    thresh, maxval = _split_thresholds(thresholds)
    if maxval == 255:
        return im.point(lambda i: i < thresh and 255)
    else:
        middle = maxval - thresh + 1
        mask = [0] * thresh + [1] * middle + [0] * (255 - maxval)
        return im.point(mask, "1")

threshold = binary

def gaussian(im, k, sigma=0):
    return im.filter(ImageFilter.GaussianBlur)

def median(im, ksize: int):
    return im.filter(ImageFilter.MedianFilter)

def mean(im, kernal: tuple):
    return im.filter(ImageFilter.BLUR)

def smooth(im, more=False):
    return im.filter(ImageFilter.SMOOTH_MORE if more else ImageFilter.SMOOTH)

#####################################################################
# Transform
#####################################################################

def resize(im, output_shape, antialias=True):
    # 开启抗锯齿，耗时增加8倍左右
    resample = Image.ANTIALIAS if antialias else Image.NEAREST
    # pillow.size自成体系，无需多余处理（错上加错就OK了）
    return im.resize(output_shape, resample)

def split(im):
    return im.split()

def merge(bands):
    return Image.merge("RGB", bands)

# 色彩空间
# rgb2gray = grayscale
def rgb2gray(im):
    return im.convert('L')

# gray2rgb = colorize
def gray2rgb(im):
    return im.convert('RGB')

def _split_thresholds(thresholds):
    return [thresholds, 255] if isinstance(thresholds, int) else thresholds

def contrast(im, bright, contrast):
    im2 = im.copy()
    enhancer = ImageEnhance.Brightness(im2)
    enhancer.enhance(bright)
    enhancer_ = ImageEnhance.Contrast(im2)
    enhancer_.enhance(contrast)
    return im2

def sharpening(im, sigma=5):
    return im.filter(ImageFilter.SHARPEN)  # ImageFilter.DETAIL

def shapening_USM(im, sigma=5):
    return im.filter(ImageFilter.UnsharpMask)

#####################################################################
# Feature 特征处理
#####################################################################

def edges(im, thresholds):
    return im.filter(ImageFilter.FIND_EDGES)

def find_corners(im, quality, max_num=0, min_dis=0):
    return im.filter(ImageFilter.CONTOUR)

#####################################################################
# Drawing
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
        color = 255 if img.mode == "L" else (255,0,0)
    draw = ImageDraw.Draw(img)
    fill_color = color if fill else None
    draw.polygon(list_pnts, fill_color, color)

def draw_line(img, a, b, color=None, thickness=3):
    if color is None:
        color = 255 if img.mode == "L" else (255,0,0)
    draw = ImageDraw.Draw(img)
    draw.line([a, b], color, thickness)

def draw_lines(img, list_pnts, color=None, thickness=3):
    if color is None:
        color = 255 if img.mode == "L" else (255,0,0)
    draw = ImageDraw.Draw(img)
    draw.line(list_pnts, color, thickness)

def draw_rect2(img, top_left, bottom_right, color=None, thickness=3, fill=False):
    if color is None:
        color = 255 if img.mode == "L" else (255,0,0)
    draw = ImageDraw.Draw(img)
    fill_color = color if fill else None
    draw.rectangle([top_left, bottom_right], fill_color, color, thickness)

def draw_rect(img, x, y, w, h, color=None, thickness=3, fill=False):
    draw_rect2(img, (x,y), (x+w,y+h), color, thickness, fill)

def draw_circle(img, x, y, radius, color=None, thickness=3, fill=False):
    if color is None:
        color = 255 if img.mode == "L" else (255,0,0)
    draw = ImageDraw.Draw(img)
    fill_color = color if fill else None
    draw.ellipse((x-radius, y-radius, radius*2, radius*2), fill_color, color, thickness)

def draw_ellipse(img, center, axes, rotation: float, color=None, thickness=3, fill=False):
    """ center: tuple of point(x,y)
        axes:   tuple of (dx, dy) """
    if color is None:
        color = 255 if img.mode == "L" else (255,0,0)
    draw = ImageDraw.Draw(img)
    x, y = center
    ax_x, ax_y = axes
    fill_color = color if fill else None
    draw.ellipse((x, y, ax_x, ax_y), fill_color, color, thickness)

def draw_string(img, x, y, text, scale=1, color=None, thickness=3):
    if color is None:
        color = 255 if img.mode == "L" else (255,0,0)
    draw = ImageDraw.Draw(img)
    draw.text((x,y), text, color)

draw_text = draw_string

def draw_contours(img, list_cnts, color=None, thickness=3, fill=False):
    raise NotImplementedError()

def draw_points(img, list_points, color=None):
    if color is None:
        color = 255 if img.mode == "L" else (255,0,0)
    draw = ImageDraw.Draw(img)
    for p in list_points:
        draw.point(p, color)

def draw_cross(img, x, y, color=None, size=5, thickness=3):
    if color is None:
        color = 255 if img.mode == "L" else (255,0,0)
    x,y = int(x), int(y)
    h_0, h_1 = (x-size,y), (x+size,y)
    v_0, v_1 = (x,y-size), (x,y+size)

    draw = ImageDraw.Draw(img)
    draw.line([h_0, h_1], color, thickness)
    draw.line([v_0, v_1], color, thickness)
