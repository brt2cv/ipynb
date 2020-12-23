#!/usr/bin/env python3
# @Date    : 2020-12-04
# @Author  : Bright Li (brt2@qq.com)
# @Link    : https://gitee.com/brt2
# @Version : 0.2.2

g = {
    "index": 0
}

import numpy as np
import pycv.opencv as cv

import cv2
_FaceClassifier = cv2.CascadeClassifier("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml" )

import face_recognition

class UtilFaker:
    def imshow(self, *args, **kwargs):
        """ 不显示图像 """
uu = UtilFaker()

def improc_origin(im, *args):
    # print(">>", im.shape)
    # return im
    im_resize = cv.resize(im, (800,600))

    # 生成图像序号
    cv.draw_string(im_resize, 30, 30, f"index={g['index']}", color=0)
    g["index"] += 1
    # print(">>", g["index"])

    return im_resize

def face_recognize(im):
    faceRects = _FaceClassifier.detectMultiScale(im, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    if len(faceRects):
        for faceRect in faceRects:
            x, y, w, h = faceRect
            cv2.rectangle(im, (x, y), (x + h, y + w), (0, 255, 0), 2)
    return im

def face_recognize2(im):
    # face_recognition.load_image_file(im)
    face_locations = face_recognition.face_locations(im)
    print(type(face_locations), face_locations)
    for loc in face_locations:
        h0, h1, w0, w1 = loc
        cv.draw_rect2(im, (w0, h0), (w1, h1))
    return im

def improc_right(im, *args):
    return face_recognize2(im)
    # return im

def improc_roi(im, *args):
    return im
    # im_h, im_w = im.shape[:2]
    # w, h = 352, 100
    # left, right = int(args[1]/255 * im_w), int(args[2]/255 * im_h)
    # ROI = [left, right, w, h]
    # cv.draw_rect(im, *ROI, thickness=2)
    # im_roi = cv.crop(im, ROI)

    # im_gau = cv.gaussian(im_roi, 3)
    # im_bin = cv.threshold(im_roi, args[0])
    # im_med = cv.median(im_bin, 5)
    im_next = cv.binary(im, 188, 1)
    im_next = cv.median(im_next, 5)

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

    # %% 2D放射变换
    im_core = cv.crop(im, max_blob.bounding())
    im_deg = cv.rotate(im_core, max_blob.rotation_deg)
    uu.imshow(im_deg)

    # return im_deg
    h, w = im_deg.shape

    # %% 文字区域
    im_roi = cv.crop(im_deg, (0,0,w,h//2))
    uu.imshow(im_roi)

    return im_roi
