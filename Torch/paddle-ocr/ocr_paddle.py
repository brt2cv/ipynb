#!/usr/bin/env python3
# @Date    : 2021-02-25
# @Author  : Bright (brt2@qq.com)
# @Link    : https://gitee.com/brt2

import os
import cv2

from camera import UsbCamera
from predict_system import *

import logging
logging.basicConfig(level=logging.DEBUG)

dict_args = {
    "det_model_dir": "./models/onnx/det_db/model.onnx",
    "cls_model_dir": "./models/onnx/cls/model.onnx",
    "rec_model_dir": "./models/onnx/rec_en_crnn/model.onnx",

    # params for prediction engine
    "use_gpu": True,
    "ir_optim": True,
    "use_tensorrt": False,
    "use_fp16": False,
    "gpu_mem": 500,

    # params for text detector
    "det_algorithm": 'DB',
    "det_limit_side_len": 960,
    "det_limit_type": 'max',
    # DB parmas
    "det_db_thresh": 0.3,
    "det_db_box_thresh": 0.5,
    "det_db_unclip_ratio": 1.6,
    "max_batch_size": 10,
    # EAST parmas
    "det_east_score_thresh": 0.8,
    "det_east_cover_thresh": 0.1,
    "det_east_nms_thresh": 0.2,
    # SAST parmas
    "det_sast_score_thresh": 0.5,
    "det_sast_nms_thresh": 0.2,
    "det_sast_polygon": False,

    # params for text recognizer
    "rec_algorithm": 'CRNN',
    "rec_image_shape": "3, 32, 320",
    # "rec_char_type": 'ch',
    "rec_char_type": 'EN',
    "rec_batch_num": 6,
    "max_text_length": 25,
    # "rec_char_dict_path": "./ppocr/ppocr_keys_v1.txt",  # 中英文
    "rec_char_dict_path": "./ppocr/en_dict.txt",  # 英文
    "use_space_char": True,
    "drop_score": 0.5,

    # params for text classifier
    "use_angle_cls": False,
    "cls_image_shape": "3, 48, 192",
    "label_list": ['0', '180'],
    "cls_batch_num": 6,
    "cls_thresh": 0.9,
    "enable_mkldnn": False,
    "use_pdserving": False,
}

from collections import namedtuple
PaddleOcrArgs = namedtuple("PaddleOcrArgs", dict_args)
ocr_args = PaddleOcrArgs(**dict_args)

text_sys = None  # TextSystem(ocr_args)

def img_proc(im):
    global text_sys
    if not text_sys:
        text_sys = TextSystem(ocr_args)

    starttime = time.time()
    dt_boxes, rec_res = text_sys(im)
    print(">>> Predict time: %.3fs\n[+] " % (time.time() - starttime), end="")

    for text, score in rec_res:
        print("{}, {:.3f}".format(text, score))

def run_cv2(camera_num, isRGB, img_size, win_size=None, fps=0, func_call=None):
    camera = UsbCamera(camera_num)
    camera.set_format(isRGB)
    camera.set_resolution(img_size if img_size else [640, 480])
    # camera.set_fps(fps)
    # camera.set_white_balance(True)
    # camera.set_exposure(-9)

    i = 0  # 用于存储命名
    while True:
        im = camera.take_snapshot()
        if win_size:  # 存储原图时，需要保留原图数据
            im_win = cv2.resize(im, dsize=win_size)
        else:
            im_win = im

        wait_time = 1000//fps if fps > 0 else 1
        key = cv2.waitKey(wait_time) & 0xFF
        if key == 113:  # ord('q')
            break
        elif key in [32, 115]:  # space or ord("s")
            while True:
                path_save = os.path.join(os.getcwd(), "cap_{}.jpg".format(i))
                i += 1
                if not os.path.exists(path_save):
                    cv2.imwrite(path_save, im)
                    print("[+] 已存储图像至:", path_save)
                    break
        # cv2.imshow("OCC: OpenCV_Camera_Capture.py", im_win)
        if func_call:
            func_call(im)

def getopt():
    import argparse
    parser = argparse.ArgumentParser("Camera By OpenCV", description="读取相机")
    parser.add_argument("-n", "--camera_num", action="store", type=int, default=0, help="相机序号")
    # parser.add_argument("-c", "--color_RGB", action="store_true", help="使用彩色相机并通过RGB输出")
    parser.add_argument("-r", "--camera_resolution", action="store", default="640x480", help="相机分辨率设置，格式: 640x480")
    parser.add_argument("-R", "--window_resolution", action="store", help="窗口显示分辨率设置，格式: 800x600")
    parser.add_argument("-f", "--fps", action="store", type=int, default=0, help="设置帧率")
    parser.add_argument("-C", "--console", action="store", help="终端运行，需要配合target_ip使用")
    parser.add_argument("-t", "--target_ip", action="store", help="目标IP地址及端口号，格式: 192.168.0.1:8889")
    return parser.parse_args()

args = getopt()

img_size = [int(i) for i in args.camera_resolution.split("x")]
if args.window_resolution:
    win_size = tuple([int(i) for i in args.window_resolution.split("x")])
    print(">>> 显示窗口尺寸缩放: {}".format(win_size))
else:
    win_size = None

# assert args.color_RGB == True
args.color_RGB = True  # EAST强制要求彩图
print(">>> 使用【{}】模式".format("彩色" if args.color_RGB else "灰度"))

if args.console or args.target_ip:
    raise NotImplementedError()
run_cv2(args.camera_num, args.color_RGB, img_size, win_size,
        args.fps, func_call=img_proc)
