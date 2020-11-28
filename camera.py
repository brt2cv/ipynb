#!/usr/bin/env python3
# @Date    : 2020-11-28
# @Author  : Bright Li (brt2@qq.com)
# @Link    : https://gitee.com/brt2
# @Version : 0.2.4

import numpy as np
import cv2

try:
    from utils.log import getLogger
    print("[+] {}: 启动调试Logger".format(__file__))
    logger = getLogger(10)
except ImportError:
    from logging import getLogger
    print("[!] {}: 调用系统logging模块".format(__file__))
    logger = getLogger(__file__)


class CameraByOpenCV:
    def __init__(self, n):
        self.cap = cv2.VideoCapture(n)
        assert self.cap.isOpened()
        self.isRGB = True

    def set_resolution(self, resolution):
        """ resolution格式: [width, height] """
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        # self.cap.set(cv2.CAP_PROP_FPS, 20)
        # self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        logger.info("[+] 摄像头像素设定为【{}x{}】".format(
                self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            ))

    def set_format(self, isRGB):
        self.isRGB = isRGB

    def set_exposure(self, value=None):
        if value:
            # where 0.25 means "manual exposure, manual iris"
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            print(">>> 已切换至【手动】曝光")
            self.cap.set(cv2.CAP_PROP_EXPOSURE, value)
            # 设置手动曝光后，除非重启，否则无法恢复为自动曝光模式

    def set_white_balance(self, value=None):
        if value:
            self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)

    def take_snapshot(self):
        isOK, im_frame = self.cap.read()
        assert isOK, "[!] 相机读取异常"

        if self.isRGB:
            im_frame = cv2.cvtColor(im_frame, cv2.COLOR_BGR2RGB)
        else:
            im_frame = cv2.cvtColor(im_frame, cv2.COLOR_BGR2GRAY)
        # pil_img = Image.fromarray(im_frame, "L")
        return im_frame

#####################################################################

from threading import Thread, Event
from PyQt5.QtCore import pyqtSignal, QObject

class Qt5Camera(QObject, Thread):
    dataUpdated = pyqtSignal(np.ndarray)  # PIL.Image.Image

    def __init__(self, n, resolution=None, isRGB=True):
        QObject.__init__(self)
        Thread.__init__(self)

        self.isRunning = Event()
        self.isRunning.set()
        self.isPause = Event()

        self.camera = CameraByOpenCV(n)
        self.camera.set_format(isRGB)
        self.camera.set_resolution(resolution if resolution else [640, 480])

    listen = Thread.start

    def stop(self):
        """ 线程外调用 """
        logger.debug("准备结束Camera线程...")
        self.isRunning.clear()
        self.join()  # 等待回收线程

    def pause(self, value: bool):
        return self.isPause.set() if value else self.isPause.clear()

    def run(self):
        logger.debug("启动Camera图像传输线程...")
        while self.isRunning.is_set():
            if self.isPause.is_set():
                return
            im_frame = self.camera.take_snapshot()
            self.dataUpdated.emit(im_frame)

#####################################################################

if __name__ == "__main__":
    import os

    def run_cv2(camera_num, isRGB, img_size, win_size=None):
        camera = CameraByOpenCV(camera_num)
        camera.set_format(isRGB)
        camera.set_resolution(img_size if img_size else [640, 480])
        camera.set_white_balance(True)
        camera.set_exposure(-9)

        i = 0  # 用于存储命名
        while True:
            im = camera.take_snapshot()
            if win_size:  # 存储原图时，需要保留原图数据
                im_win = cv2.resize(im, dsize=win_size)
            else:
                im_win = im

            key = cv2.waitKey(1) & 0xFF
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
            cv2.imshow("OCC: OpenCV_Camera_Capture.py", im_win)

    def getopt():
        import argparse
        parser = argparse.ArgumentParser("Camera By OpenCV", description="读取相机")
        parser.add_argument("-n", "--camera_num", action="store", type=int, default=0, help="相机序号")
        parser.add_argument("-c", "--color_RGB", action="store_true", help="使用彩色相机并通过RGB输出")
        parser.add_argument("-r", "--camera_resolution", action="store", default="640x480", help="相机分辨率设置，格式: 640x480")
        parser.add_argument("-R", "--window_resolution", action="store", help="窗口显示分辨率设置，格式: 800x600")
        parser.add_argument("-C", "--console", action="store", help="终端运行，需要配合target_ip使用")
        parser.add_argument("-t", "--target_ip", action="store", help="目标IP地址及端口号，格式: 192.168.0.1:8889")
        return parser.parse_args()

    args = getopt()

    img_size = [int(i) for i in args.camera_resolution.split("x")]
    if args.window_resolution:
        win_size = [int(i) for i in args.window_resolution.split("x")]
        print(">>> 显示窗口尺寸缩放: {}".format(win_size))
    else:
        win_size = None
    print(">>> 使用【{}】模式".format("彩色" if args.color_RGB else "灰度"))

    if args.console or args.target_ip:
        raise NotImplementedError()
    # cam.take_snapshot(to_gray=(not args.color_RGB))
    run_cv2(args.camera_num, args.color_RGB, img_size, win_size)
