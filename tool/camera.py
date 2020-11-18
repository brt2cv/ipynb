#!/usr/bin/env python3
# @Date    : 2020-11-18
# @Author  : Bright Li (brt2@qq.com)
# @Link    : https://gitee.com/brt2
# @Version : 0.2.2

from threading import Thread, Event
from PyQt5.QtCore import pyqtSignal, QObject

import imgio
import numpy as np
import cv2

try:
    from utils.log import getLogger
    print("[+] {}: 启动调试Logger".format(__file__))
    logger = getLogger(0)
except ImportError:
    from logging import getLogger
    print("[!] {}: 调用系统logging模块".format(__file__))
    logger = getLogger(__file__)


class Qt5Camera(QObject, Thread):
    # dataUpdated = pyqtSignal(PIL.Image.Image)
    dataUpdated = pyqtSignal(np.ndarray)

    def __init__(self, n, resolution=None, isRGB=True):
        QObject.__init__(self)
        Thread.__init__(self)
        self.isRunning = Event()
        self.isRunning.set()
        self.isPause = Event()

        self.isRGB = isRGB
        self.cap = cv2.VideoCapture(n)
        assert self.cap.isOpened()

        if not resolution:
            resolution = [800,600]
        # cap_w, cap_h = resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

    # listen = Thread.start

    def take_snapshot(self, to_gray=False):
        isOK, im_frame = self.cap.read()
        assert isOK, "[!] 相机读取异常"

        if to_gray:
            im_frame = cv2.cvtColor(im_frame, cv2.COLOR_BGR2GRAY)

        # pil_img = Image.fromarray(im_frame, "L")
        # pil_img = imgio.ndarray2pillow(im_frame)
        return im_frame

    def stop(self):
        """ 线程外调用 """
        logger.debug("准备结束Camera线程...")
        self.isRunning.clear()
        self.join()  # 等待回收线程

    def run(self):
        logger.debug("启动Camera图像传输线程...")
        while self.isRunning.is_set():
            pil_frame = self.take_snapshot(to_gray=not self.isRGB)
            self.dataUpdated.emit(pil_frame)
