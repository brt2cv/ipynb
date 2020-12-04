#!/usr/bin/env python3
# @Date    : 2020-11-20
# @Author  : Bright Li (brt2@qq.com)
# @Link    : https://gitee.com/brt2
# @Version : 0.2.3

from importlib import reload
import traceback

from qt import *
import imgio
from utils.expy import path_expand
path_expand("../../..")

from tool.camera import Qt5Camera
import ocv as cv

import script

try:
    from utils.log import getLogger
    print("[+] {}: 启动调试Logger".format(__file__))
    logger = getLogger(30)
except ImportError:
    from logging import getLogger
    print("[!] {}: 调用系统logging模块".format(__file__))
    logger = getLogger(__file__)


class MainWnd(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        loadUi("ui/wx_mwnd.ui", self)

        self.isPaused = False
        self.isSwitched = True

        self.camera = Qt5Camera(0, [1920, 1080], isRGB=0)
        self._setup_ui()
        self.camera.start()

    def _setup_ui(self):
        self.setWindowTitle("HeroVision图像处理")
        self.move(0,0)

        self.controlers.setStyleSheet("""
QFrame {
    background-color: rgb(228, 231, 233);
    border-radius: 6px;
}""")

        left_win = [600,400]
        rup_win = [400,300]

        self.status_bar = QStatusBar(self)
        self.status_bar.showMessage("Welcome")
        self.footer.addWidget(self.status_bar)

        self.canvas = QLabel("Camera frame", self)
        self.canvas.setScaledContents(True)
        self.canvas.setFixedSize(*left_win)
        pixmap_label(self.canvas, QPixmap(*left_win))
        self.left.addWidget(self.canvas)
        self.camera.dataUpdated.connect(self.update_frame)

        self.processing = QLabel("Image Processing", self)
        self.processing.setScaledContents(True)
        self.processing.setFixedSize(*rup_win)
        pixmap_label(self.processing, QPixmap(*rup_win))
        self.rup.addWidget(self.processing)

        self.btn_update_script.clicked.connect(self.update_script)
        self.btn_pause.clicked.connect(self.camera_pause)
        self.btn_save_image.clicked.connect(self.save_image)
        self.btn_switch_wins.clicked.connect(self.switch_windows)

    def update_script(self):
        reload(script)

    def camera_pause(self):
        self.isPaused = not self.isPaused

    def save_image(self):
        pixmap = self.canvas.pixmap()
        im_arr = imgio.pixmap2ndarray(pixmap)
        cv.imsave(im_arr, "/home/brt/checker.jpg")

    def switch_windows(self):
        self.isSwitched = not self.isSwitched

    def closeEvent(self, event):
        super().closeEvent(event)
        self.camera.stop()

    def update_frame(self, im_arr):
        if self.isPaused:
            return
        left_win, rup_win = self.canvas, self.processing
        if self.isSwitched:
            left_win, rup_win = rup_win, left_win

        pixmap_cap = imgio.ndarray2pixmap(im_arr)
        left_win.setPixmap(pixmap_cap)

        try:
            im_proc = script.improc(im_arr)
            pixmap_proc = imgio.ndarray2pixmap(im_proc)
        except Exception:
            traceback.print_exc()
            pixmap_proc = QPixmap(400,300)
        rup_win.setPixmap(pixmap_proc)


if __name__ == "__main__":
    run_qtapp(MainWnd)
