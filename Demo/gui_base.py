#!/usr/bin/env python3
# @Date    : 2020-12-15
# @Author  : Bright Li (brt2@qq.com)
# @Link    : https://gitee.com/brt2
# @Version : 0.2.5

from pyqt.qtwx import *
from pycv.camera import Qt5Camera
import pycv.opencv as cv
import script

class MainWnd(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        loadUi("ui/wx_mwnd.ui", self)

        self.camera = Qt5Camera()
        self.camera.conn_uvc(0)

        self._setup_ui()
        self.camera.dataUpdated.connect(self.update_frame)
        self.camera.start()

    def _setup_ui(self):
        self.setWindowTitle("OpenCV图像处理")
        # self.setGeometry(100, 100, 800, 650)
        left_win = [600,400]
        rup_win = [400,300]
        self.move(0,0)

        self.status_bar = QStatusBar(self)
        self.status_bar.showMessage("Welcome")
        self.footer.addWidget(self.status_bar)

        self.canvas = QLabel("Camera frame", self)
        # self.canvas.setFixedSize(800,600)
        self.canvas.setScaledContents(True)
        pixmap_label(self.canvas, QPixmap(*rup_win))
        self.rup.addWidget(self.canvas)

        self.processing = QLabel("Image Processing", self)
        # self.processing.setFixedSize(800,600)
        self.processing.setScaledContents(True)
        pixmap_label(self.processing, QPixmap(*left_win))
        self.left.addWidget(self.processing)

    def closeEvent(self, event):
        super().closeEvent(event)
        self.camera.stop()

    def update_frame(self, im_arr):
        pixmap_cap = cv.ndarray2pixmap(im_arr)
        self.canvas.setPixmap(pixmap_cap)

        im_proc = script.improc(im_arr)
        pixmap_proc = cv.ndarray2pixmap(im_proc)
        self.processing.setPixmap(pixmap_proc)

if __name__ == "__main__":
    run_qtapp(MainWnd, None)
