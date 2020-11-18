#!/usr/bin/env python3
# @Date    : 2020-11-18
# @Author  : Bright Li (brt2@qq.com)
# @Link    : https://gitee.com/brt2
# @Version : 0.2.1

from qt import *
from camera import Qt5Camera
import imgio
import ocv as cv

class MainWnd(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        loadUi("ui/wx_mwnd.ui", self)

        self.camera = Qt5Camera(0, [800, 600], isRGB=0)
        self._setup_ui()
        self.camera.start()

    def _setup_ui(self):
        self.setWindowTitle("OpenCV图像处理")
        # self.setGeometry(100, 100, 800, 650)
        self.move(99,99)

        self.status_bar = QStatusBar(self)
        self.status_bar.showMessage("Welcome")
        self.footer.addWidget(self.status_bar)

        self.canvas = QLabel("Camera frame", self)
        self.canvas.setFixedSize(800,600)
        self.canvas.setScaledContents(True)
        pixmap_label(self.canvas, QPixmap(800, 600))
        self.left.addWidget(self.canvas)
        self.camera.dataUpdated.connect(self.update_frame)

        self.processing = QLabel("Image Processing", self)
        self.processing.setFixedSize(800,600)
        self.processing.setScaledContents(True)
        pixmap_label(self.processing, QPixmap(800, 600))
        self.rup.addWidget(self.processing)

    def closeEvent(self, event):
        super().closeEvent(event)
        self.camera.stop()

    def update_frame(self, im_arr):
        # draw = ImageDraw.Draw(pil_img)
        # draw.rectangle(self.ROI, width=2)
        pixmap_cap = imgio.ndarray2pixmap(im_arr)
        self.canvas.setPixmap(pixmap_cap)

        im_bin = cv.threshold_otsu(im_arr)
        pixmap_processing = imgio.ndarray2pixmap(im_bin)
        self.processing.setPixmap(pixmap_processing)


if __name__ == "__main__":
    run_qtapp(MainWnd)
