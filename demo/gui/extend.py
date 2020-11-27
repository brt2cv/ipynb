#!/usr/bin/env python3
# @Date    : 2020-11-27
# @Author  : Bright Li (brt2@qq.com)
# @Link    : https://gitee.com/brt2
# @Version : 0.2.3

from importlib import reload
import traceback

from pyqt import *

from camera import Qt5Camera
import opencv as cv
import script

class ScrollCanvas(ScrollCanvasBase):
    def load_image(self, path_img):
        return cv.imread(path_img)

class MainWnd(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        loadUi("ui/wx_mwnd_with_btns.ui", self)

        self.isPaused = False
        self.isSwitched = True

        self.camera = Qt5Camera(0, [800, 600], isRGB=0)
        self._setup_ui()
        self.camera.dataUpdated.connect(self.update_frame)
        self.camera.start()

    def closeEvent(self, event):
        super().closeEvent(event)
        self.camera.stop()

    def _setup_ui(self):
        self.setWindowTitle("OpenCV图像处理")
        # self.setGeometry(100, 100, 800, 650)
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

        self.canvas = ScrollCanvas(self)
        self.canvas.set_image(np.zeros(left_win))
        self.left.addWidget(self.canvas)
        self.camera.dataUpdated.connect(self.update_frame)

        self.processing = ScrollCanvas(self)
        self.canvas.set_image(np.zeros(rup_win))
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
        im_arr = self.canvas.get_image()
        path_save = dialog_file_select(self, default_suffix=".jpg")
        cv.imsave(im_arr, path_save[0])

    def switch_windows(self):
        self.isSwitched = not self.isSwitched

    def update_frame(self, im_arr):
        if self.isPaused:
            return
        left_win, rup_win = self.canvas, self.processing
        if self.isSwitched:
            left_win, rup_win = rup_win, left_win
        left_win.set_image(im_arr)

        try:
            im_proc = script.improc(im_arr)
        except Exception:
            traceback.print_exc()
            im_proc = np.zeros(400,300)
        rup_win.set_image(im_proc)


if __name__ == "__main__":
    run_qtapp(MainWnd)
