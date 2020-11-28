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


class LabelCanvas(ImarrMgrMixin, QLabel):
    def __init__(self, parent):
        super().__init__(parent)  # Minin继续调用的super(), 即QLabel
        self.setAlignment(Qt.AlignCenter)

        self.scaled_with_aspect = False
        self.setPixmap(QPixmap())

    def setScaledContents(self, type_):
        if type_ == 1:
            super().setScaledContents(True)
        else:
            super().setScaledContents(False)
        self.scaled_with_aspect = type_ > 1

    def update_canvas(self):
        pixmap = asQPixmap(self.curr)
        if self.scaled_with_aspect:
            im_h, im_w = self.curr.shape[:2]
            aspect_ratio = im_h / im_w
            w, h = self.width(), self.height()
            w2 = h / aspect_ratio
            h2 = w * aspect_ratio
            pixmap = pixmap.scaled(min(w,w2), min(h,h2))
        self.setPixmap(pixmap)


class MainWnd(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        loadUi("ui/wx_mwnd_with_btns.ui", self)

        self.isPaused = False
        self.isSwitched = True

        self.camera = Qt5Camera(0, [640,480], isRGB=0)
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

        self.status_bar = QStatusBar(self)
        self.status_bar.showMessage("Welcome")
        self.footer.addWidget(self.status_bar)

        self.canvas = LabelCanvas(self)
        self.canvas.setScaledContents(2)
        self.left.addWidget(self.canvas)

        self.processing = LabelCanvas(self)
        self.processing.setScaledContents(1)
        self.rup.addWidget(self.processing)

        self.camera.dataUpdated.connect(self.update_frame)
        self.btn_update_script.clicked.connect(self.update_script)
        self.btn_pause.clicked.connect(self.camera_pause)
        self.btn_save_image.clicked.connect(self.save_image)
        self.btn_switch_wins.clicked.connect(self.switch_windows)

    def update_script(self):
        reload(script)
        self.status_bar.showMessage("预处理脚本已更新")

    def camera_pause(self):
        self.isPaused = not self.isPaused
        self.status_bar.showMessage("相机暂停" if self.isPaused else "相机恢复")

    def save_image(self):
        im_arr = self.canvas.get_image()
        path_save = dialog_file_select(self, default_suffix="jpg")
        if path_save:
            cv.imsave(im_arr, path_save[0])
        self.status_bar.showMessage(f"图像保存成功【{path_save[0]}】")

    def switch_windows(self):
        self.isSwitched = not self.isSwitched

    def update_frame(self, im_arr):
        if self.isPaused:
            return

        im_left = im_arr
        try:
            im_right = script.improc(im_left)
        except Exception:
            traceback.print_exc()
            im_right = im_left

        if self.isSwitched:
            im_left, im_right = im_right, im_left

        im_resized = cv.rescale(im_right, 0.5)
        self.canvas.set_image(im_left)
        self.processing.set_image(im_resized)


if __name__ == "__main__":
    run_qtapp(MainWnd)
