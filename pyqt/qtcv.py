#!/usr/bin/env python3
# @Date    : 2020-12-16
# @Author  : Bright Li (brt2@qq.com)
# @Link    : https://gitee.com/brt2
# @Version : 0.2.3

import numpy as np

from .qtwx import *
import pycv.opencv as cv

#####################################################################
# Transform
#####################################################################

def asQImage(im_arr):
    h, w = im_arr.shape[:2]
    if im_arr.ndim < 3:
        qimg_fmt = QImage.Format_Grayscale8
    elif im_arr.shape[2] == 3:  # RGB
        qimg_fmt = QImage.Format_RGB888
    elif im_arr.shape[2] == 4:  # RGBA
        qimg_fmt = QImage.Format_ARGB32
    else:
        raise NotImplementedError("未知的数据格式")
    # print(">>> QImage_Format:", {
    #         QImage.Format_Grayscale8: "Grayscale8",
    #         QImage.Format_RGB888: "RGB888",
    #         QImage.Format_ARGB32: "ARGB32"
    #     }[qimg_fmt])
    return QImage(im_arr.data, w, h, qimg_fmt)

def asQPixmap(im_arr):
    qimg = asQImage(im_arr)
    return QPixmap.fromImage(qimg)

#####################################################################
# Mixin
#####################################################################

class ImarrMgrMixin:
    imageUpdated = pyqtSignal()  # np.ndarray

    def __init__(self, *args, **kwargs):
        # print(">>>", *args, **kwargs)
        super().__init__(*args, **kwargs)
        self.curr = None
        self._snapshots = []  # Stack()
        self.imageUpdated.connect(self.update_canvas)

    @property
    def snapshot(self):
        return self.get_snapshot()

    @property
    def snapshots(self):
        return self._snapshots

    def get_snapshot(self):
        return self._snapshots[-1]

    def push_snapshot(self):
        self._snapshots.append(self.curr)

    take_snapshot = push_snapshot

    def pop_snapshot(self):
        return self._snapshots.pop()

    @property
    def image(self):
        return self.get_image()

    def get_image(self):
        return self.curr

    def set_image(self, im_arr):
        # assert isinstance(im_arr, np.ndarray), f"请传入np.ndarray的图像格式：【{type(im_arr)}】"
        self.curr = im_arr
        self.imageUpdated.emit()

    def update_canvas(self):
        raise NotImplementedError()

#####################################################################
# Widgets
#####################################################################

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
            w2 = round(h / aspect_ratio)
            h2 = round(w * aspect_ratio)
            pixmap = pixmap.scaled(min(w,w2), min(h,h2))
        self.setPixmap(pixmap)

#####################################################################
# Entrance
#####################################################################

from pycv.camera import Qt5Camera
from importlib import reload
import traceback

class BaseCvWnd(QWidget):
    statusbar_msg = 'Welcome'

    def __init__(self, parent, camera_idx=0, solution=None, fps=0, isRGB=False, roi=None):
        super().__init__(parent)

        self.isPaused = False
        self.isSwitched = False

        if solution is None:
            solution = [640,480]
        self.camera = Qt5Camera()
        if camera_idx < 0:
            self.camera.conn_hik(None)
        else:
            self.camera.conn_uvc(camera_idx, solution, fps, isRGB=isRGB)

        # self._setup_ui()  # 留待子类传入ui后调用 ??
        self.define_improc()
        self.camera.dataUpdated.connect(self.update_frame)
        self.camera.readError.connect(self.close)
        self.set_roi(roi)
        self.camera.start()

    def closeEvent(self, event):
        super().closeEvent(event)
        self.camera.stop()

    def _setup_ui(self):
        self.setWindowTitle("OpenCV图像处理")
        self.controlers.setStyleSheet("""
QFrame {
background-color: rgb(228, 231, 233);
border-radius: 6px;
}""")

        self.status_bar = QStatusBar(self)
        self.status_bar.showMessage(self.statusbar_msg)
        self.footer.addWidget(self.status_bar)

        self.list_params = []
        for i in range(3):
            wx = UnitSlider(self, f"param{i}", val_range=[0,256], val_default=128, isCheckbox=False)
            self.ctrller.addWidget(wx)
            self.list_params.append(wx)

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
        """ 由于关乎可变脚本script，故需要在子类重写 """
        # reload(script)
        # self.status_bar.showMessage("预处理脚本已更新")
        raise NotImplementedError()

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

    def define_improc(self):
        """ 由于关乎可变脚本script，故需要在子类重写 """
        # self.improc_methods = {
        #     "window2": script.improc,  # make_right
        #     "parse_roi": None,
        # }
        raise NotImplementedError()

    def update_frame(self, im_arr):
        if self.isPaused:
            return

        im_left = im_arr
        list_params = []
        for wx_slider in self.list_params:
            list_params.append(wx_slider.get_value())
        try:
            im_right = self.improc_methods["window2"](im_left, *list_params)
        except Exception:
            traceback.print_exc()
            im_right = im_left

        # 绘制ROI区域
        if self.ROI:
            cv.draw_rect(im_left, *self.ROI, thickness=2)
            func = self.improc_methods.get("parse_roi")
            if func:
                func(cv.crop(im_left, self.ROI), *list_params)

        if self.isSwitched:
            im_left, im_right = im_right, im_left

        im_resized = cv.rescale(im_right, 0.5)
        self.canvas.set_image(im_left)
        self.processing.set_image(im_resized)

    def set_roi(self, roi):
        self.ROI = roi


if __name__ == "__main__":
    try:
        import script
    except ImportError:
        def improc(im):
            return cv.threshold(im, 55)
    run_qtapp(BaseCvWnd, None)
