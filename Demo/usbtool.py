#!/usr/bin/env python3
# @Date    : 2020-12-09
# @Author  : Bright Li (brt2@qq.com)
# @Link    : https://gitee.com/brt2


from importlib import reload
import traceback

from pyqt.qtwx import *
from pyqt.qtcv import *
from pycv.camera import Qt5Camera
import pycv.opencv as cv

import script


class MainWnd(BaseCvWnd):
    def __init__(self, parent, camera_idx=0, solution=None, isRGB=False):
        super().__init__(parent, camera_idx, solution, isRGB)
        loadUi("ui/wx_mwnd_with_ctrllers.ui", self)
        super()._setup_ui()
        self.setWindowTitle("OpenCV 图像处理")

    def update_script(self):
        """ 由于关乎可变脚本script，故需要在子类重写 """
        reload(script)
        self.define_improc()
        self.status_bar.showMessage("预处理脚本已更新")


    def define_improc(self):
        """ 由于关乎可变脚本script，故需要在子类重写 """
        self.improc_methods = {
            "window2": script.improc_roi,  # make_right
        }

    def ocr_result(self, result):
        # logger.debug(">>> OCR: {}".format(result))
        msg = f'OCR识别结果: {result}' if result else self.statusbar_msg
        self.status_bar.showMessage(msg)

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

        if self.isSwitched:
            im_left, im_right = im_right, im_left

        im_resized = cv.rescale(im_right, 0.5)
        self.canvas.set_image(im_left)
        self.processing.set_image(im_resized)


# run_qtapp(SimpleOCR)
run_qtapp(MainWnd, None, solution=[800,600])
