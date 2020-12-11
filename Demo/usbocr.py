#!/usr/bin/env python3
# @Date    : 2020-12-04
# @Author  : Bright Li (brt2@qq.com)
# @Link    : https://gitee.com/brt2
# @Version : 0.2.2

from importlib import reload
import traceback

from pyqt.qtwx import *
from pyqt.qtcv import *
from pycv.camera import Qt5Camera
import pycv.opencv as cv
from ocr import OcrEngineMixin

import script

TessEnv = {
    # "TesseractBinPath": "D:/programs/Tesseract",
    # "TessDataDir": "/home/brt/ws/ipynb/tutorial/tesseract/tessdata",
    "TessDataDir": "D:/Home/workspace/ipynb/tutorial/tesseract/tessdata",
    "Lang": "xlt"
}

class MainWnd(OcrEngineMixin, BaseCvWnd):
    statusbar_msg = '请移动画面，将字符置于识别框中'

    def __init__(self, parent, camera_idx=0, solution=None, isRGB=False):
        super().__init__(parent, camera_idx, solution, isRGB)
        loadUi("ui/wx_mwnd_with_ctrllers.ui", self)
        super()._setup_ui()

        self.setWindowTitle("OCR字符识别")
        # self.set_roi([250, 200, 300, 100])
        self.ROI = True

        self.ocr_init(TessEnv["TessDataDir"], TessEnv["Lang"])

    def update_script(self):
        """ 由于关乎可变脚本script，故需要在子类重写 """
        reload(script)
        self.status_bar.showMessage("预处理脚本已更新")
        self.define_improc()

    def define_improc(self):
        """ 由于关乎可变脚本script，故需要在子类重写 """
        self.improc_methods = {
            "window2": script.improc_ocr,  # make_right
            "parse_roi": script.improc_ocr,
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

        # 绘制ROI区域
        if self.ROI:
            func = self.improc_methods.get("parse_roi")
            if func:
                im_text = func(im_left, *list_params)
                self.ocr_exec(im_text)

        if self.isSwitched:
            im_left, im_right = im_right, im_left

        im_resized = cv.rescale(im_right, 0.5)
        self.canvas.set_image(im_left)
        self.processing.set_image(im_resized)


# run_qtapp(SimpleOCR)
run_qtapp(MainWnd, None, solution=[800,600])


