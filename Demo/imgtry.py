#!/usr/bin/env python3
# @Date    : 2020-12-16
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
    def __init__(self, parent, camera_idx=0, solution=None, fps=0, isRGB=False):
        super().__init__(parent, camera_idx, solution, fps, isRGB)
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
            "window1": script.improc_origin,
            "window2": script.improc_right,
        }

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

        im_left = self.improc_methods["window1"](im_left, *list_params)

        if self.isSwitched:
            im_left, im_right = im_right, im_left

        im_resized = cv.rescale(im_right, 0.5)
        self.canvas.set_image(im_left)
        self.processing.set_image(im_resized)


if __name__ == "__main__":

    def getopt():
        import argparse

        parser = argparse.ArgumentParser("cam_img_processing", description="相机取图与图像预处理工具")
        parser.add_argument("-H", "--hik-camera", action="store_true", help="是否为海康工业相机")
        parser.add_argument("-u", "--uvc-index", action="store", type=int, default=0, help="如果使用UVC相机，输入index（默认为0）")
        parser.add_argument("-c", "--color_RGB", action="store_true", help="使用彩色相机并通过RGB输出")
        parser.add_argument("-r", "--camera-resolution", action="store", default="640x480", help="相机分辨率设置，格式: 640x480")
        # parser.add_argument("-R", "--window_resolution", action="store", help="窗口显示分辨率设置，格式: 800x600")
        parser.add_argument("-f", "--fps", action="store", type=int, default=0, help="设置相机的帧率")
        return parser.parse_args()

    args = getopt()
    camera_idx = -1 if args.hik_camera else args.uvc_index

    img_size = [int(i) for i in args.camera_resolution.split("x")]
    print(">>> 使用【{}】模式".format("彩色" if args.color_RGB else "灰度"))

    run_qtapp(MainWnd, None, camera_idx, solution=img_size, fps=args.fps, isRGB=args.color_RGB)
