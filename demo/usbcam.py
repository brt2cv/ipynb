#!/usr/bin/env python3
# @Date    : 2020-12-04
# @Author  : Bright Li (brt2@qq.com)
# @Link    : https://gitee.com/brt2
# @Version : 0.2.1

import cv2
from pycv.camera import CameraByOpenCV

if __name__ == "__main__":
    import os

    def run_cv2(camera_num, isRGB, img_size, win_size=None):
        camera = CameraByOpenCV(camera_num)
        camera.set_format(isRGB)
        camera.set_resolution(img_size if img_size else [640, 480])
        # camera.set_white_balance(True)
        # camera.set_exposure(-9)

        i = 0  # 用于存储命名
        while True:
            im = camera.take_snapshot()
            if win_size:  # 存储原图时，需要保留原图数据
                im_win = cv2.resize(im, dsize=win_size)
            else:
                im_win = im

            key = cv2.waitKey(1) & 0xFF
            if key == 113:  # ord('q')
                break
            elif key in [32, 115]:  # space or ord("s")
                while True:
                    path_save = os.path.join(os.getcwd(), "cap_{}.jpg".format(i))
                    i += 1
                    if not os.path.exists(path_save):
                        cv2.imwrite(path_save, im)
                        print("[+] 已存储图像至:", path_save)
                        break
            cv2.imshow("OCC: OpenCV_Camera_Capture.py", im_win)

    def getopt():
        import argparse
        parser = argparse.ArgumentParser("Camera By OpenCV", description="读取相机")
        parser.add_argument("-n", "--camera_num", action="store", type=int, default=0, help="相机序号")
        parser.add_argument("-c", "--color_RGB", action="store_true", help="使用彩色相机并通过RGB输出")
        parser.add_argument("-r", "--camera_resolution", action="store", default="640x480", help="相机分辨率设置，格式: 640x480")
        parser.add_argument("-R", "--window_resolution", action="store", help="窗口显示分辨率设置，格式: 800x600")
        parser.add_argument("-C", "--console", action="store", help="终端运行，需要配合target_ip使用")
        parser.add_argument("-t", "--target_ip", action="store", help="目标IP地址及端口号，格式: 192.168.0.1:8889")
        return parser.parse_args()

    args = getopt()

    img_size = [int(i) for i in args.camera_resolution.split("x")]
    if args.window_resolution:
        win_size = [int(i) for i in args.window_resolution.split("x")]
        print(">>> 显示窗口尺寸缩放: {}".format(win_size))
    else:
        win_size = None
    print(">>> 使用【{}】模式".format("彩色" if args.color_RGB else "灰度"))

    if args.console or args.target_ip:
        raise NotImplementedError()
    # cam.take_snapshot(to_gray=(not args.color_RGB))
    run_cv2(args.camera_num, args.color_RGB, img_size, win_size)
