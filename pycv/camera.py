#!/usr/bin/env python3
# @Date    : 2020-12-23
# @Author  : Bright Li (brt2@qq.com)
# @Link    : https://gitee.com/brt2
# @Version : 0.3.2

from time import sleep
import numpy as np
import cv2

try:
    from utils.log import getLogger
    print("[+] {}: 启动调试Logger".format(__file__))
    logger = getLogger(10)
except ImportError:
    from logging import getLogger
    print("[!] {}: 调用系统logging模块".format(__file__))
    logger = getLogger(__file__)

try:
    from pycv.HikVision.MvCameraControl_class import *
    import threading
    ENABLE_MODULE_HIKVISION = True
except Exception:
    ENABLE_MODULE_HIKVISION = False


class ICamera:
    def __init__(self):
        """ 设置相机序号或GigE的IP地址 """
        self.isRGB = True
        self.fps_err = False
    def set_fps(self, fps:int):
        """ 设置帧率 """
    def set_resolution(self, resolution):
        """ 设置分辨率 """
    def set_format(self, isRGB):
        """ 设置色彩模式 """
    def set_exposure(self, value=None):
        """ 设置曝光参数 """
    def set_white_balance(self, value=None):
        """ 设置白平衡 """
    def take_snapshot(self):
        """ 抓取图像 """

class UsbCamera(ICamera):
    def __init__(self, n):
        super().__init__()
        self.cap = cv2.VideoCapture(n)  # n, cv2.CAP_DSHOW
        assert self.cap.isOpened()

    def set_fps(self, fps:int=0):
        if not fps:
            return
        # self.cap.set(cv2.CAP_PROP_FPS, fps)  # 测试无效，反而导致取图失败
        curr_fps = self.cap.get(cv2.CAP_PROP_FPS)  # 测试并不准确……额
        curr_fps = 60; print(">>> 由于OpenCV::cap.get(cv2.CAP_PROP_FPS)不准确，定义curr_fps估计值:", curr_fps)
        if fps < curr_fps:
            # self.fps_err = int(round(curr_fps / fps))  # 每隔fps_err帧显示一张图
            self.fps_err = 1/fps  # sleep(fps_err)
            print("[!] 帧率设置失败，尝试通过减少图像传输来降低帧率:", self.fps_err)
        else:
            if fps > curr_fps:
                print(">>> 不支持设置的高帧率，当前帧率为:", curr_fps)
            else:
                print(">>> 当前帧率设置为:", curr_fps)
            self.fps_err = False

    def set_resolution(self, resolution):
        """ resolution格式: [width, height] """
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        try:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))
        except AttributeError:  # if cv2.__version__ <= "3.2.0"
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))

        logger.info("[+] 摄像头像素设定为【{}x{}】".format(
                self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            ))

    def set_format(self, isRGB):
        self.isRGB = isRGB

    def set_exposure(self, value=None):
        if value:
            # where 0.25 means "manual exposure, manual iris"
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            print(">>> 已切换至【手动】曝光")
            self.cap.set(cv2.CAP_PROP_EXPOSURE, value)
            # 设置手动曝光后，除非重启，否则无法恢复为自动曝光模式

    def set_white_balance(self, value=None):
        if value:
            self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)

    def take_snapshot(self):
        isOK, im_frame = self.cap.read()
        assert isOK, "[!] 相机读取异常"

        if self.isRGB:
            im_frame = cv2.cvtColor(im_frame, cv2.COLOR_BGR2RGB)
        else:
            im_frame = cv2.cvtColor(im_frame, cv2.COLOR_BGR2GRAY)
        # pil_img = Image.fromarray(im_frame, "L")
        return im_frame

class HikCamera(ICamera):
    def __init__(self, n=None):
        assert ENABLE_MODULE_HIKVISION
        super().__init__()

        SDKVersion = MvCamera.MV_CC_GetSDKVersion()
        print ("SDKVersion[0x%x]" % SDKVersion)

        deviceList = MV_CC_DEVICE_INFO_LIST()
        tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE

        # ch:枚举设备 | en:Enum device
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        assert ret == 0, "enum devices fail! ret[0x%x]" % ret
        assert deviceList.nDeviceNum > 0, "find no device!"
        print ("Find %d devices!" % deviceList.nDeviceNum)

        if n is None:
            assert deviceList.nDeviceNum == 1, "[!] Camera index error!"
            n = 0
        else:
            assert n < deviceList.nDeviceNum, "[!] Camera index error!"

        # ch:创建相机实例 | en:Creat Camera Object
        cam = MvCamera()
        # ch:选择设备并创建句柄| en:Select device and create handle
        stDeviceList = cast(deviceList.pDeviceInfo[n], POINTER(MV_CC_DEVICE_INFO)).contents

        ret = cam.MV_CC_CreateHandle(stDeviceList)
        assert ret == 0, "create handle fail! ret[0x%x]" % ret

        # ch:打开设备 | en:Open device
        ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        assert ret == 0, "open device fail! ret[0x%x]" % ret

        # ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
        if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
            nPacketSize = cam.MV_CC_GetOptimalPacketSize()
            if int(nPacketSize) > 0:
                ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize",nPacketSize)
                assert ret == 0, "Warning: Set Packet Size fail! ret[0x%x]" % ret
            else:
                print ("Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)

        # ch:设置触发模式为off | en:Set trigger mode as off
        ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        assert ret == 0, "set trigger mode fail! ret[0x%x]" % ret

        # ch:获取数据包大小 | en:Get payload size
        stParam =  MVCC_INTVALUE()
        memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))

        ret = cam.MV_CC_GetIntValue("PayloadSize", stParam)
        assert ret == 0, "get payload size fail! ret[0x%x]" % ret
        nPayloadSize = stParam.nCurValue

        self.cap = cam
        self._stFrameInfo = MV_FRAME_OUT_INFO_EX()
        memset(byref(self._stFrameInfo), 0, sizeof(self._stFrameInfo))

        # ch:开始取流 | en:Start grab image
        ret = cam.MV_CC_StartGrabbing()
        assert ret == 0, "start grabbing fail! ret[0x%x]" % ret
        self.data_buf = (c_ubyte * nPayloadSize)()
        self.nDataSize = nPayloadSize

    def __del__(self):
        # ch:停止取流 | en:Stop grab image
        ret = self.cap.MV_CC_StopGrabbing()
        if ret != 0:
            del self.data_buf
            raise Exception ("stop grabbing fail! ret[0x%x]" % ret)

        # ch:关闭设备 | Close device
        ret = self.cap.MV_CC_CloseDevice()
        if ret != 0:
            del self.data_buf
            raise Exception("close deivce fail! ret[0x%x]" % ret)

        # ch:销毁句柄 | Destroy handle
        ret = self.cap.MV_CC_DestroyHandle()
        if ret != 0:
            del self.data_buf
            raise Exception("destroy handle fail! ret[0x%x]" % ret)
        del self.data_buf

    def take_snapshot(self):
        pData = byref(self.data_buf)
        ret = self.cap.MV_CC_GetOneFrameTimeout(pData, self.nDataSize, self._stFrameInfo, 1000)
        if ret == 0:
            # print ("get one frame: Width[%d], Height[%d], PixelType[0x%x], nFrameNum[%d]".format(
            #         self._stFrameInfo.nWidth, self._stFrameInfo.nHeight,
            #         self._stFrameInfo.enPixelType, self._stFrameInfo.nFrameNum))
            ndarray = np.asarray(pData._obj)
            return ndarray.reshape((self._stFrameInfo.nHeight, self._stFrameInfo.nWidth))  # 灰度图
            # ndarray = ndarray.reshape((self._stFrameInfo.nHeight, self._stFrameInfo.nWidth, 3))  # RGB
            # ndarray = cv2.cvtColor(ndarray , cv2.COLOR_RGB2BGR)
        else:
            print ("no data[0x%x]" % ret)

#####################################################################

from threading import Thread, Event
from PyQt5.QtCore import pyqtSignal, QObject

class Qt5Camera(QObject, Thread):
    dataUpdated = pyqtSignal(np.ndarray)  # PIL.Image.Image
    readError = pyqtSignal()

    def __init__(self):
        QObject.__init__(self)
        Thread.__init__(self)

        self.isRunning = Event()
        self.isRunning.set()
        self.isPause = Event()

    def conn_uvc(self, n, resolution=None, fps=0, isRGB=False):
        self.camera = UsbCamera(n)
        self.camera.set_fps(fps)
        self.camera.set_format(isRGB)
        self.camera.set_resolution(resolution if resolution else [640, 480])

    def conn_hik(self, n_or_ip=None):
        self.camera = HikCamera(n_or_ip)

    listen = Thread.start

    def stop(self):
        """ 线程外调用 """
        logger.debug("准备结束Camera线程...")
        self.isRunning.clear()
        self.join()  # 等待回收线程

    def pause(self, value: bool):
        return self.isPause.set() if value else self.isPause.clear()

    def run(self):
        logger.debug("启动Camera图像传输线程...")
        # fps_idx = 1
        while self.isRunning.is_set():
            if self.isPause.is_set():
                return

            # 如果帧率设置失败，手动降低显示速度
            if self.camera.fps_err:
                # print(">>>", self.camera.fps_err, fps_idx)
                # if fps_idx < self.camera.fps_err:
                #     fps_idx += 1
                #     continue
                # else:
                #     fps_idx = 1
                sleep(self.camera.fps_err)

            try:
                im_frame = self.camera.take_snapshot()
                if im_frame is None:
                    print("[-] Capture Nothing")
                    time.sleep(1)
                    continue
                self.dataUpdated.emit(im_frame)
            except Exception as e:
                logger.error(e)
                self.isRunning.clear()
                self.readError.emit()
import time

#####################################################################

if __name__ == "__main__":
    import os

    def run_cv2(camera_num, isRGB, img_size, win_size=None):
        camera = UsbCamera(camera_num)
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
