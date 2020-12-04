#!/usr/bin/env python3
# @Date    : 2020-12-02
# @Author  : Bright Li (brt2@qq.com)
# @Link    : https://gitee.com/brt2
# @Version : 0.2.5

try:
    from utils.log import getLogger
    print("[+] {}: 启动调试Logger".format(__file__))
    logger = getLogger(1)
except ImportError:
    from logging import getLogger
    print("[!] {}: 调用系统logging模块".format(__file__))
    logger = getLogger(__file__)

#####################################################################
import os
from PIL import Image

try:
    import tesserocr
    from tesserocr import PyTessBaseAPI, RIL
    HasImportTesserocr = True
except ImportError:
    HasImportTesserocr = False

try:
    from io import BytesIO
    from aip import AipOcr
    HasImportBaiduApi = True
except ImportError:
    HasImportBaiduApi = False

class OcrEngine:

    def destroy(self):
        """ 注销OCR解析引擎 """

    def img2text(self, pil_img):
        """ 解析PIL图像数据 """

    def bytes2text(self, data_bytes, mode, size):
        """ 字节流OCR解析 """
        pil_img = Image.frombytes(mode, size, data_bytes)
        return self.img2text(pil_img)

    def ndarray2text(self, im_arr, mode=None):
        """ numpy图像格式解析 """
        if mode is None:
            if im_arr.ndim < 3:
                mode = "L"
            else:
                shape = im_arr.shape
                if shape[2] == 3:
                    return "RGB"  # 无法区分BGR (OpenCV)
                elif shape[2] == 4:
                    return "RGBA"
                else:
                    raise Exception("未知的图像类型")
        pil_img = Image.fromarray(im_arr, mode)
        return self.img2text(pil_img)


class BaiduOcrApi(OcrEngine):
    def __init__(self, app_id, api_key, secret_key):
        assert HasImportBaiduApi
        self.client = AipOcr(app_id, api_key, secret_key)

    def _parse(self, pil_img, func):
        with BytesIO() as im_bytes:
            pil_img.save(im_bytes, format="JPEG")
            im_bytes.seek(0)
            dict_res = func(im_bytes.read())
            # im_bytes.write(b"save.jpg")

        # pil_img.save("save.jpg")
        # with open("save.jpg", "rb") as fp:
        #     dict_res = self.client.basicGeneral(fp.read())

        if "error_msg" in dict_res:
            # Example: {'log_id': 1307160134551601152, 'error_msg': 'empty image', 'error_code': 216200}
            logger.error("[-] Error: {}".format(dict_res["error_msg"]))
            return ""
        # else: {'words_result': [{'words': '生日快乐'}], 'log_id': 1307160671200215040, 'words_result_num': 1}

        list_results = [d["words"] for d in dict_res["words_result"]]
        # list_results.remove("")
        return "\n".join(list_results)

    def img2text(self, pil_img):
        """ 每日限制50000次调用 """
        return self._parse(pil_img, self.client.basicGeneral)

    def img2text_ex(self, pil_img):
        """ 每日限制300次调用 """
        return self._parse(pil_img, self.client.basicAccurate)

class Tesseract(OcrEngine):
    """ 对PIL友好的Tesseract封装 """
    def __init__(self, dir_tessdata, lang):
        assert HasImportTesserocr
        self.isRunning = True

        # 对于portable版本的Tesseract，需要在PyTessBaseAPI对象初始化时，指定path
        assert os.path.exists(dir_tessdata)
        self.api = PyTessBaseAPI(path=dir_tessdata, lang=lang)
        self.config = {}

    def __del__(self):
        if self.isRunning:
            self.destroy()

    def version(self):
        print("python-tesserocr Version:\n", tesserocr.tesseract_version())

    def destroy(self):
        logger.debug("释放Tesseract引擎")
        self.api.End()
        self.isRunning = True

    def recognize(self):
        text = self.api.GetUTF8Text()
        text = text.strip()
        # text = text.strip("\n").strip()
        # text = text
        # if not text.strip():
        #     return ""
        if "confidence" in self.config:
            isValid = self.check_confidence(text)
            if not isValid:
                return ""
        return text

    def detect_recognize(self):
        """ 分析页面，再进行OCR识别，适合全页面检测 """
        list_text = []
        boxes = self.api.GetComponentImages(RIL.TEXTLINE, True)
        for _, box, _, _ in boxes:
            # box is a dict with x, y, w and h keys
            for k, v in box.items():
                box[k] = int(v)

            # widening the box with offset can greatly improve the text output
            offset = box['h'] // 3  # 无法准确获取到边界补偿值 ??

            x, y = max(box['x'] - offset, 0), max(box['y'] - offset, 0)
            w, h = box['w'] + 2*offset, box['h'] + 2*offset

            # if "position" in self.config:
            #     draw_text(im, str([x,y,w,h]), (x,y-1), 16, DRAW_TEXT_SIZE, (255,0,255), 1)

            self.api.SetRectangle(x, y, w, h)
            text = self.recognize()
            if text:
                list_text.append(text)

        return "\n".join(list_text)

    def check_confidence(self, text):
        list_confidence = self.api.AllWordConfidences()
        logger.debug(f"OCR：【{text}】，置信度：【{list_confidence}】")

        isValid = True
        for confidence in list_confidence:
            if confidence < self.config["confidence"]:
                isValid = False
                break
        return isValid

    def filter_confidence(self, confidence: int):
        """ 目前仍为测试版本，可能降低识别率和识别准度 """
        if confidence > 0:
            self.config["confidence"] = confidence
        elif "confidence" in self.config:
            del self.config["confidence"]

    def img2text(self, pil_img):
        self.api.SetImage(pil_img)
        res = self.recognize()
        return res

    def file2text(self, path_img):
        self.api.SetImageFile(path_img)
        # res = self.api.GetUTF8Text()
        # score = self.api.AllWordConfidences()
        res = self.recognize()
        return res


#####################################################################

from threading import Thread, Event
from PyQt5.QtCore import pyqtSignal, QObject

# class KeepThreadingMixin(Thread):
#     def __init__(self):
#         super().__init__()
#         self.isRunning = Event()
#         self.isRunning.set()
#         self.isPause = Event()

#     def stop(self):
#         """ 线程外调用 """
#         logger.debug(f"准备结束【{self.__class__.__name__}】线程...")
#         self.isRunning.clear()
#         self.join()  # 等待回收线程

#     def pause(self, value: bool):
#         return self.isPause.set() if value else self.isPause.clear()

#     def run(self):
#         logger.debug(f"启动【{self.__class__.__name__}】线程...")
#         while self.isRunning.is_set():
#             if self.isPause.is_set():
#                 return
#             self.exec()

#     def exec(self):
#         raise NotImplementedError()

import queue

class TesseractThread(QObject, Thread, OcrEngine):
    """ 提供了独立的Python线程运行，以pyqtSignal异步的方式通知result """
    textRecognized = pyqtSignal(str)
    buff_size = 3

    def __init__(self, dir_tessdata, lang):
        QObject.__init__(self)
        OcrEngine.__init__(self)
        Thread.__init__(self, daemon=True)
        self.isRunning = Event()
        self.isRunning.set()
        self.isPause = Event()

        self.img_buff = queue.Queue(maxsize=self.buff_size)
        # self.img_count = self.buff_size
        self.ocr_engine = Tesseract(dir_tessdata, lang)

    def stop(self):
        """ 线程外调用 """
        logger.debug(f"准备结束【{self.__class__.__name__}】线程...")
        self.isRunning.clear()
        self.join()  # 等待回收线程

    def pause(self, value: bool):
        return self.isPause.set() if value else self.isPause.clear()

    def run(self):
        logger.debug(f"启动【{self.__class__.__name__}】线程...")
        while self.isRunning.is_set():
            if self.isPause.is_set():
                return
            self.exec()

    def exec(self):
        pil_img = self.img_buff.get(block=True)
        res = self.ocr_engine.img2text(pil_img)
        if res:
            # print("[+] 字符识别结果：", res)
            self.textRecognized.emit(res)

    def img2text(self, pil_img):
        if self.img_buff.full():
            # print(">>> Tesseract引擎满负载")
            return
        self.img_buff.put(pil_img)

    def ndarray2text(self, im_arr, mode=None):
        """ numpy图像格式解析 """
        if self.img_buff.full():
            # print(">>> Tesseract引擎满负载")
            return
        super().ndarray2text(im_arr)


# import multiprocessing

# class TesseractProcess(multiprocessing.Process, OcrEngine):
#     buff_size = 3

#     def __init__(self, dir_tessdata, lang):
#         super().__init__(daemon=True)
#         self.ocr_engine = Tesseract(dir_tessdata, lang)
#         self.img_buff = multiprocessing.Queue(maxsize=self.buff_size)

#     def run(self):
#         while True:
#             pil_img = self.img_buff.get(block=True)
#             res = self.ocr_engine.img2text(pil_img)
#             if res:
#                 print("[+] 字符识别结果：", res)
#                 # 需要通过 Pipe 输出信息 ??

#     def img2text(self, pil_img):
#         if self.img_buff.full():
#             print(">>> Tesseract引擎满负载")
#             return
#         self.img_buff.put(pil_img)

#     def ndarray2text(self, im_arr, mode=None):
#         """ numpy图像格式解析 """
#         if self.img_buff.full():
#             print(">>> Tesseract引擎满负载")
#             return
#         super().ndarray2text(im_arr)


class OcrEngineMixin:
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)

    def ocr_init(self, dir_tessdata, lang):
        self.ocr_engine = TesseractThread(dir_tessdata, lang)
        self.ocr_engine.textRecognized.connect(self.ocr_result)
        self.ocr_engine.start()

    def ocr_exec(self, im_arr):
        """ 异步处理，故而无返回值 """
        self.ocr_engine.ndarray2text(im_arr)

    def ocr_result(self, result):
        """ 回调函数，用于处理result结果 """
        logger.debug(">>> OCR: {}".format(result))
        # msg = f'OCR识别结果: {result}' if result else self.statusbar_msg
        # self.status_bar.showMessage(msg)


#####################################################################

if __name__ == "__main__":

    from importlib import reload
    import traceback
    from qtwx import *
    from qtcv import BaseCvWnd
    from camera import Qt5Camera
    import opencv as cv
    import script

    CAMERA_RESOLUTION = [800,600]

    class SimpleOCR(QMainWindow):
        statusbar_msg = '请移动画面，将字符置于识别框中'

        def __init__(self):
            super().__init__()
            self.setWindowTitle("OCR字符识别")
            self.setGeometry(100, 100, 800, 630)
            self.statusBar().showMessage(self.statusbar_msg)

            self.camera = Qt5Camera(0, CAMERA_RESOLUTION, isRGB=False)
            self.camera.dataUpdated.connect(self._update_frame)
            # vbox = QVBoxLayout()
            # self.setLayout(vbox)
            self.canvas = QLabel("img_frame", self)
            self.canvas.setFixedSize(800,600)
            self.canvas.setScaledContents(True)

            self.ocr_engine = Tesseract(TessEnv["TessDataDir"],
                                        TessEnv["Lang"])
            self.set_roi([250, 200, 300, 100])
            self.camera.start()

        def closeEvent(self, event):
            super().closeEvent(event)
            self.camera.stop()

        def set_roi(self, roi):
            self.ROI = roi

        def set_roi2(self, point_1, point_2):
            x, y = point_1
            x2, y2 = point_2
            self.ROI = [x, y, x2-x, y2-y]

        def _update_frame(self, im_arr):
            im_roi = cv.crop(im_arr, self.ROI)
            cv.draw_rect(im_arr, *self.ROI, thickness=2)
            pixmap = cv.ndarray2pixmap(im_arr)
            self.canvas.setPixmap(pixmap)
            self._recognize(im_roi)

        def _recognize(self, im_arr):
            result = self.ocr_engine.ndarray2text(im_arr)
            # logger.debug(">>> OCR: {}".format(result))
            msg = f'OCR识别结果: {result}' if result else self.statusbar_msg
            self.statusBar().showMessage(msg)
            return result


    class MainWnd(OcrEngineMixin, BaseCvWnd):
        statusbar_msg = '请移动画面，将字符置于识别框中'

        def __init__(self, parent, camera_idx=0, solution=None, isRGB=False):
            super().__init__(parent, camera_idx, solution, isRGB)

            self.setWindowTitle("OCR字符识别")
            self.set_roi([250, 200, 300, 100])

            self.ocr_init(TessEnv["TessDataDir"], TessEnv["Lang"])

        def update_script(self):
            """ 由于关乎可变脚本script，故需要在子类重写 """
            reload(script)
            self.status_bar.showMessage("预处理脚本已更新")

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
                cv.draw_rect(im_left, *self.ROI, thickness=2)
                func = self.improc_methods.get("parse_roi")
                if func:
                    im_text = func(cv.crop(im_left, self.ROI), *list_params)
                    self.ocr_exec(im_text)

            if self.isSwitched:
                im_left, im_right = im_right, im_left

            im_resized = cv.rescale(im_right, 0.5)
            self.canvas.set_image(im_left)
            self.processing.set_image(im_resized)


    # run_qtapp(SimpleOCR)
    run_qtapp(MainWnd, None, solution=[800,600])
