#!/usr/bin/env python3
# @Date    : 2020-11-27
# @Author  : Bright Li (brt2@qq.com)
# @Link    : https://gitee.com/brt2
# @Version : 0.2.2

try:
    from utils.log import getLogger
    print("[+] {}: 启动调试Logger".format(__file__))
    logger = getLogger(1)
except ImportError:
    from logging import getLogger
    print("[!] {}: 调用系统logging模块".format(__file__))
    logger = getLogger(__file__)

TessEnv = {
    # "TesseractBinPath": "D:/programs/Tesseract",
    "TessDataDir": "/home/brt/ws/tmv/src/plugins/tesseract/tessdata/demo",
    # "TessDataDir": "D:/Home/workspace/tmv/src/plugins/tesseract/tessdata/demo",
    "Lang": "eng"
}

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
        print("..", text)
        text = text.strip("\n")
        # text = text.strip()
        if not text.strip():
            return ""
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

if __name__ == "__main__":
    from PyQt5.QtWidgets import QMainWindow, QLabel

    from camera import Qt5Camera
    import opencv as cv

    class MainWnd_OCR(QMainWindow):
        def __init__(self, resolution):
            super().__init__()
            self.setWindowTitle("HeroJe - OCR字符识别")
            self.setGeometry(100, 100, 800, 630)
            self.statusBar().showMessage('请移动画面，将字符置于识别框中')

            self.camera = Qt5Camera(0, resolution, isRGB=False)
            self.camera.dataUpdated.connect(self._update_frame)
            # vbox = QVBoxLayout()
            # self.setLayout(vbox)
            self.canvas = QLabel("img_frame", self)
            self.canvas.setFixedSize(800,600)
            self.canvas.setScaledContents(True)

            self.ocr_engine = Tesseract(TessEnv["TessDataDir"],
                                        TessEnv["Lang"])

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
            logger.debug(">>> OCR: {}".format(result))
            msg = 'OCR识别结果: {}'.format(result.strip())
            self.statusBar().showMessage(msg)
            return result


    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    w = MainWnd_OCR([800, 600])
    w.set_roi([250, 200, 300, 100])
    w.show()

    sys.exit(app.exec_())
