#!/usr/bin/env python3
# @Date    : 2021-02-24
# @Author  : Bright Li (brt2@qq.com)
# @Link    : https://gitee.com/brt2
# @Version : 0.2.1

# 使用OpenCV加载pytorch的ONNX模型，实现EAST检测与CRNN的识别

import math
import numpy as np
import cv2

def fourPointsTransform(im_arr, vertices):
    vertices = np.asarray(vertices)
    outputSize = (100, 32)
    targetVertices = np.array([
        [0, outputSize[1] - 1],
        [0, 0],
        [outputSize[0] - 1, 0],
        [outputSize[0] - 1, outputSize[1] - 1]], dtype="float32")

    rotationMatrix = cv2.getPerspectiveTransform(vertices, targetVertices)
    result = cv2.warpPerspective(im_arr, rotationMatrix, outputSize)
    return result

def decodeText(scores):
    text = ""
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    for i in range(scores.shape[0]):
        c = np.argmax(scores[i][0])
        if c != 0:
            text += alphabet[c - 1]
        else:
            text += '-'

    # adjacent same letters as well as background text must be removed to get the final output
    char_list = []
    for i in range(len(text)):
        if text[i] != '-' and (not (i > 0 and text[i] == text[i - 1])):
            char_list.append(text[i])
    return ''.join(char_list)

def decodeBoundingBoxes(scores, geometry, scoreThresh):
    detections = []
    confidences = []
    ############ CHECK DIMENSIONS AND SHAPES OF geometry AND scores ############
    assert len(scores.shape) == 4, "Incorrect dimensions of scores"
    assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
    assert scores.shape[0] == 1, "Invalid dimensions of scores"
    assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
    assert scores.shape[1] == 1, "Invalid dimensions of scores"
    assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
    assert scores.shape[2] == geometry.shape[2], "Invalid dimensions of scores and geometry"
    assert scores.shape[3] == geometry.shape[3], "Invalid dimensions of scores and geometry"
    height = scores.shape[2]
    width = scores.shape[3]
    for y in range(0, height):
        # Extract data from scores
        scoresData = scores[0][0][y]
        x0_data = geometry[0][0][y]
        x1_data = geometry[0][1][y]
        x2_data = geometry[0][2][y]
        x3_data = geometry[0][3][y]
        anglesData = geometry[0][4][y]
        for x in range(0, width):
            score = scoresData[x]

            # If score is lower than threshold score, move to next x
            if (score < scoreThresh):
                continue

            # Calculate offset
            offsetX = x * 4.0
            offsetY = y * 4.0
            angle = anglesData[x]

            # Calculate cos and sin of angle
            cosA = math.cos(angle)
            sinA = math.sin(angle)
            h = x0_data[x] + x2_data[x]
            w = x1_data[x] + x3_data[x]

            # Calculate offset
            offset = ([offsetX + cosA * x1_data[x] + sinA * x2_data[x], offsetY - sinA * x1_data[x] + cosA * x2_data[x]])

            # Find points for rectangle
            p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
            p3 = (-cosA * w + offset[0], sinA * w + offset[1])
            center = (0.5 * (p1[0] + p3[0]), 0.5 * (p1[1] + p3[1]))
            detections.append((center, (w, h), -1 * angle * 180.0 / math.pi))
            confidences.append(float(score))

    # Return detections and confidences
    return [detections, confidences]

class OpencvDnnMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def ocr_init(self, det_model, rec_model, resize_to=(320,320), conf_thresh=0.5, nms_thresh=0.4):
        # Load network
        self.detector = cv2.dnn.readNet(det_model)
        # 使用GPU加速
        self.detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        self.recognizer = cv2.dnn.readNet(rec_model)
        # 使用GPU加速
        self.recognizer.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.recognizer.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        self.outNames = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"
        ]
        self.input_size = resize_to
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

    def _detect_text(self, im_arr):
        blob = cv2.dnn.blobFromImage(im_arr, 1.0, self.input_size, (123.68, 116.78, 103.94), True, False)
        self.detector.setInput(blob)
        scores, geometry = self.detector.forward(self.outNames)

        # Get scores and geometry
        boxes, confidences = decodeBoundingBoxes(scores, geometry, self.conf_thresh)

        # Apply NMS
        indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, self.conf_thresh, self.nms_thresh)
        return [boxes[x[0]] for x in indices]  # nms_boxes

    def _recongnize_text(self, im_gray):
        # Create a 4D blob from cropped image
        blob = cv2.dnn.blobFromImage(im_gray, size=(100, 32), mean=127.5, scalefactor=1 / 127.5)
        self.recognizer.setInput(blob)

        # Run the recognition model
        result = self.recognizer.forward()

        # decode the result into text
        wordRecognized = decodeText(result)
        return wordRecognized

    def ocr_exec(self, im_arr):
        results = []
        h, w = im_arr.shape[:2]
        rW = w / self.input_size[0]
        rH = h / self.input_size[1]

        nms_boxes = self._detect_text(im_arr)
        for box in nms_boxes:
            # get 4 corners of the rotated rect
            vertices = cv2.boxPoints(box)
            # scale the bounding box coordinates based on the respective ratios
            for j in range(4):
                vertices[j][0] *= rW
                vertices[j][1] *= rH

            # get cropped image using perspective transform
            cropped = fourPointsTransform(im_arr, vertices)
            im_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

            text = self._recongnize_text(im_gray)
            results.append(text)

        return results

    def ocr_post(self, result):
        print(">> CRNN result:", result)


if __name__ == "__main__":
    from importlib import reload
    import traceback

    from pyqt.qtwx import *
    from pyqt.qtcv import *
    from pycv.camera import Qt5Camera

    import script

    OpencvDnnModels = {
        "Detector": "/home/brt/Desktop/workspace/ipynb/Tutorial/OCR/weights/frozen_east_text_detection.pb",
        "Recognizer": "/home/brt/Desktop/workspace/ipynb/Tutorial/OCR/weights/crnn.onnx"
    }

    class MainWnd(OpencvDnnMixin, BaseCvWnd):
        statusbar_msg = '使用OpenCV进行OCR检测识别'

        def __init__(self, parent, camera_idx=0, solution=None, fps=0, isRGB=True):
            super().__init__(parent, camera_idx, solution, fps, isRGB)
            loadUi("ui/wx_mwnd_with_ctrllers.ui", self)
            super()._setup_ui()

            self.setWindowTitle("OCR字符识别")
            # self.set_roi([250, 200, 300, 100])

            self.ocr_init(OpencvDnnModels["Detector"],
                          OpencvDnnModels["Recognizer"])

        def update_script(self):
            """ 由于关乎可变脚本script，故需要在子类重写 """
            reload(script)
            self.define_improc()
            self.status_bar.showMessage("预处理脚本已更新")

        def define_improc(self):
            """ 由于关乎可变脚本script，故需要在子类重写 """
            self.improc_methods = {
                "window1": script.improc_origin,
                "window2": script.improc_roi,  # make_right
                "parse_ocr": script.improc_roi,
            }

        def ocr_post(self, result):
            pass

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

            # OCR数据处理
            func = self.improc_methods.get("parse_ocr")
            if func:
                im_text = func(im_left, *list_params)
                text = self.ocr_exec(im_text)
                msg = f'OCR识别结果: {text}' if text else self.statusbar_msg
                print(msg)
                self.status_bar.showMessage(msg)

            im_left = self.improc_methods["window1"](im_left, *list_params)

            if self.isSwitched:
                im_left, im_right = im_right, im_left

            im_resized = cv.rescale(im_right, 0.5)
            self.canvas.set_image(im_left)
            self.processing.set_image(im_resized)

    run_qtapp(MainWnd, None, camera_idx=0, fps=1, solution=[800,600], isRGB=True)
