# [OpenCV_sample: dnn/text_detection.py](https://github.com/opencv/opencv/blob/master/samples/dnn/text_detection.py)

# Usage: python3 opencv_ocr.py --input images/lebron_james.jpg --model weights/frozen_east_text_detection.pb --ocr weights/crnn.onnx

'''
Text detection model: https://github.com/argman/EAST
Download link: https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.tar.gz?dl=1
CRNN Text recognition model taken from here: https://github.com/meijieru/crnn.pytorch
How to convert from pb to onnx:
Using classes from here: https://github.com/meijieru/crnn.pytorch/blob/master/models/crnn.py
More converted onnx text recognition models can be downloaded directly here:
Download link: https://drive.google.com/drive/folders/1cTbQ3nuZG-EKWak6emD_s8_hHXWz7lAr?usp=sharing
And these models taken from here:https://github.com/clovaai/deep-text-recognition-benchmark

import torch
from models.crnn import CRNN
model = CRNN(32, 1, 37, 256)
model.load_state_dict(torch.load('crnn.pth'))
dummy_input = torch.randn(1, 1, 32, 100)
torch.onnx.export(model, dummy_input, "crnn.onnx", verbose=True)
'''

# Import required modules
import numpy as np
import cv2
import math

# Read and store arguments
confThreshold = 0.5
nmsThreshold = 0.4
inpWidth = 320
inpHeight = 320
modelDetector = "weights/frozen_east_text_detection.pb"
modelRecognition = "weights/crnn.onnx"  # if just EAST, set None

camera_as_input = True
path_img = "/xxx.jpg"

############ Utility functions ############

def fourPointsTransform(frame, vertices):
    vertices = np.asarray(vertices)
    outputSize = (100, 32)
    targetVertices = np.array([
        [0, outputSize[1] - 1],
        [0, 0],
        [outputSize[0] - 1, 0],
        [outputSize[0] - 1, outputSize[1] - 1]], dtype="float32")

    rotationMatrix = cv2.getPerspectiveTransform(vertices, targetVertices)
    result = cv2.warpPerspective(frame, rotationMatrix, outputSize)
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


if __name__ == "__main__":
    # Load network
    detector = cv2.dnn.readNet(modelDetector)
    # 使用GPU加速
    detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    if modelRecognition:
        recognizer = cv2.dnn.readNet(modelRecognition)
        # 使用GPU加速
        recognizer.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        recognizer.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    # Create a new named window
    kWinName = "EAST: An Efficient and Accurate Scene Text Detector"
    run_in_terminal = False
    try:
        cv2.namedWindow(kWinName, cv2.WINDOW_NORMAL)
    except cv2.error:
        run_in_terminal = True

    outNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"
    ]

    tickmeter = cv2.TickMeter()

    def ocr_system(frame):
        # Get frame height and width
        height_ = frame.shape[0]
        width_ = frame.shape[1]
        rW = width_ / float(inpWidth)
        rH = height_ / float(inpHeight)

        # Create a 4D blob from frame.
        blob = cv2.dnn.blobFromImage(frame, 1.0, (inpWidth, inpHeight), (123.68, 116.78, 103.94), True, False)

        # Run the detection model
        detector.setInput(blob)

        tickmeter.start()
        outs = detector.forward(outNames)
        tickmeter.stop()

        # Get scores and geometry
        scores = outs[0]
        geometry = outs[1]
        [boxes, confidences] = decodeBoundingBoxes(scores, geometry, confThreshold)

        # Apply NMS
        indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, confThreshold, nmsThreshold)
        if run_in_terminal:
            results = []
        for i in indices:
            # get 4 corners of the rotated rect
            vertices = cv2.boxPoints(boxes[i[0]])
            # scale the bounding box coordinates based on the respective ratios
            for j in range(4):
                vertices[j][0] *= rW
                vertices[j][1] *= rH

            if not run_in_terminal:
                for j in range(4):
                    p1 = (vertices[j][0], vertices[j][1])
                    p2 = (vertices[(j + 1) % 4][0], vertices[(j + 1) % 4][1])
                    cv2.line(frame, p1, p2, (0, 255, 0), 1)

            # get cropped image using perspective transform
            if modelRecognition:
                cropped = fourPointsTransform(frame, vertices)
                cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

                # Create a 4D blob from cropped image
                blob = cv2.dnn.blobFromImage(cropped, size=(100, 32), mean=127.5, scalefactor=1 / 127.5)
                recognizer.setInput(blob)

                # Run the recognition model
                tickmeter.start()
                result = recognizer.forward()
                tickmeter.stop()

                # decode the result into text
                wordRecognized = decodeText(result)
                if run_in_terminal:
                    results.append(wordRecognized)
                else:
                    cv2.putText(frame, wordRecognized, (int(vertices[1][0]), int(vertices[1][1])),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
        tickmeter.reset()

        # Put efficiency information
        label = 'Inference time: %.2f ms' % (tickmeter.getTimeMilli())
        if not run_in_terminal:
            cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

        # Display the frame
        if run_in_terminal:
            print("识别结果：\n", results, "\n", label)
        else:
            cv2.imshow(kWinName, frame)

    if camera_as_input:
        cap = cv2.VideoCapture(0)
        while cv2.waitKey(1) < 0:
            # Read frame
            hasFrame, frame = cap.read()
            if not hasFrame:
                cv2.waitKey()
                break
            ocr_system(frame)
    else:
        frame = cv2.imread(camera_as_input)
        ocr_system(frame)
