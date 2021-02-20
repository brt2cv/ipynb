# 依赖opencv-contrib-python包
# 否则无法加载cv2.text, cv2.dnn等模块

# %%
import ipyenv as uu
uu.enpy("img")

import numpy as np
import cv2
import pycv.opencv as cv
uu.chdir(__file__)

# %% SWT文本检测
path_img, isDark = uu.rpath("images/line.jpg"), 1  # 额，最简单的数字，检测不准确
# path_img, isDark = uu.rpath("images/nature-french.jpg"), 0  # 自然场景的法语倒识别可以
# path_img, isDark = uu.rpath("images/page-zh.jpg"), 1  # 中文的轮廓不准确
# path_img, isDark = uu.rpath("images/book-zh.jpeg"), 1  # 但可以通过重叠度判断区域
# path_img, isDark = uu.rpath("images/nature-zh.jpg"), 1  # 挂了。。。
im_detect_text = cv.imread(path_img, 0)  # SWT要求len_shape=3
# uu.imshow(im_detect_text)

results, draw, chainBBs = cv2.text.detectTextSWT(im_detect_text,
                                                 dark_on_light=isDark)
print(chainBBs)
# uu.imshow(draw)

im_dr = im_detect_text.copy()
for rect in results:
    cv2.rectangle(im_dr, rect, (255,0,0), 2)
uu.imshow(im_dr, 1)

#####################################################################
# %% OpenCV_EAST检测#####################################################################
# [Tutorial](https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/)
# [github: EAST-Detector-for-text-detection-using-OpenCV](https://github.com/ZER-0-NE/EAST-Detector-for-text-detection-using-OpenCV)
# [gitee_fork](https://gitee.com/information235/EAST-Detector-for-text-detection-using-OpenCV)

path_img = uu.rpath("EAST-Detector-for-text-detection-using-OpenCV/images/lebron_james.jpg")

image = cv.imread(path_img)
_H, _W = image.shape[:2]

H, W = 320, 320  # 调整后的图像尺寸，EAST文本要求输入图像尺寸必须是32的倍数
image = cv2.resize(image, (W, H))
uu.imshow(image)

# %%
# 为了使用OpenCV和EAST深度学习模型进行文本检测，
# 我们需要定义两个输出层，得到两个输出特征图：
layerNames = [
    "feature_fusion/Conv_7/Sigmoid",  # 第一层是sigmoid激活层，它给出一个区域包含文本的概率
    "feature_fusion/concat_3"  # 第二层是表示图像“几何”的输出要素图，使用它来导出输入图像中的文本的边界框坐标
]

# 加载OpenCV的EAST文本检测器
path_east_model = uu.rpath("EAST-Detector-for-text-detection-using-OpenCV/frozen_east_text_detection.pb")
net = cv2.dnn.readNet(path_east_model)

# 将图像转换为blob
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                             (123.68, 116.78, 103.94), swapRB=True, crop=False)

import time
start = time.time()
"""
要预测文本，我们可以简单地将blob设置为输入并调用 `net.forward`。通过向net.forward提供输出层的名称，告诉OpenCV返回我们想要的两个特征映射：
+ 几何图（geometry map），用于导出输入图像中文本边界框坐标的
+ 分数图（scores map），包含给定区域包含文本的概率
"""
net.setInput(blob)
scores, geometry = net.forward(layerNames)
end = time.time()
print("[INFO] text detection took {:.6f} seconds".format(end - start))

# %%
# 接下来要逐个遍历这些输出值，对其进行解码得到文本框的位置信息（包括方向）：
min_confidence = 0.5

numRows, numCols = scores.shape[2:4]
rects = []
confidences = []

# loop over the number of rows
for y in range(0, numRows):
    # extract the scores (probabilities), followed by the geometrical
    # data used to derive potential bounding box coordinates that
    # surround text
    scoresData = scores[0, 0, y]
    xData0 = geometry[0, 0, y]
    xData1 = geometry[0, 1, y]
    xData2 = geometry[0, 2, y]
    xData3 = geometry[0, 3, y]
    anglesData = geometry[0, 4, y]

    # loop over the number of columns
    for x in range(0, numCols):
        # if our score does not have sufficient probability, ignore it
        if scoresData[x] < min_confidence:
            continue

        # compute the offset factor as our resulting feature maps will
        # be 4x smaller than the input image
        (offsetX, offsetY) = (x * 4.0, y * 4.0)

        # extract the rotation angle for the prediction and then
        # compute the sin and cosine
        angle = anglesData[x]
        cos = np.cos(angle)
        sin = np.sin(angle)

        # use the geometry volume to derive the width and height of
        # the bounding box
        h = xData0[x] + xData2[x]
        w = xData1[x] + xData3[x]

        # compute both the starting and ending (x, y)-coordinates for
        # the text prediction bounding box
        endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
        endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
        startX = int(endX - w)
        startY = int(endY - h)

        # add the bounding box coordinates and probability score to
        # our respective lists
        rects.append((startX, startY, endX, endY))
        confidences.append(scoresData[x])

# %%
# 以下依赖：imutils，执行极大值抑制
from imutils.object_detection import non_max_suppression

boxes = non_max_suppression(np.array(rects), probs=confidences)
print("boxes >>> ", boxes)

# loop over the bounding boxes
im_dr = image.copy()
for (startX, startY, endX, endY) in boxes:
    # scale the bounding box coordinates based on the respective
    # ratios
    startX = int(startX * rW)
    startY = int(startY * rH)
    endX = int(endX * rW)
    endY = int(endY * rH)
    # draw the bounding box on the image
    cv2.rectangle(im_dr, (startX, startY), (endX, endY), (0, 255, 0), 2)

uu.imshow(im_dr)

#####################################################################
# %%
# [OpenCV_sample: dnn/text_detection.py](https://github.com/opencv/opencv/blob/master/samples/dnn/text_detection.py)
# 使用官网的EAST脚本，检测图像
# 具体见同目录下的脚本: east_text_detection_by_opencv.py
