
# %%
import ipyenv as uu
uu.enpy("img")
uu.chdir(__file__)

import numpy as np
import cv2
import pycv.opencv as cv

# %%
path_img = uu.rpath("img/曝光不足.jpg")
im = cv.imread(path_img)
uu.imshow(im)

# %% 普通的gamma矫正
def adjust_gamma(img, gamma=1.0):
    table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]  # 建立映射表
    table = np.array(table).astype("uint8")
    return cv2.LUT(img, table)

im_gamma = adjust_gamma(im, 2.0)
uu.imshow(im_gamma)

# %% 自动gamma的确定（不太理想）
import math
gamma = math.log10(0.5) / math.log10(np.mean(im)/255)    # 公式计算gamma
print(">> gamma的取值:", gamma)  # 0.7325
im_gamma = adjust_gamma(im, gamma)
uu.imshow(im_gamma)

# %% 基于二维伽马函数的光照不均匀图像自适应校正算法
HSV = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
H,S,V = cv2.split(HSV)

# %%
k = min(V.shape)
if k % 2 == 0:
    k -= 1
kernel = (k, k)

# %%
V_float = V.astype("float")
SIGMA1 = 15
SIGMA2 = 80
SIGMA3 = 250
q = math.sqrt(2)
F1 = cv2.GaussianBlur(V_float, kernel, SIGMA1 / q)
F2 = cv2.GaussianBlur(V_float, kernel, SIGMA2 / q)
F3 = cv2.GaussianBlur(V_float, kernel, SIGMA3 / q)
F = (F1 + F2 + F3) / 3

# %%
h, w = F.shape
average = np.mean(F)
out = np.zeros((h, w), "uint8")
for i in range(h):
    for j in range(w):
        y = (average - F[i][j]) / average
        gamma = np.power(0.5, y)
        out[i][j] = np.power(V[i][j] / 255, gamma) * 255
# uu.imshow(out)

# %%
im_merge = cv2.merge([H,S,out])
im_rgb = cv2.cvtColor(im_merge, cv2.COLOR_HSV2BGR)
# im_rgb = cv2.cvtColor(im_merge, cv2.COLOR_HSV2RGB)
uu.imshow(im_rgb)
