# %%
import util as uu
uu.enpy("opencv")

import numpy as np
import cv2
import ocv as cv
uu.chdir(__file__)


# %%
path = uu.rpath("img/usb_cam_1.jpg")
im = cv.imread(path)
uu.imshow(im)

# %%
roi = cv.crop2(im, [80, 180], [1850, 900])
uu.imshow(roi)

# %%
im_fil = cv.gaussian(roi, 3)
im_bin = cv.threshold(im_fil, 200, invert=1)
im_med = cv.median(im_bin, 5)
uu.imshow(im_med, 1)

# %%
im_dr = roi.copy()

list_blobs = cv.find_blobs(im_med)
list_cc = []
for b in list_blobs:
    if b.area() < 99:
        continue

    pnts, results = cv.bounding_rect(b.cnt, 1)
    h, w = results[1]
    area_ratio = round(b.area() / (h*w) / std_area_ratio, 2)

    roundness = round(b.elongation(), 2)
    x, y = b.center()
    cv.draw_string(im_dr, x+5, y, f"\{roundness}", color=0)

    if roundness < 0.9:
        continue
    else:


    # cv.draw_string(im_dr, x+5, y+22, f"/{area_ratio}", color=0)
for cc in list_cc:



    cv.draw_polygon(im_dr, pnts, color=0)
uu.imshow(im_dr, 1)

# %%
list_items = cv.find_circles(im, 100, 40, 80)
# im_dr = np.zeros(im_bin.shape)
im_dr = im.copy()
for cc in list_items:
    if cc.r < 10:
        continue
    cv.draw_circle(im_dr, cc.cx, cc.cy, cc.r+5, 0)

uu.imshow(im_dr, 1)

# %%
im_fil =
