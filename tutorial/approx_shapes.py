# %%
import util as uu
uu.enpy("opencv")
import ocv as cv

# %%
uu.reload(cv)

#####################################################################
# %% 检测圆
path_circles = "tutorial/img/circles.jpg"
im = cv.imread(path_circles)

# %%
im = cv.invert(im)

# %% 通过 blob.roundness() 粗略推断圆度
im_dr = im.copy()
for b in cv.find_blobs(im, 20):
    if b.area() < 99: continue
    rn = round(b.roundness(),3)
    # print(">>> roundness:", rn)
    cv.draw_string(im_dr, b.cx + 10, b.cy, f"{rn}", 0.5)
uu.imshow(im_dr)

# %% 通过 HoughCircle 查找圆
im_dr = cv.gray2rgb(im)
list_cc = cv.find_circles(im, 100, 20, 300)
for cc in list_cc:
    cv.draw_circle(im_dr, cc.cx, cc.cy, cc.r, (255,0,0))
uu.imshow(im_dr)

# %% 通过approx_ellipse拟合圆
im_one = cv.crop(im, (114,100,100,100))
im_bin = cv.threshold_otsu(im_one)
cnts = cv.find_cnts(im_bin)
assert len(cnts) == 1
elp = cv.approx_ellipse(cnts[0])
im_dr = cv.gray2rgb(im_one)
cv.draw_ellipse(im_dr, *elp, color=(255,0,0))
uu.imshow(im_dr)
