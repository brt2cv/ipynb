# http://codec.wang/#/opencv/
# https://github.com/CodecWang/Blog/tree/master/docs/opencv
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html

# %%
import util as uu
uu.enpy("opencv")

import ocv as cv
uu.reload(cv)

#####################################################################
# %% 查找直线与圆
path_shapes = "tutorial/src/shapes.jpg"
im_shapes = cv.imread(path_shapes)
uu.imshow(im_shapes)

# %% find_lines or circles
im_edges = cv.canny(im_shapes, [50,100])
uu.imshow(im_edges)
# %%
lines = cv.find_lines(im_edges, 0.8, 90, min_length=50)
im_dr = np.zeros(im_shapes.shape)
for line in lines:
    cv.draw_line(im_dr, line.a, line.b, 255, 1)
    print(cv.distance(*line))
uu.imshow(im_dr)

# %%
circles = cv.find_circles(im_edges, im_edges.shape[1]*0.5, 30)
im_dr = np.zeros(im_shapes.shape)
for cc in circles:
    cv.draw_circle(im_dr, cc.cx, cc.cy, cc.r, 255, 1)
uu.imshow(im_dr)

#####################################################################
# %% 凸包及更多轮廓特征
path_convex = "tutorial/src/convex.jpg"
im_convex = cv.imread(path_convex)
uu.imshow(im_convex)

# %%
blobs = cv.find_blobs(im_convex, 111)
assert len(blobs) == 1
blob = blobs[0]
hull = cv.approx_convex(blob._cnt)
# cv.draw_polygon(im_convex, hull, 255)
# uu.imshow(im_convex)

# %%
defects = cv.convex_defects(blob._cnt)
im_dr = im_convex.copy()
# cv.draw_points(im_dr, defects, (255,0,0), 5)
for p in defects:
    cv.draw_circle(im_dr, *p, 5, (255,0,0), 5)
uu.imshow(im_dr)
