# %%
# 富士康微小零件的视觉检测
import util as uu
uu.enpy("opencv")
import ocv as cv

# %%
uu.reload(cv)

#####################################################################
# %%
import os
img_dir = "check_富士康_j307/"
path = lambda fn: os.path.join(img_dir, f"usb_cam_{fn}.jpg")

# %% 胶未打饱
im_glue_ng = cv.imread(path("11"))  # 2,
uu.imshow(im_glue_ng)

# %% 横梁毛边
im_cross_rail_ng = cv.imread(path("17"))
uu.imshow(im_cross_rail_ng)

# %% 胶口高（杂质）
im_ = cv.imread(path("9"))  # 7,

# %% 头部溢胶
im_ = cv.imread(path("12"))

# %% 多料
im_ = cv.imread(path("0"))

# %% 可修毛边
im_ = cv.imread(path("55"))

# %% 定位柱变形
im_ = cv.imread(path("59"))

# %% 脏污
im_ = cv.imread(path("62"))
