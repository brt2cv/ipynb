# %%
import ipyenv as uu
uu.chdir(__file__)
uu.enpy("opencv")

import pycv.opencv as cv
import numpy as np

# %%
uu.reload(cv)

# %%
path_tpl = uu.rpath("image/cap_1.jpg")
im_tpl = cv.imread(path_tpl)
uu.imshow(im_tpl)

# # %% 增强对比度
# tpl_con = cv.contrast(im_tpl, 14, 90)
# uu.imshow(tpl_con)

#####################################################################
# 方法1，利用标签边界
#####################################################################

# %% ~~ 去除边界
h, w = im_tpl.shape[:2]
margin = [w//12, h//5]
tpl_no_margin = cv.crop2(im_tpl, (margin[0],margin[1]), (w-margin[0],h-margin[1]))
uu.imshow(tpl_no_margin)


# %%
""" 尝试使用角点检测
x = cv.find_corners(im, 0.8, 1)
print(x)
center = x[0]
im_dr = im.copy()
cv.draw_circle(im_dr, *center, 20)
uu.imshow(im_dr)
"""

def pos_corner(im, thresh, margin_ratio=0.1):
    im_bin = cv.binary(im, thresh)
    im_proc = cv.median(im_bin, 5)
    # im_open = cv.opening(im_bin, (9,9))
    # im_gau = cv.gaussian(im_open, 3)

    list_blobs = cv.find_cnts(im_proc)
    # assert len(list_blobs) == 1
    if len(list_blobs) == 1:
        polygon = list_blobs[0]
    else:
        max_area = 0
        for b in list_blobs:
            b_area = cv.cnt_area(b)
            if b_area > max_area:
                max_area = b_area
                polygon = b

    list_pnts = cv.approx_polygon(polygon, epsilon=11)
    w, h = im.shape[:2]
    w_margin = w * margin_ratio
    h_margin = h * margin_ratio
    w_range = (w_margin, w-w_margin)
    h_range = (h_margin, h-h_margin)

    corners = []
    for p in list_pnts:
        if w_range[0] <= p[0] <= w_range[1] and h_range[0] <= p[1] <= h_range[1]:
            corners.append(p)

    nCor = len(corners)
    if nCor == 1:
        return corners[0]
    elif nCor > 1:
        x = round(sum([p[0] for p in corners]) / nCor)
        y = round(sum([p[1] for p in corners]) / nCor)
        return [int(x),int(y)]
    else:
        print(">>>", list_pnts)
        uu.imshow(im_proc)

cor_len = 160
h, w = tpl_no_margin.shape[:2]
list_corners = []
for x,y in [(0,0), (w-cor_len, 0), (w-cor_len, h-cor_len), (0, h-cor_len)]:
    im_corner = cv.crop(tpl_no_margin, (x,y,cor_len,cor_len))
    relpos = pos_corner(im_corner, thresh=111)
    abspos = (x+relpos[0], y+relpos[1])
    list_corners.append(abspos)

print(list_corners)

# %% 透视变换
tpl_psp = cv.perspect2rect(tpl_no_margin, list_corners)
uu.imshow(tpl_psp, 1)

# %% 滤波，二值化，多边形拟合
tpl_bin = cv.binary(tpl_psp, 111, 1)
tpl_med = cv.median(tpl_bin, 3)
uu.imshow(tpl_med)

#####################################################################
# %% 比对待检测图片
#####################################################################
path_check = uu.rpath("image/cap_2.jpg")
im_check = cv.imread(path_check)
uu.imshow(im_check)

# %%
cor_len = 160
h, w = check_no_margin.shape[:2]
list_corners2 = []
for x,y in [(0,0), (w-cor_len, 0), (w-cor_len, h-cor_len), (0, h-cor_len)]:
    im_corner = cv.crop(check_no_margin, (x,y,cor_len,cor_len))
    relpos = pos_corner(im_corner, thresh=111)
    if relpos is None:
        continue
    abspos = (x+relpos[0], y+relpos[1])
    list_corners2.append(abspos)

print(list_corners2)

# %% 透视为模板的尺寸
check_psp = cv.perspect2rect(check_no_margin, list_corners2, (2472, 1884))
check_bin = cv.binary(check_psp, 111, 1)
check_med = cv.median(check_bin, 3)
uu.imshow(check_med)

# %% 比对： diff
im_diff = check_med - tpl_med
uu.imshow(im_diff, 1)





# %%
#####################################################################
# 方法2: 由于文字区域相对于标签会有变化，所以，改由文字区域的关键点位来进行定位
#####################################################################

# 左下角区域图像
h, w = im_tpl.shape[:2]

left_bottom = cv.crop(im_tpl, (0, h//2, w//2, h//2))
uu.imshow(left_bottom)

# %% ~~ 通过模板的方式查找位置
logo_size = (350, 360)
tpl_logo = cv.crop(left_bottom, (260,210,*logo_size))
uu.imshow(tpl_logo)

# %%
pos, similarity = cv.find_template(im_tpl, tpl_logo)
assert similarity > 0.67, f"模板匹配错误，相似度太低【{similarity}】"
print(pos)

# %% 获取到logo的图像区域
MarginWidth = 30
im_logo = cv.crop_margins(im_tpl, [*pos, *tpl_logo.shape[:2]], MarginWidth)
uu.imshow(im_logo)

# %%
logo_bin = cv.threshold_otsu(im_logo, 1)
logo_bin = cv.median(logo_bin, 3)
uu.imshow(logo_bin)

# %% 投影
logo_x_proj = cv.project(logo_bin, 0)
logo_y_proj = cv.project(logo_bin, 1)

def first_index(arr_1d):
    if isinstance(arr_1d, list):
        arr = arr_1d
    else:
        arr = list(arr_1d)
    return arr.index(True)

def last_index(arr_id):
    return len(arr_id) - first_index(reversed(arr_id))

corner_left_bottom = [first_index(logo_x_proj), last_index(logo_y_proj)]
print(corner_left_bottom)

abspos_left_bottom = (pos[0] - MarginWidth + corner_left_bottom[0],
                      pos[1] - MarginWidth + corner_left_bottom[1])
print("[+] 获取到左下角的绝对坐标:", abspos_left_bottom)

# %%
#####################################################################
# 右下角点
#####################################################################

h, w = im_tpl.shape[:2]

right_bottom = cv.crop(im_tpl, (w//2, h//2, w//2, h//2))
uu.imshow(right_bottom)

# %% 获取条形码区域
im_next = cv.binary(right_bottom, 80, 1)
im_next = cv.closing(im_next, (29, 1))
# im_next = cv.median(im_next, 3)
uu.imshow(im_next)

list_cnts = cv.find_blobs(im_next)
im_dr = cv.new(im_next.shape)
corner_right_bottom = [0, 0]
for b in list_cnts:
    if b.area() < 40000:
        continue
    x, y, w_, h_ = b.bounding()
    if w_ < h_:
        continue
    cv.draw_rect(im_dr, x, y, w_, h_)
    if y > corner_right_bottom[1]:
        corner_right_bottom = [x+w_, y+h_]

uu.imshow(im_dr)

# %% 获取corner的绝对坐标
abspos_right_bottom = [w//2 + corner_right_bottom[0], h//2 + corner_right_bottom[1]]
print("[+] 获取到右下角的绝对坐标:", abspos_right_bottom)

# %%
#####################################################################
# 右上角
#####################################################################

right_top = cv.crop(im_tpl, (w//2, 0, w//2, h//2))
uu.imshow(right_top)

# %%
im_next = cv.binary(right_top, 80, 1)
im_next = cv.median(im_next, 3)
im_next = cv.dilate(im_next, (29, 1))
uu.imshow(im_next)

# %% ~~ 获取最上方一行文字的坐标 ~~
x_arr = cv.project(im_next, 0)
whitespacas_num = 0
for i, v in enumerate(x_arr):
    if v:
        whitespacas_num = 0
        continue
    whitespacas_num += 1
    if whitespacas_num > 10:
        print(">>> 文字的x坐标至于：", i - 10)
        break

# %% 通过blob获取文字区域
list_cnts = cv.find_blobs(im_next)
im_dr = cv.new(im_next.shape)
corner_right_top = [0, im_next.shape[1]]

comp = lambda list_size: list_size[0] - list_size[1]

for b in list_cnts:
    if b.area() < 10000:
        continue
    x, y, w_, h_ = b.bounding()
    if w_ < h_:
        continue
    cv.draw_rect(im_dr, x, y, w_, h_)
    if comp([x+w_, y]) > comp(corner_right_top):
        corner_right_top = [x+w_, y]

uu.imshow(im_dr)
print(corner_right_top)

# %% 绝对坐标
abspos_right_top = [w//2 + corner_right_top[0], corner_right_top[1]]

# %% 汇总3个角点，取图
#####################################################################

print(">>>", abspos_left_bottom, abspos_right_bottom, abspos_right_top)
pos_left_top = [abspos_left_bottom[0], abspos_right_top[1]]

# %%
tpl_text_zone = cv.crop2(im_tpl, pos_left_top, abspos_right_bottom)
uu.imshow(tpl_text_zone)

#####################################################################

# %% 滤波，二值化，多边形拟合
tpl_bin = cv.binary(tpl_text_zone, 111, 1)
tpl_med = cv.median(tpl_bin, 3)
uu.imshow(tpl_med)

#####################################################################

# %% 载入
path_check = uu.rpath("image/cap_2.jpg")
im_check = cv.imread(path_check)
uu.imshow(im_check)

# %% 重复tpl中的操作
tpl_med_bak = tpl_med
im_tpl = im_check

#####################################################################

# %% 将图像2转换为模板的尺寸
tpl_med = cv.resize(tpl_med, cv.size(tpl_med_bak))

# %% 比对： diff
im_diff = tpl_med_bak - tpl_med
uu.imshow(im_diff, 1)





#####################################################################
# %% 尝试通过文字切割，识别文字
#####################################################################

im_tpl = cv.rotate(im_tpl, 180)
uu.imshow(im_tpl)

# %%
import ocr
uu.reload(ocr)

TessEnv = {
    # "TesseractBinPath": "D:/programs/Tesseract",
    "TessDataDir": "/home/brt/ws/ipynb/Tutorial/tesseract/tessdata",
    # "TessDataDir": "D:/Home/workspace/ipynb/Tutorial/tesseract/tessdata",
    "Lang": "eng"
}

ocr_engine = ocr.Tesseract(TessEnv["TessDataDir"], TessEnv["Lang"])

# %%
x = ocr_engine.ndarray2text(im_tpl)
print(x)

# %%
im_check = cv.rotate(im_check, 180)
uu.imshow(im_check)

# %%
y = ocr_engine.ndarray2text(im_check)
print(y)

# %%
im_txt = cv.crop2(im_tpl, (600,180), (2000, 350))
uu.imshow(im_txt)

# %%
# ocr_engine.api.SetVariable("tessedit_char_whitelist", "")
# ocr_engine.api.SetVariable("classify_bln_numeric_mode", "1")
ocr_engine.ndarray2text(im_txt)
