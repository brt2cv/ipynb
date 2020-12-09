> [API说明](https://pypi.org/project/tesserocr/)
> [Tesserocr_Github](https://github.com/sirfz/tesserocr)
> [csdn: Tesseract-OCR学习系列（四）API](https://blog.csdn.net/striving1234/article/details/78086255)

# %%
import ipyenv as uu
# uu.enpy("opencv")
uu.reload(cv)

# %%
from PIL import Image
import pycv.pillow as cv

import tesserocr
from tesserocr import PyTessBaseAPI, RIL

TessDataDir = "D:/Home/workspace/tmv/src/plugins/tesseract/tessdata/demo"
TessLang = "eng"

tesserocr.tesseract_version()

# %%
api = PyTessBaseAPI(path=TessDataDir, lang=TessLang)
# api.SetVariable("tessedit_char_whitelist", "0123456789")
# api.SetVariable("tessedit_char_blacklist", "abcdefghijklmn")
# api.SetVariable("classify_bln_numeric_mode", "1")

# %%
path_img = "tmp/ocr.jpg"
im = cv.imread(path_img)
uu.imshow(im)

# %%
im2 = cv.invert(im)
uu.imshow(im2)

# %% 识别文字方向、角度、识别结果
api.SetImage(im_smooth)
# api.SetVariable("tessedit_char_whitelist", "0123456789")

# api.Recognize()
print(">>>{}<<<".format(api.GetUTF8Text()))
print(api.AllWordConfidences())

it = api.AnalyseLayout()
orientation, direction, order, deskew_angle = it.Orientation()
print("Orientation: {:d}".format(orientation))
print("WritingDirection: {:d}".format(direction))
print("TextlineOrder: {:d}".format(order))
print("Deskew angle: {:.4f}".format(deskew_angle))

# %% 划分字符
im_dr = im.copy()

api.SetImage(im_dr)
# 查找图像内的图像块，并将分割到的图像块返回给Boxa结构中
boxes = api.GetComponentImages(RIL.TEXTLINE, True)
"""
参数1：
enum PageIteratorLevel {
  RIL_BLOCK,     // Block of text/image/separator line.
  RIL_PARA,      // Paragraph within a block.
  RIL_TEXTLINE,  // Line within a paragraph.
  RIL_WORD,      // Word within a textline.
  RIL_SYMBOL     // Symbol/character within a word.
};

参数2：
text_only如果是true的话，就表示只返回文字区域坐标，不返回图像区域坐标
"""
print('Found {} textline image components.'.format(len(boxes)))

for i, (_im, box, _, _) in enumerate(boxes):
    # _im is a PIL image object
    # box is a dict with x, y, w and h keys
    off = 5
    rectangle = [box['x']-off, box['y']-off, box['w']+2*off, box['h']+2*off]
    api.SetRectangle(*rectangle)
    api.SetVariable("tessedit_char_whitelist", "0123456789")
    ocrResult = api.GetUTF8Text()
    conf = api.MeanTextConf()
    print(u"Box[{0}]: x={x}, y={y}, w={w}, h={h}, "
            "confidence: {1}, text: {2}".format(i, conf, ocrResult, **box))
    cv.draw_rect(im_dr, *rectangle, 0)

uu.imshow(im_dr)

# %% 逐个字符求取其可选

from tesserocr import PyTessBaseAPI, RIL, iterate_level

api.SetImage(im_dr)
api.SetVariable("save_blob_choices", "T")
# api.SetRectangle(37, 228, 548, 31)
api.Recognize()

ri = api.GetIterator()
level = RIL.SYMBOL
for r in iterate_level(ri, level):
    symbol = r.GetUTF8Text(level)  # r == ri
    conf = r.Confidence(level)
    if symbol:
        print(u'symbol {}, conf: {}'.format(symbol, conf), end='')
    indent = False
    ci = r.GetChoiceIterator()
    for c in ci:
        if indent:
            print('\t\t ', end='')
        print('\t- ', end='')
        choice = c.GetUTF8Text()  # c == ci
        print(u'{} conf: {}'.format(choice, c.Confidence()))
        indent = True
    print('---------------------------------------------')
