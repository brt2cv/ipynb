
###############################################################################
# Name:         imgio
# Usage:
# Author:       Bright Li
# Modified by:
# Created:      2020-11-18
# Version:      [0.2.6]
# RCS-ID:       $$
# Copyright:    (c) Bright Li
# Licence:
###############################################################################

from PIL import Image

io_conf = {
    "backend": "pillow",
    "opencv_enable": True,
    "imageio_enable": True
}

try:
    import cv2
except ImportError:
    io_conf["opencv_enable"] = False
try:
    import numpy as np
    import imageio
except ImportError:
    io_conf["imageio_enable"] = False


def set_io_backend(name="pillow"):
    if name == "opencv":
        assert io_conf["opencv_enable"], "Module【opencv】不可用"
        io_conf["backend"] = "opencv"
    elif name == "numpy":
        assert io_conf["imageio_enable"], "Module【imageio】不可用"
        io_conf["backend"] = "numpy"
    elif name == "pillow":
        io_conf["backend"] = "pillow"
    else:
        raise Exception(f"未知的 Bakcend【{name}】")


#####################################################################

def imread(uri, format=None, **kwargs):
    """
    pilmode : str
        From the Pillow documentation:

        * 'L' (8-bit pixels, grayscale)
        * 'P' (8-bit pixels, mapped to any other mode using a color palette)
        * 'RGB' (3x8-bit pixels, true color)
        * 'RGBA' (4x8-bit pixels, true color with transparency mask)
        * 'CMYK' (4x8-bit pixels, color separation)
        * 'YCbCr' (3x8-bit pixels, color video format)
        * 'I' (32-bit signed integer pixels)
        * 'F' (32-bit floating point pixels)

        PIL also provides limited support for a few special modes, including
        'LA' ('L' with alpha), 'RGBX' (true color with padding) and 'RGBa'
        (true color with premultiplied alpha).

        When translating a color image to grayscale (mode 'L', 'I' or 'F'),
        the library uses the ITU-R 601-2 luma transform::

            L = R * 299/1000 + G * 587/1000 + B * 114/1000
    as_gray : bool
        If True, the image is converted using mode 'F'. When `mode` is
        not None and `as_gray` is True, the image is first converted
        according to `mode`, and the result is then "flattened" using
        mode 'F'.
    ----------------------
    Parameters for JPEG
        + exifrotate : bool
            Automatically rotate the image according to exif flag. Default True.
    ----------------------
    Parameters for PNG
        + ignoregamma : bool
            Avoid gamma correction. Default True.
    """
    if io_conf["backend"] == "pillow":
        im_pil = Image.open(uri)
        if kwargs.get("as_gray"):
            im_pil = im_pil._convert("L")  # 这里与imageio处理不同，并没有转换为32位的'F'灰度图
        return im_pil
    elif io_conf["backend"] == "opencv":
        if kwargs.get("as_gray"):
            im = cv2.imread(uri, cv2.IMREAD_GRAYSCALE)
        elif kwargs.get("pilmode") == "RGB":
            im = cv2.imread(uri, cv2.IMREAD_COLOR)  # 忽视透明度
        else:
            im = cv2.imread(uri, cv2.IMREAD_UNCHANGED)
        return im
    else:
        if kwargs.get("as_gray"):
            pilmode = kwargs.get("pilmode", "L")
            assert pilmode != "L", f"pilmode设置【{pilmode}】与 as_gray->【L】不匹配，请验证参数"
            kwargs["pilmode"] = "L"
            kwargs["as_gray"] = False
        return imageio.imread(uri, format, **kwargs)

def imwrite(uri, im):
    if io_conf["backend"] == "pillow":
        return im.save(uri)
    elif io_conf["backend"] == "opencv":
        return cv2.imwrite(uri, im)
    else:
        return imageio.imwrite(uri, im)

imsave = imwrite

#####################################################################

def shape2size(shape):
    """ im_arr.shape: {h, w, c}
        PIL.Image.size: {w, h}
    """
    size = (shape[1], shape[0])
    return size

def shape2mode(shape):
    """ 简单猜测 """
    if len(shape) < 3:
        return "L"
    elif shape[2] == 3:
        return "RGB"  # 无法区分BGR (OpenCV)
    elif shape[2] == 4:
        return "RGBA"
    else:
        raise Exception("未知的图像类型")

def ndarray2mode(im_arr):
    """ 一种预测图像mode的简单方式，用于ndarray转PIL时自动判断mode。
        mode: "1", "L", "P", "RGB", "BGR", "RGBA", "YUV", "LAB"
    """
    if im_arr.ndim < 3:
        return "L"
    else:
        shape = im_arr.shape
        if shape[2] == 3:
            return "RGB"  # 无法区分BGR (OpenCV)
        elif shape[2] == 4:
            return "RGBA"
        else:
            raise Exception("未知的图像类型")

#####################################################################

class _ImageFormat:
    def from_bytes(self, data, mode, **kwargs):
        """
        kwargs: size or shape of the image_data
        """
        if "size" in kwargs:
            size = kwargs["size"]
        elif "shape" in kwargs:
            size = shape2size(kwargs["shape"])
        else:
            raise Exception("必须传入size或shape参数")

        self._img = Image.frombytes(mode, size, data)

    def to_bytes(self):
        return self._img.tobytes()

    def from_numpy(self, im_arr, mode):
        """ 这里设定mode为显式参数，因为无法通过channel完全确定mode：
            * 2dim: "1", "L", "P", "I", "F"
            * 3dim: "RGB", "BGR"
            * 4dim: "RGBA", "CMYK", "YCbCr"
        """
        if mode == "BGR":
            cv2.cvtColor(im_arr, cv2.COLOR_BGR2RGB)
            mode = "RGB"
        self._img = Image.fromarray(im_arr, mode)

    def to_numpy(self):
        im = np.asarray(self._img)
        return im

    def from_pixmap(self, qt_img, type="QPixmap"):
        if type == "QPixmap":
            self._img = Image.fromqpixmap(qt_img)
        elif type == "QImage":
            self._img = Image.fromqimage(qt_img)
        else:
            raise Exception(f"Unkown type 【{type}】")

    def to_pixmap(self, type_="QPixmap"):
        if type_ == "QPixmap":
            return self._img.toqpixmap()
        elif type_ == "QImage":
            return self._img.toqimage()
        else:
            raise Exception(f"Unkown type 【{type_}】")

_convert = _ImageFormat()  # 全局转换器对象，用于图片格式转换

#####################################################################

def pillow2ndarray(pil_img):
    im_arr = np.asarray(pil_img)
    return im_arr

    # _convert.from_pillow(pil_img)
    # im_arr = _convert.to_numpy()
    # return im_arr

def ndarray2pillow(im_arr, mode=None):
    if mode is None:
        mode = ndarray2mode(im_arr)

    _convert.from_numpy(im_arr, mode)
    return _convert._img

def pillow2pixmap(pil_img, type_="QPixmap"):
    if type_ == "QPixmap":
        return pil_img.toqpixmap()
    elif type_ == "QImage":
        return pil_img.toqimage()
    else:
        raise Exception(f"Unkown type 【{type_}】")

def pixmap2pillow(qt_img):
    try:
        return Image.fromqpixmap(qt_img)
    except Exception as e:
        print(">>>", e)
        return Image.fromqimage(qt_img)

def ndarray2pixmap(im_arr, mode=None):
    if mode is None:
        mode = ndarray2mode(im_arr)

    _convert.from_numpy(im_arr, mode)
    pixmap = _convert.to_pixmap()
    return pixmap

def pixmap2ndarray(pixmap):
    _convert.from_pixmap(pixmap)
    im_arr = _convert.to_numpy()
    return im_arr

def ndarray2bytes(im_arr, mode=None):
    if mode is None:
        mode = ndarray2mode(im_arr)

    _convert.from_numpy(im_arr, mode)
    bytes_ = _convert.to_bytes()
    return bytes_

def bytes2ndarray(data, mode, **kwargs):
    """ kwargs:
            - size: (w, h)
            - shape: (h, w, ...)
    """
    if "size" in kwargs:
        size = kwargs["size"]
    elif "shape" in kwargs:
        size = shape2size(kwargs["shape"])
    else:
        raise Exception("必须传入size或shape参数")

    _convert.from_bytes(data, mode, size=size)
    im_arr = _convert.to_numpy()
    return im_arr


if __name__ == "__main__":
    path_img = "test/sample.jpg"

    def run_test(backend):
        set_io_backend(backend)
        im = imread(path_img)
        print(type(im))

    run_test("opencv")
    run_test("numpy")
    run_test("pillow")
