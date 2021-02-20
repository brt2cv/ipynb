import os.path
import random
from PIL import Image, ImageFont, ImageDraw

from utils.log import getLogger
logger = getLogger(10)

class FontTplCreator:
    def __init__(self, dir_fonts, font_size, bkg_color="#FFFFFF", font_color="#000000"):
        self.dir_fonts = dir_fonts
        self.font_size = font_size
        self.font_color = font_color
        self.bkg_color = bkg_color

    def load_font(self, font_name):
        font_path = "{}/{}.ttf".format(self.dir_fonts, font_name)
        if not os.path.exists(font_path):
            font_path = "C:\\Windows/Fonts/{}.ttf".format(font_name)
            if not os.path.exists(font_path):
                raise Exception(f"未找到字体样式【{font_name}】")

        logger.debug(f"Font: {font_path}")
        font = ImageFont.truetype(font_path, self.font_size)
        return font

    def make_image(self, text, font, img_size):
        img = Image.new("L", img_size, self.bkg_color)
        # if img.mode == "RGB":
        #     im.draft("L", im.size)
        #     img.convert("L")
        draw = ImageDraw.Draw(img)
        draw.text((3, 3), text, font=font, fill=self.font_color)
        return img

    def random_image(self, dict_sample, chars_num, img_size):
        # font = random.choice(list(dict_sample.keys()))
        font_name, sample = random.choice(list(dict_sample.items()))
        font = self.load_font(font_name)
        list_chars = random.choices(sample, k=chars_num)
        chars = "".join(list_chars)
        logger.debug(f"生成字符图像【{chars}】")
        return self.make_image(chars, font, img_size)


# 以下两个函数根据项目调整目录名称
def gen_file_path(file_name, file_ext="jpg"):
    """ 用于解决重复文件名 """
    # 创建模板目录
    dir_path = f"./{file_ext}/"
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    file_path = dir_path + file_name + f".{file_ext}"
    if os.path.exists(file_path):
        file_path = dir_path + file_name + f"_ex.{file_ext}"
    return file_path

def save_img(img, file_name, file_ext="jpg"):
    try:
        save_path = gen_file_path(file_name, file_ext)
        img.save(save_path)
    except:
        save_path_ = gen_file_path(hex(ord(file_name)), file_ext)  # 使用Hex字符作为文件名
        logger.warning(f"无法保存的文件名【{save_path}】，另存为【{save_path_}】")
        img.save(save_path_)


if __name__ == "__main__":
    SET_NMBR = [str(i) for i in range(10)]
    SET_CHAR = [chr(i) for i in range(ord('A'), ord('Z')+1)]
    SET_CHAR_LOWER = [chr(i) for i in range(ord('a'), ord('z')+1)]
    SET_CHAR_ALL = SET_CHAR + SET_CHAR_LOWER

    #######################################################
    import os
    import shutils
    dir_ = "./jpg"
    if os.path.exists(dir_):
        # for file_ in os.listdir(dir_):
        #     os.remove(os.path.join(dir_, file_))
        shutils.rmtree(dir_)
    # else:

    os.makedirs(dir_)

    def create_random_chars():
        chars = {
            "OCR-A": SET_NMBR,
            "OCR-B-10-BT": SET_CHAR + ["-", "-", "."]
        }

        creator = FontTplCreator("./Fonts", 32)
        for i in range(10):
            img = creator.random_image(chars, 10, (350, 35))
            # save_img(img, char, file_ext="jpg")
            save_img(img, str(i), file_ext="jpg")

    def craete_chars():
        chars = [
            "0B0B0B0B0B",
            "00BB0B0B0BB"
        ]
        creator = FontTplCreator("./Fonts", 32)
        font = creator.load_font("BCS-SEMI-OCR-DEMO")
        for i, text in enumerate(chars):
            img = creator.make_image(text, font, (350, 35))
            # save_img(img, char, file_ext="jpg")
            save_img(img, str(i), file_ext="jpg")


    # create_random_chars()
    craete_chars()
