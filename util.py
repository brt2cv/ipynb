import os
from importlib import reload

try:
    from utils.expy import *
except ImportError:
    #####################################################################
    # utils.expy@Version: 2.1.4
    #####################################################################
    import sys
    import os

    def path_expand(dir_lib, __file__=None, addsitedir=False):
        """ 当__file__为None时，dir_lib为绝对路径（或相对工作目录）
            否则，相对于传入的__file__所在目录引用dir_lib
        """
        if __file__ is not None:
            dir_lib = os.path.join(os.path.dirname(__file__), dir_lib)
        dir_lib_abs = os.path.abspath(dir_lib)
        if os.path.exists(dir_lib_abs):
            if dir_lib_abs not in sys.path:
                if addsitedir:
                    import site
                    str_func = "site_expand: "
                    site.addsitedir(dir_lib_abs)
                else:
                    str_func = "path_expand: "
                    sys.path.append(dir_lib_abs)
                print(str_func + f"动态加载Lib目录【{dir_lib_abs}】")
        else:
            raise Exception(f"无效的路径【{dir_lib_abs}】")
            # print(f"path_append: 无效的路径【{dir_lib_abs}】")

    def site_expand(dir_lib, __file__=None):
        """ 功能上强于path_append，会调用path目录下的*.pth文件
            但pyinstaller打包时，会提示site无法导入addsitedir问题
        """
        path_expand(dir_lib, __file__, True)


    from platform import system

    VENV_PATH_SITE_PACKAGES = "lib/site-packages" if system() == "Windows" else \
        "lib/python{}.{}/site-packages".format(*sys.version_info[:2])

    def venv_expand(path_venv):
        dir_lib = os.path.join(path_venv, VENV_PATH_SITE_PACKAGES)
        # if not os.path.exists(dir_lib):
        #     raise Exception(f"无效的路径【{dir_lib}】")
        site_expand(dir_lib)

    def topdir(dir_dst, override=False):
        dir_dst_abs = os.path.abspath(dir_dst)
        # 修改顶层目录
        if override:
            sys.path[0] = dir_dst_abs  # os.getcwd()
        else:
            sys.path.insert(0, dir_dst_abs)
        return dir_dst_abs

    def enpy(folder_name):
        """ 注意，目前的配置目录仅自用（个人配置的所有venv目录均位于 '$HOME/.enpy' ）"""
        if folder_name[0] == ".":  # 相对路径
            ENPY_PREFIX = os.path.abspath(os.path.curdir)
        elif folder_name[0] in ["/", "~"]:  # 绝对路径
            ENPY_PREFIX = ""
        else:
            ENPY_PREFIX = os.path.join(os.getenv("HOME"), ".enpy")
        path_venv = os.path.join(ENPY_PREFIX, folder_name)
        venv_expand(path_venv)

    def expy_pydev():
        path_pydev = os.path.join(os.getenv("HOME"), "local/utils/pydev")
        site_expand(path_pydev)

    #####################################################################
    # end of expy
    #####################################################################

expy_pydev()

import matplotlib.pyplot as plt
def imshow(im):
    plt.imshow(im)

dir_mvdev = os.path.join(os.path.abspath(
                         os.curdir), "mvdev")
if os.path.exists(dir_mvdev):
    path_expand(dir_mvdev)  # 用于 `import mvlib`

# from utils.base import get_caller_path  # ipython中不适用

#####################################################################
# dirRoot = os.path.join(os.environ["HOME"], "workspace/tfdev")
# path_expand(dirRoot)
# dirData = os.path.join(dirRoot, "datasets")

def isIpynbEnv():
    """ 网页中运行的 jupyter notebook """
    try:
        __file__
    except NameError:
        is_ipynb = True
    else:
        is_ipynb = False

    print(">>", f"Jutyper Notebook运行环境？【{is_ipynb}】")
    return is_ipynb


def isVscodeEnv():
    """ vscode中运行的jupyter """
    is_vscode = "JPY_PARENT_PID" in os.environ  # or ('VSCODE_PID' in os.environ)
    print(">>", f"基于Vscode的Jutyper运行环境？【{is_vscode}】")
    return is_vscode

isIpynb = isIpynbEnv()
isVscode = isVscodeEnv()
path = {
    "root": None,
    "data": None,
    "curr2root": None
}
# path["curr2root"] = ""
# dirRoot = ""
# dirData = ""

#####################################################################

def dirname(__file__):
    """ os.path.dirname(__file__)的使用：
        1. 当"print os.path.dirname(__file__)"所在脚本是以完整路径被运行的，那么将输出该脚本所在的完整路径，比如：
        `python d:/pythonSrc/test/test.py`
        那么将输出 d:/pythonSrc/test

        2. 当"print os.path.dirname(__file__)"所在脚本是以相对路径被运行的，那么将输出空目录，比如：
        `python test.py`
        那么将输出空字符串!!! 额，好大一个坑。。。
    """
    path = __file__.replace("\\", "/")
    return path.rsplit("/", 1)[0]

def isabs(path):
    """
    os.path.isabs('d:\...\yzx.py')
    在Windows调用Linux的Jupyter服务时，出现bug：Linux不能识别Windows的绝对路径
    """
    if path.find(":") > 0:  # 判断Windows系统的绝对路径
        return True
    return os.path.isabs(path)

#####################################################################

def chdir(__file__):
    """ jupyter中运行程序，ipynb文件会自动将工作目录定位到文件当前目录；
        而vscode中调用jupyter，工作目录为jupyter配置所定义的目录位置(tfdev目录)；
        而在实际环境中，则根据python运行的实际目录位置。

        由于每个例程相对独立，在项目目录内部运行而无需tfdev内部交互调用，故该函数
        的目的在于，将vscode的运行目录统一为jupyter的模式，也就类似进入到yolo_v3
        目录下，执行 `python main.py` 操作。
    """
    if isIpynb:
        # 获取root目录
        print("Jupyter Notebook 无需更换工作目录")
    elif not isVscode:
        # 获取root目录
        print("脚本运行时无需更换工作目录")
    else:
        # 切换工作目录
        cwd = os.getcwd()
        curr_dir = os.path.dirname(__file__)
        path["root"] = cwd
        path["curr2root"] = os.path.relpath(curr_dir, cwd)
        path["data"] = os.path.join(path["root"], "datasets")
        os.chdir(path["curr2root"])

    print("当前工作目录：", os.getcwd())

# def rpath(rel2curr):
#     """ 将相对于当前目录的路径转换为相对于根目录的路径 """
#     return os.path.join(path["curr2root"], rel2curr)

"""
def chdir2(curr2root):
    if isabs(curr2root):
        # 传入了__file__做参数
        print(">> 警告: 您当前使用__file__做参数，请确保并未在分布式环境下运行！")
        if isIpynb:
            raise Exception("Ipynb 环境中，chdir(curr2root) 参数不能为绝对路径")
        path["curr2root"] = os.path.relpath(dirname(curr2root), dirname(__file__))
    else:
        path["curr2root"] = curr2root

    # 获取root
    def root2curr():
        x = path["curr2root"].strip("/")
        size = len(x.split("/"))
        # y = "/".join([".."] * size)
        y = "../" * size
        return y

    path["root"] = root2curr()
    if isIpynb:
        os.chdir(path["root"])  # 重置工作目录

    path["data"] = os.path.join(path["root"], "datasets")
    print("当前工作目录：", os.getcwd())
"""

#####################################################################

def input(string):
    if isIpynb or isVscode:
        print(string)
    else:
        # 以脚本方式运行时，输出input
        return input(string)

def plot_history(histories, key):
    plt.figure(figsize=(16,10))

    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_'+key],
                       '--', label=name.title()+' Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
                 label=name.title()+' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_',' ').title())
    plt.legend()

    plt.xlim([0, max(history.epoch)])

    # plot_history([('baseline', baseline_history),
    #               ('small', small_history),
    #               ('big', big_history)], "acc")

"""
# 预置图像
from skimage import data

coffee = data.coffee()
logo = data.logo()
cat = data.chelsea()
# camera = data.camera()
# brick = data.brick()
# checkerboard = data.checkerboard()
# clock = data.clock()
# coffee = data.coffee()
# coins = data.coins()
# retina = data.retina()
# page = data.page()
# moon = data.moon()
# rocket = data.rocket()
"""
