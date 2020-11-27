#!/usr/bin/env python3
# @Date    : 2020-10-18
# @Author  : Bright Li (brt2@qq.com)
# @Link    : https://gitee.com/brt2
# @Version : 1.1.3

import os.path

#####################################################################
# pcall@Version : 0.2.1.x
#####################################################################
import subprocess

if hasattr(subprocess, 'run'):
    __PY_VERSION_MINOR = 5  # 高于3.5
# except AttributeError:
else:
    __PY_VERSION_MINOR = 4  # 低于3.4

def _popen(str_cmd):
    completing_process = subprocess.Popen(str_cmd,
                                shell=True,
                                # stdin=subprocess.DEVNULL,
                                # stderr=subprocess.PIPE,
                                stdout=subprocess.PIPE)
    # stdout, stderr = completing_process.communicate()
    return completing_process


def pcall(str_cmd, block=True):
    ''' return a list stdout-lines '''
    if block:
        if __PY_VERSION_MINOR == 5:
            p = subprocess.run(str_cmd,
                                shell=True,
                                check=True,
                                stdout=subprocess.PIPE)
        else:
            p = subprocess.check_call(str_cmd,
                                shell=True,
                                stdout=subprocess.PIPE)
        stdout = p.stdout
    else:
        p = _popen(str_cmd)
        stdout = p.communicate()  # timeout=timeout
    # rc = p.returncode
    return stdout.decode().splitlines()

#####################################################################
# end of pcall
#####################################################################

#####################################################################
# jhead@Version : 1.1.0
#####################################################################
def get_exif(path_jpg):
    stdout_ = pcall("jhead -se " + path_jpg)
    return stdout_

def remove_exif(path_jpg, forced=False):
    if forced:
        stdout_ = pcall("jhead -de " + path_jpg)
    else:
        stdout_ = pcall("jhead -purejpg " + path_jpg)
    return stdout_

def set_comment(path_jpg, comment):
    stdout_ = pcall("jhead -cl {} {}".format(comment, path_jpg))
    return stdout_

def clear_comment(path_jpg):
    stdout_ = pcall("jhead -dc " + path_jpg)
    return stdout_

def get_comment(path_jpg):
    stdout_ = pcall(r'jhead -se "%s"|grep Comment|awk "{print $3}" ' % path_jpg)
    if stdout_:
        return stdout_[0]

def get_resolution(path_jpg):
    stdout_ = pcall(r'jhead -se "%s"|grep Resolution ' % path_jpg)
    if stdout_:
        str_resolution = stdout_[0].split(": ")[1]
        return [int(x) for x in str_resolution.split(" x ")]

def get_size(path_jpg):
    """ depresscated: 建议直接使用 os.path.getsize()
        return an int-number of size by Bytes
    """
    stdout_ = pcall(r"jhead -se %s|grep 'File size'" % path_jpg)
    if not stdout_:
        return
    str_size = stdout_[0].split(": ")[1]
    return int(str_size.split()[0])

def size_density(path_jpg):
    """ 密度: 单位像素的空间占比，通过PIL压缩后的尺寸约为0.1-0.2 """
    # size = get_size(path_jpg, unit="B")
    stdout_ = pcall("jhead -se " + path_jpg)
    no_size = True
    for line in stdout_:
        if no_size:
            if line[:9] == "File size":
                str_size = line.split(": ")[1]
                size = int(str_size.split()[0])
                no_size = False
        elif line[:10] == "Resolution":
            str_resolution = line.split(": ")[1]
            w, h = [int(x) for x in str_resolution.split(" x ")]
            n_pixel = w * h
            break
    ratio = size / n_pixel
    return ratio

#####################################################################
# end of jhead
#####################################################################

if __name__ == "__main__":
    def getopt():
        import argparse

        parser = argparse.ArgumentParser("extract-exif", description="提取jpg图像的exif信息")
        parser.add_argument("path_jpg", action="store", help="图像路径")
        parser.add_argument("-c", "--comment", action="store_true", help="查看图像注释信息")
        parser.add_argument("-r", "--resolution", action="store_true", help="查看图像像素")
        parser.add_argument("-s", "--size", action="store_true", help="查看文件所占空间(KB)")
        return parser.parse_args()

    args = getopt()
    path_jpg = args.path_jpg

    if args.comment:
        print(get_comment(path_jpg))
    elif args.resolution:
        print(get_resolution(path_jpg))
    elif args.size:
        print(get_size(path_jpg), "KiB")
    else:
        print(get_exif(path_jpg))
