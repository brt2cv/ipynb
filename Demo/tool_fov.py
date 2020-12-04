#!/usr/bin/env python3
# @Date    : 2020-12-04
# @Author  : Bright Li (brt2@qq.com)
# @Link    : https://gitee.com/brt2
# @Version : 0.2.1

import math

UvcResolution = {
    "0.3MP": [640,480],
    "0.5Mp": [800,600],
    "1MP": [1280,800],
    "2MP": [1920,1080],  # [1600,1200]
    "5MP": [2592, 1944],  # [2560,1920]
}

def test_for_params(resolution, dist, fov: list):
    print(f">>> 当前测试用例的分辨率【{resolution}】工作距离【{dist}】视野范围【{fov}】")
    w,h = fov
    # print("w/h:", round(w/h, 3))

    deg_w = math.atan2(w/2, dist) *2
    # deg_w = round(math.degrees(deg_w), 3)
    deg_h = math.atan2(h/2, dist) *2
    # deg_h = round(math.degrees(deg_h), 3)
    print("deg:", deg_w, deg_h)

    acc_w = round(w / resolution[0], 3)
    acc_h = round(h / resolution[1], 3)
    print("acc:", acc_w, acc_h)
    print()

    return (deg_w, deg_h)

def run_for_test():
    args = [  # 相机测量
        ([640,480], 60, [83,61.5]),
        ([640,480], 95, [122,91]),
        ([1920,1080], 95, [122,69]),
        ([2592, 1944], 95, [122.5,91.5]),
        ([2592, 1944], 137, [187,141])
    ]

    list_params = []
    for arg in args:
        deg_w, deg_h = test_for_params(*arg)
        list_params.append([deg_w, deg_h])

    deg_w_avg = round(sum([i[0] for i in list_params]) / len(list_params), 3)
    list_h_max = [i[1] for i in list_params if i[1] > 0.8]
    deg_h_avg = round(sum(list_h_max) / len(list_h_max), 3)

    print("[+] 当前摄像头的视野角为:", deg_w_avg, deg_h_avg)


FieldDegree = [1.167, 0.922]  # [66.88, 50.25]

def compute_fov(resolution, dist):
    def fov(dist):
        w = math.tan(FieldDegree[0] /2) * dist *2
        h = math.tan(FieldDegree[1] /2) * dist *2
        return (w,h)

    def acc(resolution, dist):
        w, _ = fov(dist)
        return round(w/resolution[0],3)

    FOV = fov(dist)
    accurate = acc(resolution, dist)
    print(f"[+] 当前条件下的视野范围【{FOV}】，精度【{accurate}】")

def comput_dist(fov):
    dist = round(fov/2 / math.tan(FieldDegree[0] /2), 2)
    print(f"[+] 需要至少在【{dist}】距离才能获取到足够的视野")


if __name__ == "__main__":
    import sys

    def getopt():
        import argparse

        parser = argparse.ArgumentParser("upload_cnblog", description="")
        parser.add_argument("-t", "--test", action="store_true", help="测试当前的相机参数")
        parser.add_argument("-r", "--resolution", type=float, default=5, help="相机分辨率，可选[0.3,0.5,1,2,5]")
        parser.add_argument("-d", "--dist", type=int, help="求特定工作距离dist下的视野范围，单位mm")
        parser.add_argument("-f", "--FOV", type=int, help="求满足特定FOV时的最小工作距离，单位mm")
        return parser.parse_args()

    args = getopt()
    if args.test:
        run_for_test()
        sys.exit()

    # ok = input("请选择相机分辨率: __MP [0.3,0.5,1,2,5]:")
    # assert ok
    # resolution = UvcResolution[f"{ok}MP"]
    resolution = UvcResolution[f"{args.resolution}MP"]
    if args.dist:
        # ok = input("请输入目标工作距离:")
        # assert ok
        # dist = float(ok)
        dist = args.dist
        compute_fov(resolution, dist)
    elif args.FOV:
        min_size = args.FOV
        comput_dist(min_size)
