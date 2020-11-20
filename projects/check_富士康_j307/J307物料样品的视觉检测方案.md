<!--
+++
title       = "富士康J307物料样品的视觉检测方案"
description = ""
date        = "2020-11-17"
weight      = 5
tags        = []
categories  = []
keywords    = []
+++ -->

### J307物料样品的视觉检测方案

#### 整体描述

使用相机：HeroVision_535: 200万像素彩色嵌入式相机

光照环境：环形光源（R=70mm, color=Blue, angle=30）

光照强度：弱

+ 2.37面

    ![](/home/brt/workspace/ipynb2/check_富士康_j307/usb_cam_43.jpg)

+ 基准面

    ![](/home/brt/workspace/ipynb2/check_富士康_j307/usb_cam_1.jpg)

+ 底面

    ![](/home/brt/workspace/ipynb2/check_富士康_j307/usb_cam_45.jpg)

+ 横梁顶部

    ![](/home/brt/workspace/ipynb2/check_富士康_j307/usb_cam_16.jpg)

+ 彩色效果

    ![](/home/brt/workspace/ipynb2/check_富士康_j307/usb_cam_62.jpg)

#### 可检测的缺陷分类，及其检测原理

+ 胶未打饱

    ![](/home/brt/workspace/ipynb2/check_富士康_j307/usb_cam_11.jpg)

    ![](/home/brt/workspace/ipynb2/check_富士康_j307/usb_cam_2.jpg)

+ 定位柱变形

    ![](/home/brt/workspace/ipynb2/check_富士康_j307/usb_cam_59.jpg)

+ 横梁毛边

    ![](/home/brt/workspace/ipynb2/check_富士康_j307/usb_cam_17.jpg)

+ 胶口高（毛刺）

    ![](/home/brt/workspace/ipynb2/check_富士康_j307/usb_cam_9.jpg)

    ![](/home/brt/workspace/ipynb2/check_富士康_j307/usb_cam_7.jpg)

+ 头部溢胶

    ![](/home/brt/workspace/ipynb2/check_富士康_j307/usb_cam_12.jpg)

+ 多料

    ![](/home/brt/workspace/ipynb2/check_富士康_j307/usb_cam_0.jpg)

+ 可修毛边

    ![](/home/brt/workspace/ipynb2/check_富士康_j307/usb_cam_55.jpg)

#### 未检测类型，及其原因分析

+ 脏污

    ![](/home/brt/workspace/ipynb2/check_富士康_j307/usb_cam_62.jpg)

    尽管使用了彩色图像，但由于金属表面的脏污对比度较低，难以稳定的检测到该类性错误。

+ 框口漏铜

    框口的截面过细，像素宽度小于3，且截面光线反射零散，难以获得稳定特征。

+ 压伤，鼓包，柱面溢胀 & 其他

    未能观察到伤口位置，需要进一步沟通检测方式。
