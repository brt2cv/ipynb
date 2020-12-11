<!-- 参考: https://blog.csdn.net/a745233700/article/details/80175883 -->
<!-- https://www.jianshu.com/p/c8ba23ec672a -->

## 文本训练的要求
1. 在一幅图片内，字体统一，决不能将多种字体混合出现在一幅训练图片内；如果不是通过扫描文本获取的字符图片，这个条件很容易被忽视。
2. 理想条件下，同种字体的字符图片集中到一幅大的训练图片中，在同一页内；
3. 要保留一定的字符间距与行间距；
4. 字符高度（大小），只要满足高度最小条件即可，对于小写字符x，其高度要至少大于10个像素，一般统一采用一种大小即可，tesseract engine默认的training数据集也是一种大小；
5. 对于非字母字符，如!@#$%^&(),.{}<>/?，不要集中在一起出现，原因是这样不利于tesseract找出文本行基线baseline，不利于文本高度及大小的检测，baseline检测是tesseract engine的第一步；
6. 一般每个字符需要10个样本，高频常见字符至少20个样本，不常见字符需要5个样本；
7. 对于同种字体，多页训练图片，可以在训练中，件用相同的方式合并tr文件和box文件，两类文件内的字符次序要相同，利于提高训练效果。

在获取训练字符图片方面，不一定非要从待识别图片中收集，可以利用word字符集找到对应字体，打印，扫描，获取训练图片，简单、方便。这个根据实际情况来应用。

原文链接：https://blog.csdn.net/viewcode/article/details/7849448

## 训练集流程
1. 使用draw_ocr_tpl.py生成字符样本(jpg图片即可)
1. 使用jTessBoxEditor工具 > Tools > Merge TIFF，输出文件名：num.font.exp0.tif
合并后的文件命名格式有一定要求 
【语法】：[lang].[fontname].exp[num].tif 
1. cmd 执行（生成文件num.font.exp0.box）：
tesseract num.font.exp0.tif num.font.exp0 -l eng --psm 7 batch.nochop makebox

1. 将上一步生成的.box和.tif样本文件放在同一目录，运行jTessBoxEditor > Box Editor > open > num.font.exp0.tif ，校正文件识别的错误，保存！
1. 执行如下命令： echo "test 0 0 0 0 0" > font_properties
表示字体test的粗体、倾斜等共计5个属性。
1. 生成.tr训练文件（生成num.font.exp0.tr文件）
执行如下命令： tesseract num.font.exp0.tif num.font.exp0 nobatch box.train
1. 生成字符集文件（生成一个名为“unicharset”的文件）
执行命令： unicharset_extractor num.font.exp0.box
-->> 无法执行unicharset_extractor
1. 生成shape文件（生成 shapetable 和 num.unicharset 两个文件）
shapeclustering -F font_properties -U unicharset -O num.unicharset num.font.exp0.tr
1. 生成聚字符特征文件（生成 inttemp、pffmtable、shapetable和zwp.unicharset四个文件）
mftraining -F font_properties -U unicharset -O num.unicharset num.font.exp0.tr
1. 生成字符正常化特征文件（生成 normproto 文件）
cntraining num.font.exp0.tr
1. 文件重命名
mv normproto num.normproto
mv inttemp num.inttemp
mv pffmtable num.pffmtable
mv shapetable num.shapetable

1. 合并训练文件（生成 num.traineddata 文件）
combine_tessdata num.
-->> Log输出中的Offset 1、3、4、5、13这些项不是-1，表示新的语言包生成成功。