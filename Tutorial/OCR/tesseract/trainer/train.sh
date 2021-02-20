#!/usr/bin/bash
# @Date    : 2020-12-09
# @Author  : Bright Li (brt2@qq.com)
# @Link    : https://gitee.com/brt2
# @Version : 0.2.2

if [ $# -lt 2 ]; then
    echo "Usage: train.sh <workdir> <lang_name> <base_trainer>"  # <base> is a box-file
    # 注意：<base_trainer>需要将tif/box文件拷贝到<folder_dir>目录中
    echo "  Example: train.sh  ./png/  ocr_b  ocr_a.font.exp0 eng.font.exp0..."
    exit
fi

DIR_EDITOR=/d/Home/workspace/ipynb/tutorial/tesseract/jTessBoxEditor
export TESSDATA_PREFIX=/d/Home/workspace/ipynb/tutorial/tesseract/tessdata/default

jTessBoxEditor=$DIR_EDITOR/jTessBoxEditor.jar
PATH=$PATH:$DIR_EDITOR/tesseract-ocr

cd $1  # folder_dir

lang=$2
font="font"  # 固定格式

num="0"
train_name="${lang}.${font}.exp${num}"

base_trainer=$3  # wafer.font.exp0

# 使用jTessBoxEditor合并样本图片（手动运行）
read -p "是否开启【jTessBoxEditor】程序 [Y/n]: " any
if [[ ${any}_ != "n_" ]]; then
    java -jar $jTessBoxEditor &
fi

read -p "请确认已生成tif图像【${train_name}.tif】，输入回车继续: " any

function create_box() {
# lang_base=$1

    read -p "绘制Box的解析语言 [eng]: " lang_base
    test -z $lang_base && lang_base="eng"
    # echo "使用【$lang_base】绘制box"

    # 如提示：read_params_file: Can't open batch.nochop，请确认TESSDATA_PREFIX路径下存在\tessconfigs\batch.nochop文件
    tesseract ${train_name}.tif $train_name -l $lang_base --psm 7 batch.nochop makebox
    [[ $? -ne 0 ]] && exit

    read -p "请手动校正样本的识别结果，完成后输入回车继续..." any
}

# 生成BOX文件
if [ -f ${train_name}.box ]; then
    read -p "已存在【${train_name}.box】，是否重置[n/Y]: " override
    if [[ ${override}_ == "Y_" ]]; then
        create_box
    fi
else
    create_box
fi

function append_ext() {
args=($*)
ext=${args[-1]}
unset args[-1]

    new_names=""
    for name in ${args[*]}; do
        new_names=$new_names$name$ext" "
    done
    echo $new_names
}

function train_tif() {
names=$*

    # 粗体、倾斜等共计5个属性: <fontname> <italic> <bold> <fixed> <serif> <fraktur>
    for name in $names; do
        echo -e "\n################## train_tif( ${name} ) ##################\n"
        echo "${name} 0 0 0 0 0" >> font_properties

        # 生成.tr训练文件
        tesseract ${name}.tif ${name} nobatch box.train

        # 生成字符集文件（生成unicharset文件）
        unicharset_extractor ${name}.box
    done

    tr_names=`append_ext ${names} .tr`

    # # 生成shape文件（生成 shapetable 和 num.unicharset 两个文件）
    # shapeclustering -F font_properties -U unicharset -O ${lang}.unicharset ${name}.tr

    # 生成聚字符特征文件（生成 inttemp、pffmtable、shapetable和unicharset四个文件）
    mftraining -F font_properties -U unicharset -O ${lang}.unicharset ${tr_names}

    # 生成字符正常化特征文件（生成 normproto 文件）
    cntraining ${tr_names}
}

train_tif $train_name $base_trainer

# 文件重命名
mv normproto ${lang}.normproto
mv inttemp ${lang}.inttemp
mv pffmtable ${lang}.pffmtable
mv shapetable ${lang}.shapetable

# 合并训练文件（生成 num.traineddata 文件）
combine_tessdata ${lang}.

# 清理过程文件
read -p "是否清除过程文件[Y/n]: " clear
if [[ ${clear}_ != "n_" ]]; then
    rm font_properties unicharset *.unicharset *.tr *.inttemp *.pffmtable *.shapetable *.normproto program.log
fi

echo "已生成训练文件，Well Done"