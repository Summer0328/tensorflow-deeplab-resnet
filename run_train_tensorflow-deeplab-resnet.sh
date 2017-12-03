#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

train_script=${HOME}/codes/PycharmProjects/tensorflow-deeplab-resnet/train.py

#dataDir="/home/hlc/Data/aws_SpaceNet/un_gz/voc_format"
dataDir=/home/hlc/Data/VOCdevkit/VOC2012
#data_list=/home/hlc/Data/aws_SpaceNet/un_gz/voc_format/AOI_2_Vegas_Train/trainval_aug_path.txt
#num_classes=2
batchSize=2

pre_train_model=${HOME}/Data/aws_SpaceNet/deeplab_exper/spacenet_rgb_aoi_2/model/deeplab_resnet_init.ckpt
#pre_train_model=./snapshots/model.ckpt-3000.data-00000-of-00001

#run script with python3 and tensorflow
#python ${train_script} --data-dir ${dataDir} --data-list ${data_list} --restore-from ${pre_train_model} --num-classes ${num_classes} --batch-size ${batchSize}  --ignore-label 255 --not-restore-last

python ${train_script} --data-dir ${dataDir}  --restore-from ${pre_train_model}  --batch-size ${batchSize}  --ignore-label 255 --not-restore-last --save-pred-every 2000
