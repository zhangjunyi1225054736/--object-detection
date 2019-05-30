#!/bin/bash

#n is the number od test images
#n=4
#step is the iteration step of the trained model
step=34999

cuda=2

rm -rf ./VOC2007/VOCdevkit2007/annotations_cache

#cp ./VOC2007/ImageSets/Main/test_$n.txt ./VOC2007/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt
#cp ./VOC2007/ImageSets/Main/test_$n.txt ./VOC2007/ImageSets/Main/test.txt
 
#python ./tools/pascal_voc_xml2coco_json_converter.py ./data/ 2007

#CUDA_VISIBLE_DEVICES=$cuda python tools/test_net.py --dataset voc2007 --cfg configs/baselines/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x.yaml \
         #--load_ckpt Outputs/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x/Apr28-14-53-17_zdyhpc_step/ckpt/model_step$step.pth --vis #--multi-gpu-testing

CUDA_VISIBLE_DEVICES=$cuda python tools/test_net.py --dataset voc2007 --cfg configs/baselines/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x.yaml \
         --load_ckpt /mnt/md126/zhangjunyi/365-object-detection/Outputs/voc2007/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x/May29-21-54-03_ubuntu_step/ckpt/model_step$step.pth #--vis #--multi-gpu-testing

