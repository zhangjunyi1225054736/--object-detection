CUDA_VISIBLE_DEVICES=3 python infer_simple.py  --dataset coco --cfg ../configs/baselines/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x.yaml \
         --load_ckpt /mnt/md126/zhangjunyi/365-object-detection/Outputs/voc2007/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x/May29-21-54-03_ubuntu_step/ckpt/model_step34999.pth \
         --images /mnt/md126/zhangjunyi/365-object-detection/VOC2007/ImageSets/Main/test.txt


