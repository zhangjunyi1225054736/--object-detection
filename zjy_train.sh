rm -rf /mnt/md126/zhangjunyi/365-object-detection/data/cache

CUDA_VISIBLE_DEVICES=0,1,2,3 python  tools/train_net_step.py --dataset voc2007 --cfg configs/baselines/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x.yaml  --bs 4 #--load_ckpt ./data/pretrained_model/UAVDT_model.pth  #--lr 0.00001

#CUDA_VISIBLE_DEVICES=0,1,2,3 python  tools/train_net_step.py --dataset voc2007 --cfg configs/baselines/e2e_mask_rcnn_X-101-64x4d-FPN_2x.yaml  --bs 4 #--load_ckpt ./data/pretrained_model/UAVDT_model.pth  #--lr 0.00001
