
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import logging
import numpy as np
import os
import uuid

import _init_paths 
from pycocotools.cocoeval import COCOeval

from core.config import cfg
from utils.io import save_object
import utils.boxes as box_utils
from datasets.dataset_catalog import DATASETS
from datasets.dataset_catalog import DEVKIT_DIR
from datasets.json_dataset import JsonDataset
from datasets.voc_eval import voc_eval

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
import matplotlib
from six.moves import cPickle as pickle

def _log_detection_eval_metrics(json_dataset, coco_eval):
    def _get_thr_ind(coco_eval, thr):
        ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                       (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
        iou_thr = coco_eval.params.iouThrs[ind]
        assert np.isclose(iou_thr, thr)
        return ind

    IoU_lo_thresh = 0.5
    IoU_hi_thresh = 0.95
    ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
    ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)
    # precision has dims (iou, recall, cls, area range, max dets)
    # area range index 0: all area ranges
    # max dets index 2: 100 per image
    precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, :, 0, 2]
    ap_default = np.mean(precision[precision > -1])
    print(
        '~~~~ Mean and per-category AP @ IoU=[{:.2f},{:.2f}] ~~~~'.format(
            IoU_lo_thresh, IoU_hi_thresh))
    print('{:.1f}'.format(100 * ap_default))
    for cls_ind, cls in enumerate(json_dataset.classes):
        if cls == '__background__':
            continue
        # minus 1 because of __background__
        precision = coco_eval.eval['precision'][
            ind_lo:(ind_hi + 1), :, cls_ind - 1, 0, 2]
        ap = np.mean(precision[precision > -1])
        print('{:.1f}'.format(100 * ap))
    print('~~~~ Summary metrics ~~~~')
    coco_eval.summarize()


def _do_detection_eval(json_dataset, res_file, output_dir):
    coco_dt = json_dataset.COCO.loadRes(str(res_file))
    coco_eval = COCOeval(json_dataset.COCO, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    _log_detection_eval_metrics(json_dataset, coco_eval)
    eval_file = os.path.join(output_dir, 'detection_results.pkl')
    save_object(coco_eval, eval_file)
    print('Wrote json eval results to: {}'.format(eval_file))
    return coco_eval


def _do_python_eval(json_dataset, salt, output_dir='output'):
    info = voc_info(json_dataset)
    year = info['year']
    anno_path = info['anno_path']
    image_set_path = info['image_set_path']
    devkit_path = info['devkit_path']
    cachedir = os.path.join(devkit_path, 'annotations_cache')
    aps = []
    # The PASCAL VOC metric changed in 2010
    # use_07_metric = True if int(year) < 2010 else False
    use_07_metric = False
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    # wrong_sample = None
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    confuse_mat = []
    for _, cls in enumerate(json_dataset.classes):
        if cls == '__background__' or cls == 'ignored_regions' or cls == 'others':
            confuse_mat.append([0 for i in range(len(cls_list))])
            continue
        filename = _get_voc_results_file_template(
            json_dataset, salt).format(cls)
        # pre_cls, num_obj, wrong_sample = voc_eval(
        #     filename, anno_path, image_set_path, cls, cachedir, ovthresh=0.5,
        #     use_07_metric=use_07_metric, debug=True, class_name=cls)
        # with open(output_dir+"/wrong_sample/"+cls+".json", 'w') as f:
        #     json.dump(wrong_sample, f)
        # print(len(pre_cls))
        # num_T = len(pre_cls)*3//4
        # temp = [np.sum(np.array(pre_cls[:num_T])==cls_list[i]) for i in range(len(cls_list))]
        # confuse_mat.append((np.array(temp)))
        

        rec, prec, ap = voc_eval(
            filename, anno_path, image_set_path, cls, cachedir, ovthresh=0.5,
            use_07_metric=use_07_metric)
        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls, ap))
        res_file = os.path.join(output_dir, cls + '_pr.pkl')
        save_object({'rec': rec, 'prec': prec, 'ap': ap}, res_file)
    print(np.mean(np.array(aps)))

def _get_voc_results_file_template(json_dataset, salt):
    info = voc_info(json_dataset)
    year = info['year']
    image_set = info['image_set']
    devkit_path = info['devkit_path']
    # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
    filename = 'comp4' + salt + '_det_' + image_set + '_{:s}.txt'
    return os.path.join(output_dir, filename)

def voc_info(json_dataset):
    year = json_dataset.name[4:8]
    image_set = json_dataset.name[9:]
    imageset_dir = json_dataset.imageset_dir
    devkit_path = DATASETS[json_dataset.name][DEVKIT_DIR]
    assert os.path.exists(devkit_path), \
        'Devkit directory {} not found'.format(devkit_path)
    anno_path = os.path.join(
        devkit_path, 'VOC' + year, 'Annotations', '{:s}.xml')
    image_set_path = os.path.join(
        devkit_path, 'VOC' + year, 'ImageSets', 'Main', imageset_dir + '.txt')
    return dict(
        year=year,
        image_set=image_set,
        devkit_path=devkit_path,
        anno_path=anno_path,
        image_set_path=image_set_path)

salt, output_dir = '_33db4568-4843-4277-82c0-4fd46b0f5973', './Outputs/voc2007/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x/May29-21-54-03_ubuntu_step/test'
dataset = JsonDataset("voc_2007_test")
cls_list = dataset.classes
cls_map = {cls_list[i]:i for i in range(len(cls_list))}
res_file = os.path.join(output_dir, "bbox_voc_2007_test_results.json")
_do_detection_eval(dataset, res_file, output_dir)
