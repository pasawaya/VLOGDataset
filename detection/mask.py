
import logging
import numpy as np

# Suppress AVX, etc. warnings when running on CPU
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# import sys
# sys.path.insert(0, 'maskrcnn-benchmark')

from demo.predictor import COCODemo
from maskrcnn_benchmark.config import cfg

config_file = "maskrcnn-benchmark/configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"


class MaskRCNN:
    def __init__(self, confidence_threshold, area_threshold, classes=None):
        # update the config options with the config file
        cfg.merge_from_file(config_file)

        # manual override some options
        cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

        self.area_threshold = area_threshold
        self.demo = COCODemo(cfg, confidence_threshold=confidence_threshold)

        # Load classes and determine indices of desired object classes
        coco_classes = open('detection/classes.txt').read().strip().split('\n')
        if not classes:
            classes = coco_classes
        self.classes = [coco_classes.index(name) for name in classes]

    def detect(self, image):
        # Retain only predictions with confidence above confidence threshold
        predictions = self.demo.compute_prediction(image)
        predictions = self.demo.select_top_predictions(predictions)

        masks = predictions.get_field("mask").numpy()
        labels = predictions.get_field("labels").numpy()
        scores = predictions.get_field("scores").numpy()
        masks = np.squeeze(masks, 1)

        # Retain only desired object classes
        matches = [i for i, class_id in enumerate(labels) if class_id in self.classes]
        scores = scores[matches]
        masks = masks[matches, :, :]

        # Retain only masks below area threshold
        total_area = image.shape[0] * image.shape[1]
        areas = [np.count_nonzero(mask) for mask in masks]
        matches = [i for i, mask_area in enumerate(areas) if (mask_area / total_area) <= self.area_threshold]
        n_rejects = scores.shape[0] - len(matches)
        if n_rejects > 0:
            logging.getLogger('progress').info('Rejected ' + str(n_rejects) + ' due to area threshold')

        scores = scores[matches]
        masks = masks[matches, :, :]

        for score, area in zip(scores, areas):
            logging.getLogger('info').info(str(score) + '\t' + str(area) + '\t' + str(area / total_area))

        masks = np.clip(masks * 255., 0, 255).astype(np.uint8)
        return scores, masks
