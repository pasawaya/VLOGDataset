import cv2
import numpy as np

import sys
sys.path.insert(0, 'segmentation/maskrcnn/demo')

from predictor import COCODemo
from maskrcnn_benchmark.config import cfg

# Suppress AVX, etc. warnings when running on CPU
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

config_file = "segmentation/maskrcnn/configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"


class MaskRCNN:
    def __init__(self, classes=None):
        # update the config options with the config file
        cfg.merge_from_file(config_file)

        # manual override some options
        cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

        self.demo = COCODemo(cfg, min_image_size=800, confidence_threshold=0.7)

        # Load classes and determine indices of desired object classes
        coco_classes = open('segmentation/model/object_detection_classes_coco.txt').read().strip().split('\n')
        if not classes:
            classes = coco_classes
        self.classes = [coco_classes.index(name) for name in classes]

    def detect(self, image):
        predictions = self.demo.compute_prediction(image)
        masks = predictions.get_field("mask").numpy()
        labels = predictions.get_field("labels").numpy()
        scores = predictions.get_field("scores").numpy()
        masks = np.squeeze(masks, 1)

        # Retain only desired object classes
        matches = [i for i, class_id in enumerate(labels) if class_id in self.classes]
        scores = scores[matches]
        masks = masks[matches, :, :]

        return scores, masks
