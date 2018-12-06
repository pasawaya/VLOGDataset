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

        self.demo = COCODemo(
            cfg,
            min_image_size=800,
            confidence_threshold=0.7,
        )

        # Load classes and determine indices of desired object classes
        coco_classes = open('segmentation/model/object_detection_classes_coco.txt').read().strip().split('\n')
        if not classes:
            classes = coco_classes
        self.classes = [coco_classes.index(name) for name in classes]

    def detect(self, image, mask_threshold=0.3):
        predictions = self.demo.run_on_opencv_image(image)
        print(predictions.shape)
        return 0
