import cv2
import numpy as np


# Suppress AVX, etc. warnings when running on CPU
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class MaskRCNN:
    def __init__(self, classes=None):
        # Load classes and determine indices of desired object classes
        coco_classes = open('segmentation/model/object_detection_classes_coco.txt').read().strip().split('\n')
        if not classes:
            classes = coco_classes
        self.classes = [coco_classes.index(name) for name in classes]

    # TODO: load model once in init
    def detect(self, image, mask_threshold=1):
        textGraph = "segmentation/model/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"
        modelWeights = "segmentation/model/frozen_inference_graph.pb"

        net = cv2.dnn.readNetFromTensorflow(modelWeights, textGraph)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)
        net.setInput(blob)

        boxes, masks = net.forward(['detection_out_final', 'detection_masks'])

        n_objects = boxes.shape[2]
        scores = []
        cleaned_masks = []

        for i in range(n_objects):
            scores.append(boxes[0, 0, i, 2])
            class_id = int(boxes[0, 0, i, 1])
            mask = masks[i, class_id]
            mask = (mask > mask_threshold)
            cleaned_masks.append(mask)

        cleaned_masks = np.array(cleaned_masks)
        print(scores.shape)
        print(cleaned_masks.shape)
        return np.array(scores), np.array(cleaned_masks)
        # result = self.model.detect([image])[0]
        # classes = result['class_ids']
        # matches = [i for i, class_id in enumerate(classes) if class_id in self.classes]
        #
        # scores = result['scores'][matches]
        # masks = result['masks'][:, :, matches]
        # masks = np.moveaxis(masks, 2, 0) * 255
        # return scores, masks.astype(np.uint8)

