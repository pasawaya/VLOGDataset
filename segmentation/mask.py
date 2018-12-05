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

        h, w = image.shape[:2]
        for i in range(n_objects):
            class_id = int(boxes[0, 0, i, 1])
            if class_id in self.classes:
                box = boxes[0, 0, i, 3:7] * np.array([w, h, w, h])
                x_start, y_start, x_end, y_end = box.astype(np.int)
                w_box, h_box = x_end - x_start, y_end - y_start

                mask = masks[i, class_id]
                mask = cv2.resize(mask, (w_box, h_box), interpolation=cv2.INTER_NEAREST)
                mask = (mask > mask_threshold)

                mask_container = np.zeros((h, w), dtype=np.uint8)
                mask_container[y_start:y_end, x_start:x_end] = mask
                cleaned_masks.append(mask_container)
                scores.append(boxes[0, 0, i, 2])

        scores = np.array(scores)
        cleaned_masks = np.dstack(cleaned_masks) if cleaned_masks else np.array([])
        cleaned_masks = (np.moveaxis(cleaned_masks, 2, 0) * 255).astype(np.uint8)
        return scores, cleaned_masks
