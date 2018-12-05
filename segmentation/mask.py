from segmentation.mrcnn import model as modellib
from segmentation.mrcnn import coco
import numpy as np
import cv2


# Suppress AVX, etc. warnings when running on CPU
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class MaskRCNN:
    def __init__(self, classes=None, n_gpus=1):
        class InferenceConfig(coco.CocoConfig):
            GPU_COUNT = n_gpus
            IMAGES_PER_GPU = 1

        coco_classes = open('segmentation/model/object_detection_classes_coco.txt').read().strip().split('\n')
        print(coco_classes)
        if not classes:
            self.classes = coco_classes

        self.model = modellib.MaskRCNN('inference', InferenceConfig(), 'segmentation/mrcnn/logs')
        self.model.load_weights('segmentation/mrcnn/mask_rcnn_coco.h5', by_name=True)
        self.classes = [coco_classes.index(name) for name in classes]

    def detect(self, image):
        textGraph = "segmentation/model/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"
        modelWeights = "segmentation/model/frozen_inference_graph.pb"

        net = cv2.dnn.readNetFromTensorflow(modelWeights, textGraph)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)
        net.setInput(blob)

        boxes, masks = net.forward(['detection_out_final', 'detection_masks'])
        print(boxes.shape)
        print(masks.shape)
        return 0
        # result = self.model.detect([image])[0]
        # classes = result['class_ids']
        # matches = [i for i, class_id in enumerate(classes) if class_id in self.classes]
        #
        # scores = result['scores'][matches]
        # masks = result['masks'][:, :, matches]
        # masks = np.moveaxis(masks, 2, 0) * 255
        # return scores, masks.astype(np.uint8)

