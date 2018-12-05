from segmentation.mrcnn import model as modellib
from segmentation.mrcnn import coco
import numpy as np
import cv2


# Suppress AVX, etc. warnings when running on CPU
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


class MaskRCNN:
    def __init__(self, classes=class_names, n_gpus=1):
        class InferenceConfig(coco.CocoConfig):
            GPU_COUNT = n_gpus
            IMAGES_PER_GPU = 1

        self.model = modellib.MaskRCNN('inference', InferenceConfig(), 'segmentation/mrcnn/logs')
        self.model.load_weights('segmentation/mrcnn/mask_rcnn_coco.h5', by_name=True)
        self.classes = [class_names.index(name) for name in classes]

    def detect(self, image):
        textGraph = "./mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"
        modelWeights = "./mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb"

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

