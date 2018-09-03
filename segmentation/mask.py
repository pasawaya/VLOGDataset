from segmentation.mrcnn import model as modellib
from segmentation.mrcnn import coco
import numpy as np

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


class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


class MaskRCNN:
    static_model = modellib.MaskRCNN('inference', InferenceConfig(), 'segmentation/mrcnn/logs')
    static_model.load_weights('segmentation/mrcnn/mask_rcnn_coco.h5', by_name=True)

    def __init__(self, classes=class_names):
        self.model = modellib.MaskRCNN('inference', InferenceConfig(), 'segmentation/mrcnn/logs')
        self.model.load_weights('segmentation/mrcnn/mask_rcnn_coco.h5', by_name=True)
        self.classes = [class_names.index(name) for name in classes]

    def detect(self, image):
        result = self.model.detect([image])[0]
        classes = result['class_ids']
        matches = [i for i, class_id in enumerate(classes) if class_id in self.classes]

        scores = result['scores'][matches]
        masks = result['masks'][:, :, matches]
        masks = np.moveaxis(masks, 2, 0) * 255
        return scores, masks.astype(np.uint8)
