from segmentation.mrcnn import model as modellib
from segmentation.mrcnn import coco

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
    def __init__(self, classes=None,
                 n_gpus=1,
                 batch_size=1,
                 model_dir='segmentation/mrcnn/logs',
                 weights_dir='segmentation/mrcnn/mask_rcnn_coco.h5'):

        class InferenceConfig(coco.CocoConfig):
            GPU_COUNT = n_gpus
            IMAGES_PER_GPU = batch_size

        self.model = modellib.MaskRCNN(mode="inference",
                                       model_dir=model_dir,
                                       config=InferenceConfig())
        self.model.load_weights(weights_dir, by_name=True)

        if classes is None:
            classes = class_names

        self.classes = [class_names.index(name) for name in classes]

    def detect(self, image):
        result = self.model.detect([image])[0]
        classes = result['class_ids']
        matches = [i for i, class_id in enumerate(classes) if class_id in self.classes]

        scores = result['scores'][matches]
        masks = result['masks'][:, :, matches]
        return scores, masks

