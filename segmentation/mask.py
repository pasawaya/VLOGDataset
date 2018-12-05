import cv2


# Suppress AVX, etc. warnings when running on CPU
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class MaskRCNN:
    def __init__(self, classes=None):
        coco_classes = open('segmentation/model/object_detection_classes_coco.txt').read().strip().split('\n')

        if not classes:
            self.classes = coco_classes
        self.classes = [coco_classes.index(name) for name in self.classes]

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

