
import numpy as np
from video import *
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import coco
import shutil
import scipy.misc as misc
import argparse
import warnings
from inpaint import *


warnings.filterwarnings('ignore')

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

weights_directory = 'mrcnn/mask_rcnn_coco.h5'
model_directory = 'mrcnn/logs'

raw_videos_directory = 'temp'
trimmed_videos_directory = 'videos'
data_directory = 'data'

original_subdir = os.path.join(data_directory, 'images')
inpainted_subdir = os.path.join(data_directory, 'inpainted')
mask_subdir = os.path.join(data_directory, 'masks')

max_video_idx = 17097

parser = argparse.ArgumentParser(description='Download VLOG dataset and bottle masks.')
parser.add_argument('--data_dir',
                    default=None,
                    type=str,
                    help='Directory containing bottle videos')
parser.add_argument('--confidence_threshold',
                    default=0.8,
                    type=float,
                    help='Confidence threshold below which mask will be discarded.')
parser.add_argument('--area_threshold',
                    default=0.07,
                    type=float,
                    help='Area threshold above which mask will be discarded.')
parser.add_argument('--inpaint_method',
                    default='generative',
                    type=str,
                    help='Inpainting method.')
parser.add_argument('--checkpoint_dir',
                    default='model_logs',
                    type=str,
                    help='Directory containing generative in-painting pre-trained model.')

args = parser.parse_args()

t_area = args.area_threshold
t_confidence = args.confidence_threshold
inpaint_method = args.inpaint_method
checkpoint = args.checkpoint_dir
visualize = True


class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


if not os.path.exists(data_directory):
    os.makedirs(data_directory)
if not os.path.exists(original_subdir):
    os.makedirs(original_subdir)
if not os.path.exists(inpainted_subdir):
    os.makedirs(inpainted_subdir)
if not os.path.exists(mask_subdir):
    os.makedirs(mask_subdir)

# Initialize Mask R-CNN
if not os.path.exists(weights_directory):
    utils.download_trained_weights(weights_directory)

model = modellib.MaskRCNN(mode="inference", model_dir=model_directory, config=InferenceConfig())
model.load_weights(weights_directory, by_name=True)

# Get indices of desired objects to mask
desired_classes = ['bottle', 'cup', 'bowl', 'wine glass']
desired_classes = [class_names.index(class_name) for class_name in desired_classes]


video_names = [video_name for video_name in os.listdir(args.data_dir) if 'mp4' in video_name]
current = 0
for video_id in range(len(video_names)):
    video_path = os.path.join(args.data_dir, video_names[video_id])
    video = Video(video_path)
    if video is not None:
        frames = video.load_frames(fps=video.fps)

        print('\t[processing video ' + str(video_id) + ']')
        for frame_id in range(len(frames)):
            frame = frames[frame_id]

            # Run Mask R-CNN
            result = model.detect([frame])[0]
            classes = result['class_ids']
            scores = result['scores']
            matches = [i for i, class_id in enumerate(classes) if class_id in desired_classes]
            masks = result['masks']

            # Save masks
            total_area = frame.shape[0] * frame.shape[1]
            for match in matches:
                mask = (masks[:, :, match] * 255).astype(np.uint8)
                confidence = scores[match]

                # Verify that detection is confident and below area threshold
                area = np.count_nonzero(mask)
                area_ratio = float(area) / float(total_area)

                if confidence >= t_confidence and area_ratio <= t_area:
                    dilated_mask = cv2.dilate(mask, np.ones((9, 9)), iterations=2)
                    temp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    inpainted = generative_inpaint(temp, dilated_mask, checkpoint)

                    # Save in-painted
                    inpainted_path = os.path.join(inpainted_subdir, str(current) + '.png')
                    inpainted = cv2.resize(inpainted, (320, 240))
                    misc.imsave(inpainted_path, inpainted)

                    # Save mask
                    mask_path = os.path.join(mask_subdir, str(current) + '.png')
                    mask = cv2.resize(mask, (320, 240))
                    misc.imsave(mask_path, mask)

                    # Save original
                    original_path = os.path.join(original_subdir, str(current) + '.png')
                    original = cv2.resize(frame, (320, 240))
                    misc.imsave(original_path, original)

                    current += 1
    else:
        print('\t[skipping video ' + str(video_id) + ']')
