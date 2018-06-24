
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

max_video_idx = 17097

parser = argparse.ArgumentParser(description='Download VLOG dataset and bottle masks.')
parser.add_argument('--gpu_count',
                    default=1,
                    type=int,
                    help='Number of GPUs to use')
parser.add_argument('--images_per_gpu',
                    default=1,
                    type=int,
                    help='Batch size')
parser.add_argument('--n_videos',
                    default=max_video_idx,
                    type=int,
                    help='Number of videos to download')
parser.add_argument('--start_video_id',
                    default=0,
                    type=int,
                    help='The index of the first video to download')
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

n = args.n_videos
start_video_id = args.start_video_id
gpu_count = args.gpu_count
images_per_gpu = args.images_per_gpu
t_area = args.area_threshold
t_confidence = args.confidence_threshold
inpaint_method = args.inpaint_method
checkpoint = args.checkpoint_dir


class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = gpu_count
    IMAGES_PER_GPU = images_per_gpu


if not os.path.exists(raw_videos_directory):
    os.makedirs(raw_videos_directory)
if not os.path.exists(trimmed_videos_directory):
    os.makedirs(trimmed_videos_directory)
if not os.path.exists(data_directory):
    os.makedirs(data_directory)

# Identify video indices where a bottle is present
bottle_idx = 4
labels = np.load('meta/hand_object/hand_object.npy')
indices = np.where(labels[:, bottle_idx] == 1)[0]

# Get links corresponding to those indices
youtube_file = open('meta/youtube_links.txt')
entries = [entry for i, entry in enumerate(youtube_file) if i in indices]
youtube_file.close()

# Initialize Mask R-CNN
if not os.path.exists(weights_directory):
    utils.download_trained_weights(weights_directory)

model = modellib.MaskRCNN(mode="inference", model_dir=model_directory, config=InferenceConfig())
model.load_weights(weights_directory, by_name=True)

# Get indices of desired objects to mask
desired_classes = ['bottle', 'cup', 'bowl', 'wine glass']
desired_classes = [class_names.index(class_name) for class_name in desired_classes]

downloader = YoutubeDownloader('mp4', start_id=start_video_id)
for video_id in range(start_video_id, min(start_video_id + n, max_video_idx - 1)):
    entry = entries[video_id]
    url, start, stop = entry.split(' ')

    # Download video, trim to specified clip, then delete original video
    print('[video ' + str(video_id) + ']')
    print('\t[downloading video ' + str(video_id) + ']')
    video = downloader.download_url(url, raw_videos_directory)
    if video is not None:
        print('\t[trimming video ' + str(video_id) + ']')
        trimmed = VideoTransformer.trim(video, int(start), int(stop), trimmed_videos_directory)
        os.remove(video.name)
        frames = trimmed.load_frames(fps=video.fps)

        # Create directories for current video frames and masks
        current_video_directory = os.path.join(data_directory, str(video_id))
        frames_directory = os.path.join(current_video_directory, 'frames')
        masks_directory = os.path.join(current_video_directory, 'masks')
        if not os.path.exists(current_video_directory):
            os.makedirs(current_video_directory)
        if not os.path.exists(frames_directory):
            os.makedirs(frames_directory)
        if not os.path.exists(masks_directory):
            os.makedirs(masks_directory)

        video_has_annotations = False
        print('\t[processing video ' + str(video_id) + ']')
        for frame_id in range(len(frames)):
            current_frames_directory = os.path.join(frames_directory, str(frame_id))
            current_masks_directory = os.path.join(masks_directory, str(frame_id))
            if not os.path.exists(current_frames_directory):
                os.makedirs(current_frames_directory)
            if not os.path.exists(current_masks_directory):
                os.makedirs(current_masks_directory)
            frame = frames[frame_id]

            # Run Mask R-CNN
            result = model.detect([frame])[0]
            classes = result['class_ids']
            scores = result['scores']
            matches = [i for i, class_id in enumerate(classes) if class_id in desired_classes]
            masks = result['masks']

            # Save masks
            mask_id = 0
            total_area = frame.shape[0] * frame.shape[1]
            frame_has_annotations = False
            for match in matches:
                mask = (masks[:, :, match] * 255).astype(np.uint8)
                confidence = scores[match]

                # Verify that detection is confident and below area threshold
                area = np.count_nonzero(mask)
                area_ratio = float(area) / float(total_area)

                if confidence >= t_confidence and area_ratio <= t_area:
                    # In-paint and save frame
                    dilated_mask = cv2.dilate(mask, np.ones((9, 9)), iterations=2)

                    if inpaint_method == 'generative':
                        inpainted = generative_inpaint(frame, dilated_mask, checkpoint)
                    else:
                        inpainted = telea_inpaint(frame, dilated_mask)

                    inpainted_path = os.path.join(current_frames_directory, str(mask_id) + '.png')
                    misc.imsave(inpainted_path, inpainted)

                    # Save mask
                    mask_path = os.path.join(current_masks_directory, str(mask_id) + '.png')
                    misc.imsave(mask_path, mask)

                    mask_id += 1
                    video_has_annotations = True
                    frame_has_annotations = True

            if not frame_has_annotations:
                shutil.rmtree(current_frames_directory)
                shutil.rmtree(current_masks_directory)

        if not video_has_annotations:
            shutil.rmtree(current_video_directory)

        # Delete trimmed video
        os.remove(trimmed.name)
    else:
        print('\t[skipping video ' + str(video_id) + ']')

# Delete raw videos directory
shutil.rmtree(raw_videos_directory)
shutil.rmtree(trimmed_videos_directory)
