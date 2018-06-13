
import numpy as np
from Video import *
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import coco
import shutil
import scipy.misc as misc
import argparse


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
annotations_directory = 'annotations'

max_video_idx = 17097

parser = argparse.ArgumentParser(description='Download VLOG dataset and bottle masks.')
parser.add_argument('--gpu_count',
                    default=1,
                    nargs=1,
                    type=int,
                    help='Number of GPUs to use')
parser.add_argument('--images_per_gpu',
                    default=1,
                    nargs=1,
                    type=int,
                    help='Batch size')
parser.add_argument('--n_videos',
                    default=max_video_idx,
                    nargs=1,
                    type=int,
                    help='Number of videos to download')
parser.add_argument('--start_video_id',
                    default=0,
                    nargs=1,
                    type=int,
                    help='The index of the first video to download')
parser.add_argument('--confidence_threshold',
                    default=0.8,
                    nargs=1,
                    type=float,
                    help='Confidence threshold below which mask will be discarded.')
parser.add_argument('--area_threshold',
                    default=0.7,
                    nargs=1,
                    type=float,
                    help='Area threshold above which mask will be discarded.')
args = parser.parse_args()

n_videos = args.n_videos[0] if type(args.n_videos) == list else args.n_videos
start_video_id = args.start_video_id[0] if type(args.start_video_id) == list else args.start_video_id
gpu_count = args.gpu_count[0] if type(args.gpu_count) == list else args.gpu_count
images_per_gpu = args.images_per_gpu[0] if type(args.images_per_gpu) == list else args.images_per_gpu
area_threshold = args.area_threshold[0] if type(args.area_threshold) == list else args.area_threshold
confidence_threshold = args.confidence_threshold[0] if type(args.confidence_threshold) == list else args.confidence_threshold


class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = gpu_count
    IMAGES_PER_GPU = images_per_gpu


if not os.path.exists(raw_videos_directory):
    os.makedirs(raw_videos_directory)
if not os.path.exists(trimmed_videos_directory):
    os.makedirs(trimmed_videos_directory)
if not os.path.exists(annotations_directory):
    os.makedirs(annotations_directory)

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
for entry in entries[start_video_id:min(start_video_id + n_videos, max_video_idx-1)]:
    url, start, stop = entry.split(' ')

    # Download video, trim to specified clip, then delete original video
    video = downloader.download_url(url, raw_videos_directory)
    trimmed = VideoTransformer().trim(video, int(start), int(stop), trimmed_videos_directory)
    os.remove(video.name)
    frames = trimmed.load_frames(fps=video.fps)

    # Create directories for current video frames and annotations
    current_video_frames_dir = os.path.join(trimmed_videos_directory, trimmed.basename)
    current_video_annotations_dir = os.path.join(annotations_directory, trimmed.basename)
    if not os.path.exists(current_video_annotations_dir):
        os.makedirs(current_video_frames_dir)
    if not os.path.exists(current_video_annotations_dir):
        os.makedirs(current_video_annotations_dir)

    for i in range(len(frames)):
        current_frame_annotations_dir = os.path.join(current_video_annotations_dir, str(i))
        if not os.path.exists(current_frame_annotations_dir):
            os.makedirs(current_frame_annotations_dir)
        frame = frames[i]

        # Save frame
        frame_path = os.path.join(current_video_frames_dir, str(i) + '.png')
        misc.imsave(frame_path, frame)

        # Run Mask R-CNN
        result = model.detect([frame])[0]
        classes = result['class_ids']
        scores = result['scores']
        matches = [i for i, class_id in enumerate(classes) if class_id in desired_classes]
        masks = result['masks']

        # Save masks
        mask_idx = 0
        mask_root = os.path.join(current_frame_annotations_dir, str(i) + '_')
        total_area = frame.shape[0] * frame.shape[1]
        for match in matches:
            mask = masks[:, :, match]
            area = np.count_nonzero(mask)

            # Verify that detection is confident and below area threshold
            if scores[match] >= confidence_threshold and area / total_area <= area_threshold:
                misc.imsave(mask_root + str(mask_idx) + '.png', mask * 255)
                mask_idx += 1

    # Delete trimmed video
    os.remove(trimmed.name)
    print('[Finished video ' + trimmed.basename + ']')

# Delete raw videos directory
shutil.rmtree(raw_videos_directory)
