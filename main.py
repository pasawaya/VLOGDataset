
import numpy as np
from Video import *
from mrcnn import utils, coco
import mrcnn.model as modellib
from mrcnn.InferenceConfig import InferenceConfig
from mrcnn import visualize
import shutil


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

display_masks = False

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
print('Loading weights...')
if not os.path.exists(weights_directory):
    utils.download_trained_weights(weights_directory)

model = modellib.MaskRCNN(mode="inference", model_dir=model_directory, config=InferenceConfig())
model.load_weights(weights_directory, by_name=True)

# Get indices of desired objects to mask
desired_classes = ['bottle', 'cup', 'bowl']
desired_classes = [class_names.index(class_name) for class_name in desired_classes]

# Download videos and trim to specified clip
n_videos_to_download = 1
downloader = YoutubeDownloader('mp4')
for entry in entries[:n_videos_to_download]:
    url, start, stop = entry.split(' ')

    print('Downloading video...')
    video = downloader.download_url(url, raw_videos_directory)

    print('Trimming...')
    trimmed = VideoTransformer().trim(video, int(start), int(stop), trimmed_videos_directory)
    os.remove(video.name)

    print('Loading frames...')
    frames = trimmed.load_frames()

    print('Running Mask R-CNN...')
    annotations = []
    for frame in frames:
        result = model.detect([frame])[0]
        classes = result['class_ids']
        matches = [i for i, class_id in enumerate(classes) if class_id in desired_classes]

        # Fetch annotations
        rois = result['rois']
        masks = result['masks']
        class_ids = result['class_ids']

        # Record frame annotations
        annotation = {'class_ids': class_ids[matches],
                      'masks': masks[:, :, matches]}
        annotations.append(annotation)

        if display_masks:
            # Display all instances
            visualize.display_instances(frame,
                                        result['rois'],
                                        result['masks'],
                                        result['class_ids'],
                                        class_names)

            # Display only matching instances
            visualize.display_instances(frame,
                                        result['rois'][matches, :],
                                        result['masks'][:, :, matches],
                                        result['class_ids'][matches],
                                        class_names)

    # Save video annotations
    annotations_name = os.path.join(annotations_directory, trimmed.basename + ".npy")
    np.save(annotations_name, annotations)

# Delete raw videos directory
shutil.rmtree(raw_videos_directory)

# Annotation loading example
annotations_name = os.path.join(annotations_directory, "0.npy")
annotations = np.load(annotations_name)
for i in range(annotations.shape[0]):
    frame_annotations = annotations[i]
    print('Frame ' + str(i))
    print('\tclass_ids: ' + str(frame_annotations['class_ids']))
    print('\tmasks: ' + str(frame_annotations['masks']) + "\n")
