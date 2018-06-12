
import numpy as np
from Video import *
from mrcnn import utils, coco
import mrcnn.model as modellib
from InferenceConfig import InferenceConfig

raw_videos_directory = 'temp'
trimmed_videos_directory = 'videos'

if not os.path.exists(raw_videos_directory):
    os.makedirs(raw_videos_directory)
if not os.path.exists(trimmed_videos_directory):
    os.makedirs(trimmed_videos_directory)

# Initialize Mask R-CNN
model_path = 'mrcnn/mask_rcnn_coco.h5'
if not os.path.exists(model_path):
    utils.download_trained_weights(model_path)

config = InferenceConfig()
config.display()

# Identify video indices where a bottle is present
bottle_idx = 4
labels = np.load('meta/hand_object/hand_object.npy')
indices = np.where(labels[:, bottle_idx] == 1)[0]

# Get links corresponding to those indices
youtube_file = open('meta/youtube_links.txt')
entries = [entry for i, entry in enumerate(youtube_file) if i in indices]
youtube_file.close()

# Download videos and trim to specified clip
n_videos_to_download = 2
downloader = YoutubeDownloader('mp4')
for entry in entries[:n_videos_to_download]:
    url, start, stop = entry.split(' ')

    print('Downloading video...')
    video = downloader.download_url(url, raw_videos_directory)

    print('Trimming...\n')
    trimmed = VideoTransformer().trim(video, int(start), int(stop), trimmed_videos_directory)

    print('Running Mask R-CNN...')
    frames = trimmed.load_frames()


# Delete directory containing full videos
