
import numpy as np
from Video import *

raw_videos_directory = 'raw'
trimmed_videos_directory = 'videos'

if not os.path.exists(raw_videos_directory):
    os.makedirs(raw_videos_directory)
if not os.path.exists(trimmed_videos_directory):
    os.makedirs(trimmed_videos_directory)

# Identify video indices where a bottle is present
bottle_idx = 4
labels = np.load('meta/hand_object/hand_object.npy')
indices = np.where(labels[:, bottle_idx] == 1)[0]

# Get links corresponding to those indices
youtube_file = open('meta/youtube_links.txt')
entries = [entry for i, entry in enumerate(youtube_file) if i in indices]
youtube_file.close()

# Download videos and trim to specified clip
n_videos_to_download = 5
downloader = YoutubeDownloader('mp4')
for entry in entries[:n_videos_to_download]:
    url, start, stop = entry.split(' ')

    print('Downloading video...')
    video = downloader.download_url(url, raw_videos_directory)

    print('Trimming...\n')
    trimmed = VideoTransformer().trim(video, int(start), int(stop), trimmed_videos_directory)