# VLOG Dataset Downloader

1. Downloads clips from the [VLOG Dataset](https://people.eecs.berkeley.edu/~dfouhey/2017/VLOG/) that contain specified objects.
2. Runs Mask R-CNN to segment the specified objects from the frame.
3. Saves the clips and masks.

Example usage:

```
# Download all videos on a single GPU with default confidence and area thresholds
python main.py

# Download all videos on 8 GPUs with batch sizes of 16
python main.py --gpu_count=8 --images_per_gpu=16

# Download the first 100 videos, then download the next 100 at some later point
python main.py -n_videos=100
...
python main.py --start_video_id=100 -n_videos=100

# Download all videos and save only high confidence detections regardless of size
python main.py --confidence_threshold=0.95 --area_threshold=1.0
```
