# VLOG Dataset Downloader

-----------------

## Features
1. Downloads clips from the [VLOG Dataset](https://people.eecs.berkeley.edu/~dfouhey/2017/VLOG/) that contain specified objects.
2. Runs [Mask R-CNN](https://github.com/matterport/Mask_RCNN) to segment the specified objects from the frame.
3. Removes the objects from the frame and [in-paints](https://github.com/JiahuiYu/generative_inpainting) the removed region.
3. Saves the original frames, in-painted frames, and object masks.


## Installation
1. Clone the repository
   ```bash
   git clone https://github.com/pasawaya/VLOGDataset.git
   ```

2. Install the dependencies
   ```bash
   pip install -r requirements.txt
   ```
3. Run the setup script from the root directory
    ```bash
    sh setup.sh
    ``` 
4. Download the in-painting [model](https://drive.google.com/drive/folders/1M3AFy7x9DqXaI-fINSynW7FJSXYROfv-) from and place it in ``inpaint/model_logs`` 



## Usage

Download and process VLOG dataset at 4 fps and resize results to 320 x 240: 
```
python vlog.py --w=320 --h=240 --fps=4
```

Download and process VLOG dataset and only save high confidence detections regardless of object size:

```
python vlog.py --confidence_threshold=0.95 --area_threshold=1.0
```

Process videos in local directory:

```
python vlog.py --input_dir=/path/to/videos
```


## Examples 