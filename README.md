# VLOG Dataset Downloader

-----------------

[![Demo](demo.gif)](https://www.youtube.com/watch?v=MbnnfXC0tBE&feature=youtu.be)

## Features
1. Downloads clips from the [VLOG Dataset](https://people.eecs.berkeley.edu/~dfouhey/2017/VLOG/) containing desired object classes.
2. [Segments](https://github.com/matterport/Mask_RCNN) and removes the objects from the frame.
3. [In-paints](https://github.com/JiahuiYu/generative_inpainting) the removed region.
4. Computes [surface normals](https://github.com/StanfordVL/taskonomy/tree/master/taskbank) for the new frame without the objects.


## Installation

1. (Highly recommended) Create a fresh Anaconda environment
   ```bash
   conda create -n vlog python=3.6 anaconda
   source activate vlog
   ```

2. Clone the repository
   ```bash
   git clone --recursive https://github.com/pasawaya/VLOGDataset.git
   ```
   
3. Run the setup script
    ```bash
    cd VLOGDataset/
    sh setup.sh
    ``` 

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
