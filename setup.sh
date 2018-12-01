##!/usr/bin/env bash

# 1. Download Taskononomy surface normals model
CURRDIR=$(pwd)

TASK="rgb2sfnorm"
mkdir -p "$CURRDIR/taskonomy/taskbank/temp"

SUBFIX="data-00000-of-00001 meta index"

mkdir -p "$CURRDIR/taskonomy/taskbank/temp/$TASK"
for s in $SUBFIX; do
    echo "Downloading ${TASK} model.${s}"
    wget "https://s3-us-west-2.amazonaws.com/taskonomy-unpacked-oregon/\
model_log_final/${TASK}/logs/model.permanent-ckpt.${s}" -P "$CURRDIR/taskonomy/taskbank/temp/${TASK}"
done

# 2. Create directory to contain generative in-painting model
mkdir "$CURRDIR/inpaint/model_logs"

# 3. Download Mask R-CNN model
wget "https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5" -P "$CURRDIR/segmentation/mrcnn"