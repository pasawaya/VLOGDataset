#!/usr/bin/env bash


# Download Mask-RCNN dependencies and build sources
conda install ipython
pip install ninja yacs cython matplotlib
conda install pytorch-nightly -c pytorch

git clone https://github.com/pytorch/vision.git
cd vision
python setup.py install

cd ..
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

cd ../..
git clone https://github.com/facebookresearch/maskrcnn-benchmark.git
cd maskrcnn-benchmark
python setup.py build develop
cd ..


# Download Taskononomy surface normals model
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


# Create directory to contain generative in-painting model
unzip "$CURRDIR/inpaint/model_logs.zip" -d "$CURRDIR/inpaint"



# Install general dependencies
pip install -r requirements.txt
pip install git+https://github.com/JiahuiYu/neuralgym
