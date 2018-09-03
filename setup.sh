##!/usr/bin/env bash

CURRDIR=$(pwd)

TASK="rgb2sfnorm"
mkdir -p "$CURRDIR/taskonomy/taskbank/temp"

SUBFIX="data-00000-of-00001 meta index"

mkdir -p "$CURRDIR/taskonomy/taskbank/temp/$TASK"
for s in $SUBFIX; do
    echo "Downloading ${TASK} model.${s}"
    wget "https://s3-us-west-2.amazonaws.com/taskonomy-unpacked-oregon/\
model_log_final/${TASK}/logs/model.permanent-ckpt.${s}" -P $CURRDIR/taskonomy/taskbank/temp/${TASK}
done
