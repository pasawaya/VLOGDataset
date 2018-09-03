
from scipy.misc import imsave

from dataset import *
from segmentation import MaskRCNN
from tqdm import tqdm
from os_utils import *
from inpaint import generative_inpaint
import skimage

import numpy as np
import cv2
import os
import argparse
from subprocess import call, DEVNULL

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir',
                    default=None,
                    type=str,
                    help='Directory containing input videos.')
parser.add_argument('--output_dir',
                    default='output',
                    type=str,
                    help='Directory to contain output.')
parser.add_argument('--fps',
                    default=None,
                    type=int,
                    help='Number of frames per second to extract from each vlog.')
parser.add_argument('--w',
                    default=256,
                    type=int,
                    help='Output width.')
parser.add_argument('--h',
                    default=256,
                    type=int,
                    help='Output height.')
parser.add_argument('--confidence_threshold',
                    default=0.8,
                    type=float,
                    help='Confidence threshold below which mask will be discarded.')
parser.add_argument('--area_threshold',
                    default=0.07,
                    type=float,
                    help='Area threshold above which mask will be discarded.')
parser.add_argument('--inpaint_model_dir',
                    default='inpaint/model_logs',
                    type=str,
                    help='Directory containing generative in-painting model.')
args = parser.parse_args()

output_dir = args.output_dir
frames_subdir = os.path.join(output_dir, 'frames')
inpainted_subdir = os.path.join(output_dir, 'inpainted')
masks_subdir = os.path.join(output_dir, 'masks')
sf_subdir = os.path.join(output_dir, 'surface_normals')

safe_makedirs(output_dir)
safe_makedirs(frames_subdir)
safe_makedirs(inpainted_subdir)
safe_makedirs(masks_subdir)
safe_makedirs(sf_subdir)

# dataset = VLOGDataset(fps=args.fps)
dataset = DirectoryDataset(args.input_dir, fps=args.fps)
detector = MaskRCNN(classes=['bottle', 'cup', 'bowl', 'wine glass'])

current = 0

for video_id in range(len(dataset)):
    frames = dataset[video_id]

    print('\nProcessing video ' + str(video_id) + '...')
    with tqdm(total=len(frames)) as t:
        for frame in frames:
            scores, masks = detector.detect(frame)
            total_area = frame.shape[0] * frame.shape[1]

            for score, mask in zip(scores, masks):
                area = np.count_nonzero(mask)
                area_ratio = area / total_area

                if score >= args.confidence_threshold and area_ratio <= args.area_threshold:
                    inpainted_path = os.path.join(inpainted_subdir, str(current) + '.png')
                    mask_path = os.path.join(masks_subdir, str(current) + '.png')
                    frame_path = os.path.join(frames_subdir, str(current) + '.png')
                    sf_path = os.path.join(sf_subdir, str(current) + '.png')

                    dilated_mask = cv2.dilate(mask, np.ones((9, 9)), iterations=2)
                    inpainted = generative_inpaint(frame, dilated_mask, args.inpaint_model_dir)

                    inpainted = cv2.resize(inpainted, (256, 256))
                    original = cv2.resize(frame, (256, 256))
                    mask = cv2.resize(mask, (256, 256))

                    imsave(inpainted_path, inpainted)
                    sf_cmd = 'python taskonomy/taskbank/tools/run_img_task.py --task rgb2sfnorm ' \
                             '--img ' + inpainted_path + ' --store ' + sf_path

                    call(sf_cmd, shell=True, stdout=DEVNULL, stderr=DEVNULL)
                    os.remove(inpainted_path)

                    sf = cv2.imread(sf_path)
                    sf = cv2.cvtColor(sf, cv2.COLOR_BGR2RGB)
                    os.remove(sf_path)

                    inpainted = cv2.resize(inpainted, (args.w, args.h))
                    mask = cv2.resize(mask, (args.w, args.h))
                    original = cv2.resize(original, (args.w, args.h))
                    sf = cv2.resize(sf, (args.w, args.h))

                    imsave(inpainted_path, inpainted)
                    imsave(mask_path, mask)
                    imsave(frame_path, original)
                    imsave(sf_path, sf)

                    current += 1
            t.update()
