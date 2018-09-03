
from dataset import *
from segmentation import MaskRCNN
from os_utils import *
from normals import *
from inpaint import generative_inpaint

import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm

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

w, h = args.w, args.h

output_dir = args.output_dir
frames_subdir = os.path.join(output_dir, 'frames')
inpainted_subdir = os.path.join(output_dir, 'inpainted')
masks_subdir = os.path.join(output_dir, 'masks')
sf_subdir = os.path.join(output_dir, 'surface_normals')

safe_makedirs([output_dir, frames_subdir, inpainted_subdir, masks_subdir, sf_subdir])

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
                    inpainted = generative_inpaint(frame, mask, args.inpaint_model_dir, dilate=True)
                    sf = surface_normals(cv2.resize(inpainted, (256, 256)))

                    imsave(os.path.join(inpainted_subdir, str(current) + '.png'), cv2.resize(inpainted, (w, h)))
                    imsave(os.path.join(masks_subdir, str(current) + '.png'), cv2.resize(mask, (w, h)))
                    imsave(os.path.join(frames_subdir, str(current) + '.png'), cv2.resize(frame, (w, h)))
                    imsave(os.path.join(sf_subdir, str(current) + '.png'), cv2.resize(sf, (w, h)))

                    current += 1
            t.update()
