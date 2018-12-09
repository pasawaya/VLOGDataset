
from normals import surface_normals
from inpaint import generative_inpaint
from detection import MaskRCNN
from data.dataset import DirectoryDataset, VLOGDataset
from utils import resize_pad, safe_makedirs, safe_deldirs

import os
import cv2
import logging
import argparse

from tqdm import tqdm
from scipy.misc import imsave


def main(args):
    w, h = args.w, args.h

    logging.basicConfig(filename='log.log', level=logging.INFO)

    frames_subdir = os.path.join(args.output_dir, 'frames')
    inpainted_subdir = os.path.join(args.output_dir, 'inpainted')
    masks_subdir = os.path.join(args.output_dir, 'masks')
    sf_subdir = os.path.join(args.output_dir, 'surface_normals')

    safe_makedirs([args.output_dir, frames_subdir, inpainted_subdir, masks_subdir, sf_subdir])

    download_dir = 'temp'
    if not args.input_dir:
        dataset = VLOGDataset(fps=args.fps, download_dir=download_dir, start=args.start_video_id, n=args.n)
    else:
        dataset = DirectoryDataset(args.input_dir, fps=args.fps)

    detector = MaskRCNN(args.confidence_threshold, args.area_threshold, classes=args.classes)
    current = 0

    for video_id, frames in dataset:
        if frames:
            logging.info('Processing video ' + str(video_id) + '...')
            with tqdm(total=len(frames)) as t:
                for frame in frames:
                    scores, masks = detector.detect(frame)
                    for score, mask in zip(scores, masks):
                        inpainted, dilated = generative_inpaint(frame, mask, args.inpaint_model_dir, dilate=True)
                        sf = surface_normals(cv2.resize(inpainted, (256, 256)))
                        imsave(os.path.join(inpainted_subdir, str(current) + '.png'), resize_pad(inpainted, (h, w)))
                        imsave(os.path.join(masks_subdir, str(current) + '.png'), resize_pad(mask, (h, w)))
                        imsave(os.path.join(frames_subdir, str(current) + '.png'), resize_pad(frame, (h, w)))
                        imsave(os.path.join(sf_subdir, str(current) + '.png'), resize_pad(sf, (h, w)))
                        # imsave(os.path.join(masks_subdir, str(current) + '_dilated.png'), resize_pad(dilated, (h, w)))
                        logging.info("\tSaved object with score " + str(score))
                        current += 1
                    t.update()
    safe_deldirs(download_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--classes',
                        default=['bottle', 'cup', 'bowl', 'wine glass'],
                        nargs='+',
                        type=str,
                        help='Object classes to detect, remove, and in-paint.')
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
                        default=0.84,
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
    parser.add_argument('--start_video_id',
                        default=0,
                        type=int,
                        help='The index of the first video to download.')
    parser.add_argument('--n',
                        default=None,
                        type=int,
                        help='Number of videos to download.')
    main(parser.parse_args())
