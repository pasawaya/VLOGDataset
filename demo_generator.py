
import argparse
import os
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',
                    default=None,
                    type=str,
                    help='Directory containing data.')
parser.add_argument('--output_name',
                    default='demo.avi',
                    type=str,
                    help='Name of output video.')
parser.add_argument('--w',
                    default=256,
                    type=int,
                    help='Input image width.')
parser.add_argument('--h',
                    default=256,
                    type=int,
                    help='Input image height.')
args = parser.parse_args()

frames_path = os.path.join(args.data_dir, 'frames')
inpainted_path = os.path.join(args.data_dir, 'inpainted')
masks_path = os.path.join(args.data_dir, 'masks')
sf_path = os.path.join(args.data_dir, 'surface_normals')

n_images = len(os.listdir(frames_path))
image_names = [str(i) + '.png' for i in range(n_images)]

out = cv2.VideoWriter(args.output_name, cv2.VideoWriter_fourcc(*'MJPG'), 8.0, (args.w * 4, args.h))

for name in image_names:
    out.write(np.hstack([cv2.imread(os.path.join(frames_path, name)),
                         cv2.imread(os.path.join(inpainted_path, name)),
                         cv2.imread(os.path.join(masks_path, name)),
                         cv2.imread(os.path.join(sf_path, name))]))

out.release()
