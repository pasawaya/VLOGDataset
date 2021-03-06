
import os
import cv2
from utils import safe_makedirs, safe_deldirs
from scipy.misc import imsave
from subprocess import call, DEVNULL


def surface_normals(image):
    h, w, _ = image.shape
    image = cv2.resize(image, (256, 256))

    temp_dir = os.path.join(os.getcwd(), 'normals_temp')
    safe_makedirs(temp_dir)
    image_path = os.path.join(temp_dir, 'temp_image.png')
    normals_path = os.path.join(temp_dir, 'temp_normals.png')

    # Run Taskonomy command on temporarily saved image
    imsave(image_path, image)
    cmd = 'python taskonomy/taskbank/tools/run_img_task.py --task rgb2sfnorm ' \
          '--img \"' + image_path + '\" --store \"' + normals_path + '\"'
    call(cmd, shell=True, stdout=DEVNULL, stderr=DEVNULL)

    # Load results from file and clean-up temporary images and folders
    sf = cv2.imread(normals_path)
    sf = cv2.cvtColor(sf, cv2.COLOR_BGR2RGB)
    sf = cv2.resize(sf, (w, h))
    safe_deldirs(temp_dir)
    return sf
