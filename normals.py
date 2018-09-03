
import cv2
import os
from scipy.misc import imsave
from subprocess import call, DEVNULL


def surface_normals(image):
    image_path = 'temp_image.png'
    normals_path = 'temp_normals.png'

    # Run Taskonomy command on temporarily saved image
    imsave(image_path, image)
    cmd = 'python taskonomy/taskbank/tools/run_img_task.py --task rgb2sfnorm ' \
          '--img ' + image_path + ' --store ' + normals_path
    call(cmd, shell=True, stdout=DEVNULL, stderr=DEVNULL)
    os.remove(image_path)

    # Load results from file and delete file
    sf = cv2.imread(normals_path)
    sf = cv2.cvtColor(sf, cv2.COLOR_BGR2RGB)
    os.remove(normals_path)
    return sf
