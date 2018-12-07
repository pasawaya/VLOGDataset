
import numpy as np
from skimage.transform import resize


def resize_pad(image, new_shape, fill=0):
    """
    Resizes an image while maintaining aspect ratio, appropriately padding borders

    :param image: image to resize (either 2D grayscale image or 3D RGB image)
    :param new_shape: (w, h) of output image
    :param fill: value to pad edges with
    :return: output image of size (w, h) and same aspect ratio as input image
    """
    new_h, new_w = new_shape

    if len(image.shape) == 2:
        h, w = image.shape
    else:
        h, w, c = image.shape

    f_xy = min(new_w / w, new_h / h)
    h, w = int(h * f_xy), int(w * f_xy)
    scaled = resize(image, (h, w), preserve_range=True, anti_aliasing=True, mode='constant').astype(np.uint8)

    if len(scaled.shape) == 3:
        scaled_h, scaled_w, _ = scaled.shape
    else:
        scaled_h, scaled_w = scaled.shape
    dw, dh = new_w - scaled_w, new_h - scaled_h
    x, y = int(np.floor(dw / 2)), int(np.floor(dh / 2))

    if len(image.shape) == 3:
        output = np.ones((new_h, new_w, c)) * fill
        output[y:y + scaled_h, x:x + scaled_w, :] = scaled
    else:
        output = np.ones((new_h, new_w)) * fill
        output[y:y + scaled_h, x:x + scaled_w] = scaled

    return output

