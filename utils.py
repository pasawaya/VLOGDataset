
import numpy as np
from skimage.transform import resize


def resize_pad(image, new_shape, fill=0):
    h, w, c = image.shape
    new_w, new_h = new_shape

    scaled = scale(image, min(new_w / w, new_h / h))
    output = np.ones((new_h, new_w, c)) * fill

    scaled_h, scaled_w, _ = scaled.shape
    dw, dh = new_w - scaled_w, new_h - scaled_h
    x, y = np.floor(dw / 2), np.floor(dh / 2)

    output[x:x+scaled_h, y:y+scaled_w, :] = scaled
    return output


def scale(image, f_xy):
    h, w, _ = image.shape
    h, w = int(h * f_xy), int(w * f_xy)
    image = resize(image, (h, w), preserve_range=True, anti_aliasing=True, mode='constant').astype(np.uint8)
    return image
