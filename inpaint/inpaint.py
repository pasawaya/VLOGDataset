
import cv2
import numpy as np
import tensorflow as tf
from inpaint.inpaint_model import InpaintCAModel


def generative_inpaint(image, mask, checkpoint):
    model = InpaintCAModel()
    h, w, _ = image.shape
    grid = 8

    mask = np.expand_dims(mask, 2)
    mask = np.repeat(mask, 3, axis=2)
    image = image[:h // grid * grid, :w // grid * grid, :]
    mask = mask[:h // grid * grid, :w // grid * grid, :]

    image = np.expand_dims(image, 0)
    mask = np.expand_dims(mask, 0)
    input_image = np.concatenate([image, mask], axis=2)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    tf.reset_default_graph()
    with tf.Session(config=sess_config) as sess:
        input_image = tf.constant(input_image, dtype=tf.float32)
        output = model.build_server_graph(input_image)
        output = (output + 1.) * 127.5
        output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)

        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable(checkpoint, from_name)
            assign_ops.append(tf.assign(var, var_value))
        sess.run(assign_ops)
        result = sess.run(output)
        return result[0][:, :, ::-1]


def telea_inpaint(image, mask):
    return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
