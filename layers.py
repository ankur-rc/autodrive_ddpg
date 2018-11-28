'''
Created Date: Sunday November 11th 2018
Last Modified: Sunday November 11th 2018 10:39:47 pm
Author: ankurrc
'''
import tensorflow as tf


def UpSampling2D_NN(stride, **kwargs):
    def layer(x):
        input_shape = tf.keras.backend.int_shape(x)
        output_shape = (stride * input_shape[1], stride * input_shape[2])
        return tf.image.resize_nearest_neighbor(x, output_shape, align_corners=True)
    return tf.keras.layers.Lambda(layer, **kwargs)


def UpSampling2D_Bilinear(stride, **kwargs):
    def layer(x):
        input_shape = tf.keras.backend.int_shape(x)
        output_shape = (stride * input_shape[1], stride * input_shape[2])
        return tf.image.resize_bilinear(x, output_shape, align_corners=True)
    return tf.keras.layers.Lambda(layer, **kwargs)
