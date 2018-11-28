'''
Created Date: Saturday November 3rd 2018
Last Modified: Saturday November 3rd 2018 9:18:59 pm
Author: ankurrc
'''
import tensorflow as tf


class AutoencoderDataset(object):

    def __init__(self, files, size, batch_size, epochs):
        """
        Class Instantiate
            :param files type tensor(tf.string): files that are part of the dataset
            :param size type 2-tuple: (width, height) in pixels
            :param batch_size type int: the batch_size
            :param epochs type int: number of epochs
        """
        self.size = size
        self.dataset = tf.data.Dataset.from_tensor_slices(files)
        self.dataset = self.dataset.map(
            self._parse_function, num_parallel_calls=4)
        self.dataset = self.dataset.batch(batch_size)
        self.dataset = self.dataset.shuffle(1000)
        self.dataset = self.dataset.repeat(epochs)
        self.iterator = self.dataset.make_initializable_iterator()

    def _parse_function(self, filename):
        """
        Function to load image from file and convert to tensor
            :param filename type str: path to image
            :rtype rank-3 tensor
        """
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string)
        image_resized = tf.image.resize_images(image_decoded, self.size)
        image_resized = tf.image.convert_image_dtype(image_resized, tf.float32)
        return image_resized
