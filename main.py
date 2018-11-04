'''
Created Date: Saturday November 3rd 2018
Last Modified: Saturday November 3rd 2018 10:50:05 pm
Author: ankurrc
'''
import argparse
import logging
import os

import tensorflow as tf

from dataset_api import AutoencoderDataset


def main(args):

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    epochs = args.epochs
    batch_size = args.batch
    size = tuple(args.size)
    file_path = os.path.abspath(args.file)
    if not os.path.exists(file_path):
        logger.fatal("File path does not exist: {}".format(file_path))
        raise FileNotFoundError

    filenames = []
    with open(file_path, "r") as ip:
        lines = ip.readlines()
        for line in lines:
            filenames.append(line[:-1])

    print(filenames[:5])

    filenames = tf.constant(filenames[:100])

    dataset = AutoencoderDataset(filenames, size, batch_size, epochs)
    iterator = dataset.iterator
    next_element = iterator.get_next()

    with tf.Session() as sess:

        for _ in range(epochs):
            sess.run(iterator.initializer)
            while True:
                try:
                    sess.run(next_element)
                except tf.errors.OutOfRangeError:
                    break


if __name__ == "__main__":
    parser = argparse.ArgumentParser("A test suite for tf dataset api.")
    parser.add_argument(
        "file", help="Path to file that contains the filenames comprising the dataset")
    parser.add_argument("--epochs", help="No. epochs", type=int, default=10)
    parser.add_argument("--size", help="Image resize (height, width)",
                        nargs=2, type=int, default=(160, 120))
    parser.add_argument("--batch", help="Batch size", type=int, default=32)
    args = parser.parse_args()
    main(args)
