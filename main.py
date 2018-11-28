'''
Created Date: Saturday November 3rd 2018
Last Modified: Saturday November 3rd 2018 10:50:05 pm
Author: ankurrc
'''
import argparse
import logging
import os
import random
import time

import tensorflow as tf

from dataset_api import AutoencoderDataset
from vgg_vae import build_encoder, build_decoder, build_vae


def main(args):

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    epochs = args.epochs
    batch_size = args.batch
    size = tuple(args.size)
    learning_rate = args.lr
    training_samples = args.samples[0]
    val_samples = args.samples[1]

    file_path = os.path.abspath(args.file)
    if not os.path.exists(file_path):
        logger.fatal("File path does not exist: {}".format(file_path))
        raise FileNotFoundError

    filenames = []
    with open(file_path, "r") as ip:
        lines = ip.readlines()
        for line in lines:
            filenames.append(line[:-1])

    random.shuffle(filenames)

    train_files = tf.constant(filenames[:training_samples])
    val_files = tf.constant(
        filenames[training_samples:(training_samples+val_samples)])

    train_dataset = AutoencoderDataset(train_files, size, batch_size, epochs)
    train_it = train_dataset.get_iterator()

    val_dataset = AutoencoderDataset(val_files, size, batch_size, epochs)
    val_it = val_dataset.get_iterator()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=50, monitor='val_loss', min_delta=0.1),
        tf.keras.callbacks.TensorBoard(
            log_dir='./logs', write_graph=True, write_grads=True, write_images=True),
        tf.keras.callbacks.ModelCheckpoint("weights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', verbose=0,
                                           save_best_only=False, save_weights_only=False, mode='auto', period=5),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=25, min_lr=0.001 * learning_rate)
    ]

    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

    # build model
    encoder, dim = build_encoder()
    decoder = build_decoder(dim)
    vae = build_vae(batch_size, encoder, decoder)

    encoder.summary()
    decoder.summary()
    vae.summary()

    vae.compile(optimizer=optimizer)

    # train
    start = time.time()

    vae.fit(train_it,
            steps_per_epoch=(training_samples//batch_size),
            epochs=epochs,
            callbacks=callbacks,
            # validation_data=val_it,
            # validation_steps=(val_samples//batch_size),
            verbose=2)

    done = time.time()
    elapsed = done - start
    print("Elapsed: ", elapsed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("A test suite for tf dataset api.")
    parser.add_argument(
        "file", help="Path to file that contains the filenames comprising the dataset")
    parser.add_argument("--epochs", help="No. epochs", type=int, default=10)
    parser.add_argument("--size", help="Image resize (height, width)",
                        nargs=2, type=int, default=(160, 120))
    parser.add_argument("--batch", help="Batch size", type=int, default=32)
    parser.add_argument("--lr", help="learning_rate",
                        type=float, default=0.001)
    parser.add_argument(
        "--samples", help="(Train, Validation samples)", nargs=2, type=int, default=(2048, 256))

    args = parser.parse_args()
    main(args)
