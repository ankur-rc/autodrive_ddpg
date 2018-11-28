'''
Created Date: Sunday November 11th 2018
Last Modified: Sunday November 11th 2018 1:13:05 pm
Author: ankurrc
'''
import tensorflow as tf
from layers import UpSampling2D_NN

INPUT_DIMS = (160, 120, 3)
LATENT_DIMS = 1024


def build_encoder():

    input = tf.keras.Input(shape=INPUT_DIMS, name="encoder_input")
    x = tf.keras.layers.ZeroPadding2D(
        padding=(0, 4), name="padded_input")(input)

    filters = 32
    for i in range(1, 3):
        for j in range(1, 3):
            x = tf.keras.layers.Conv2D(
                filters=filters, kernel_size=(3, 3), padding="same", activation="relu", name="block{}_conv{}".format(i, j))(x)
        x = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2), strides=(2, 2), name="block{}_pool".format(i))(x)
        filters *= 2

    for i in range(3, 6):
        for j in range(1, 4):
            x = tf.keras.layers.Conv2D(
                filters=filters, kernel_size=(3, 3), padding="same", activation="relu", name="block{}_conv{}".format(i, j))(x)
        x = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2), strides=(2, 2), name="block{}_pool".format(i))(x)
        filters *= 2

    interim_dims = x.shape[1:]

    # fc
    x = tf.keras.layers.Flatten()(x)

    # output
    z_mean = tf.keras.layers.Dense(
        LATENT_DIMS, activation="relu", name="z_mean")(x)
    z_logvar = tf.keras.layers.Dense(
        LATENT_DIMS, activation="relu", name="z_logvar")(x)

    return tf.keras.Model(inputs=input, outputs=[z_mean, z_logvar], name="encoder"), interim_dims


def build_decoder(interim_dims):

    input = tf.keras.layers.Input(shape=(LATENT_DIMS,), name="decoder_input")
    flattened_interim_dims = interim_dims[0]*interim_dims[1]*interim_dims[2]

    # fc
    x = tf.keras.layers.Dense(flattened_interim_dims,
                              activation="relu", name="fc2")(input)

    x = tf.keras.layers.Reshape(interim_dims)(x)

    filters = 512
    for i in range(1, 4):
        x = UpSampling2D_NN(stride=2, name="block{}_upsample".format(i))(x)
        for j in range(1, 4):
            x = tf.keras.layers.Conv2D(
                filters=filters, kernel_size=(3, 3), padding="same", activation="relu", name="block{}_conv{}".format(i, j))(x)
        filters //= 2

    for i in range(4, 6):
        x = UpSampling2D_NN(stride=2, name="block{}_upsample".format(i))(x)
        for j in range(1, 3):
            x = tf.keras.layers.Conv2D(
                filters=filters, kernel_size=(3, 3), padding="same", activation="relu", name="block{}_conv{}".format(i, j))(x)
        filters //= 2

    x = tf.keras.layers.Conv2D(filters=3, kernel_size=(
        3, 3), activation="relu", padding="same", name="padded_output")(x)
    output = tf.keras.layers.Cropping2D(cropping=(0, 4), name="output")(x)

    return tf.keras.Model(inputs=input, outputs=output, name="decoder")


def sample(args):

    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dims = tf.shape(z_mean)[1]
    epsilon = tf.random_normal(shape=(batch, dims))

    return z_mean + tf.exp(z_log_var)*epsilon


def build_vae(batch_size, encoder, decoder):
    x_input = tf.keras.Input(batch_shape=(batch_size,) + INPUT_DIMS)

    z_mean, z_log_var = encoder(x_input)
    z = tf.keras.layers.Lambda(sample)([z_mean, z_log_var])
    _output = decoder(z)

    reconstruction_loss = tf.keras.losses.mse(
        tf.keras.backend.flatten(x_input), tf.keras.backend.flatten(_output))
    reconstruction_loss *= INPUT_DIMS[0]*INPUT_DIMS[1]*INPUT_DIMS[2]

    kl_loss = 1 + z_log_var - \
        tf.keras.backend.square(z_mean) - tf.keras.backend.exp(z_log_var)
    kl_loss = tf.keras.backend.sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    vae_loss = tf.keras.backend.mean(reconstruction_loss + kl_loss)

    vae = tf.keras.Model(inputs=x_input, outputs=_output, name="vae")
    vae.add_loss(vae_loss)

    return vae


def main():
    batch_size = 128

    encoder, dim = build_encoder()
    decoder = build_decoder(dim)

    vae = build_vae(batch_size, encoder, decoder)

    encoder.summary()
    decoder.summary()
    vae.summary()


if __name__ == "__main__":
    main()
