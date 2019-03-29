'''
Created Date: Sunday December 2nd 2018
Last Modified: Sunday December 2nd 2018 10:13:25 am
Author: ankurrc
'''

from keras.layers import TimeDistributed, Conv2D, LSTM, Input, \
                        BatchNormalization, MaxPooling2D, \
                        Flatten, Dense, Concatenate, GRU, AveragePooling2D, CuDNNGRU
from keras.initializers import RandomUniform
from keras.models import Model
from keras.utils import plot_model
from keras.regularizers import l2


class Models(object):

    def __init__(self, image_shape=None, odometry_shape=None, window_length=None, nb_actions=None):
        self.window_length = window_length
        self.odometry_shape = odometry_shape
        self.image_shape = image_shape
        self.nb_actions = nb_actions

        self.ih_img, self.ih_odo, self.ih_out = self._build_inputhead()
        self.actor = None
        self.critic = None

    def _build_inputhead(self):

        layer_prefix = "ih"
        img_ip = Input(shape=(self.window_length,) +
                       self.image_shape, name="{}/image_in".format(layer_prefix))
        odo_ip = Input(shape=(self.window_length,) +
                       self.odometry_shape, name="{}/odometry_in".format(layer_prefix))

        x = TimeDistributed(Conv2D(filters=16, kernel_size=(
            5, 5), padding="same", strides=2, activation="relu", name="{}/conv1".format(layer_prefix)))(img_ip)
        x = TimeDistributed(MaxPooling2D(pool_size=2, name="{}/maxpool1".format(layer_prefix)))(x)
        s1x1 = TimeDistributed(Conv2D(filters=16, kernel_size=(
            1, 1), padding="same", strides=1, activation="relu", name="{}/fire/s1x1".format(layer_prefix)))(x)
        e1x1 = TimeDistributed(Conv2D(filters=32, kernel_size=(
            1, 1), padding="same", strides=1, activation="relu", name="{}/fire/e1x1".format(layer_prefix)))(s1x1)
        e3x3 = TimeDistributed(Conv2D(filters=32, kernel_size=(
            3, 3), padding="same", strides=1, activation="relu", name="{}/fire/e3x3".format(layer_prefix)))(s1x1)
        x = Concatenate(axis=-1,  name="{}/fire/concat".format(layer_prefix))([e1x1, e3x3])
        # x = TimeDistributed(Concatenate(axis=-1,  name="{}/fire/concat".format(layer_prefix)))([e1x1, e3x3])
        x = TimeDistributed(AveragePooling2D(pool_size=3,  name="{}/avgpool1".format(layer_prefix)))(x)
        x = TimeDistributed(Flatten( name="{}/flatten".format(layer_prefix)))(x)
        # x = LSTM(200, recurrent_dropout=0.2, dropout=0.2)(x)
        x = CuDNNGRU(32,  name="{}/img/gru1".format(layer_prefix))(x)
        x = BatchNormalization()(x)

        # y = TimeDistributed(Dense(16, activation="relu",  name="{}/fc1".format(layer_prefix)))(odo_ip)
        # y = TimeDistributed(BatchNormalization())(y)
        # y = LSTM(16, recurrent_dropout=0.2, dropout=0.2)(y)
        y = CuDNNGRU(4,  name="{}/odom/gru1".format(layer_prefix))(odo_ip)
        y = BatchNormalization()(y)

        op = Concatenate( name="{}/out".format(layer_prefix))([x, y])
        # op = BatchNormalization(name="{}/out".format(layer_prefix))(op)

        return img_ip, odo_ip, op

    def build_actor(self):

        layer_prefix = "actor"

        x = Dense(64, activation="relu", name="{}/dense_1".format(
            layer_prefix))(self.ih_out)
        x = Dense(128, activation="relu",
                  name="{}/dense_2".format(layer_prefix))(x)
        out = Dense(self.nb_actions, activation="tanh",
                    kernel_initializer=RandomUniform(minval=-3e-4, maxval=3e-4), name="{}/out".format(layer_prefix))(x)

        self.actor = Model(
            inputs=[self.ih_odo, self.ih_img], outputs=out, name="actor")
        print(self.actor.summary())
        plot_model(self.actor, to_file="imgs/actor.png", show_shapes=True)

        return self.actor

    def build_critic(self):

        layer_prefix = "critic"

        self.action_input = action_input = Input(shape=(self.nb_actions,),
                                                 name="{}/action_inp".format(layer_prefix))
        x = Concatenate(name="{}/inp".format(layer_prefix))(
            [self.ih_out, action_input])
        x = BatchNormalization()(x)
        x = Dense(64, activation="relu", name="{}/dense_1".format(
            layer_prefix))(x)
        x = Dense(128, activation="relu",
                  name="{}/dense_2".format(layer_prefix))(x)
        out = Dense(1, activation="linear", kernel_initializer=RandomUniform(
            minval=-3e-4, maxval=3e-4), name="{}/out".format(layer_prefix), kernel_regularizer=l2(l=0.01))(x)

        self.critic = Model(
            inputs=[self.ih_odo, self.ih_img, action_input], outputs=out, name="critic")
        print(self.critic.summary())
        plot_model(self.critic, to_file="imgs/critic.png", show_shapes=True)

        return self.critic


def main():
    models = Models(image_shape=(84, 84, 3), odometry_shape=(
        4,), window_length=4, nb_actions=2)
    actor = models.build_actor()
    critic = models.build_critic()

    model = Model(inputs=[models.ih_odo, models.ih_img, models.action_input],
                  outputs=actor.outputs + critic.outputs)
    plot_model(model, to_file="imgs/model.png", show_shapes=True)
    print(model.summary())


if __name__ == "__main__":
    main()
