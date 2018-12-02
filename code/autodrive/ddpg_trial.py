'''
Created Date: Saturday December 1st 2018
Last Modified: Saturday December 1st 2018 9:52:19 pm
Author: ankurrc
'''
import numpy as np
import traceback as tb

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

from rl.processors import WhiteningNormalizerProcessor
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

from carla_rl.carla_environment_wrapper import CarlaEnvironmentWrapper as CarlaEnv
from carla_settings import get_carla_settings

ENV_NAME = "Carla"
config_file = "/media/ankurrc/new_volume/689_ece_rl/project/code/autodrive/mysettings.ini"
settings = get_carla_settings()

env = CarlaEnv(is_render_enabled=False, automatic_render=False, num_speedup_steps=10, run_offscreen=False,
               cameras=["SceneFinal"], save_screens=False, carla_settings=settings, carla_server_settings=config_file)

np.random.seed(123)
observation_shape = (4,)
nb_actions = 2

# Next, we build a very simple model.
actor = Sequential()
actor.add(Flatten(input_shape=(4,) + observation_shape))
actor.add(Dense(400))
actor.add(Activation('relu'))
actor.add(Dense(300))
actor.add(Activation('relu'))
actor.add(Dense(nb_actions))
actor.add(Activation('tanh'))
print(actor.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(
    shape=(1,) + observation_shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = Dense(400)(flattened_observation)
x = Activation('relu')(x)
x = Concatenate()([x, action_input])
x = Dense(300)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
print(critic.summary())


try:
    memory = SequentialMemory(limit=100000, window_length=4)

    random_process = OrnsteinUhlenbeckProcess(
        size=nb_actions, theta=.15, mu=0., sigma=.2)

    agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                      memory=memory, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000,
                      random_process=random_process, gamma=.99, target_model_update=1e-3, batch_size=16)

    agent.compile([Adam(lr=1e-4), Adam(lr=1e-3)], metrics=['mae'])

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    agent.fit(env, nb_steps=10**6, visualize=False,
              verbose=1, action_repetition=4)

    # After training is done, we save the final weights.
    agent.save_weights('ddpg_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

    # Finally, evaluate our algorithm for 5 episodes.
    agent.test(env, nb_episodes=5, visualize=False, nb_max_episode_steps=200)
except Exception as e:
    tb.print_exc()
    env.close_client_and_server()
