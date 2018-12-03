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

from carla_rl import carla_config
from carla_rl.carla_environment_wrapper import CarlaEnvironmentWrapper as CarlaEnv
from carla_settings import get_carla_settings
from processor import MultiInputProcessor
from models import Models

ENV_NAME = "Carla"
config_file = "mysettings.ini"  # file should be placed in CARLA_ROOT folder
settings = get_carla_settings()

env = CarlaEnv(is_render_enabled=False, automatic_render=False, num_speedup_steps=10, run_offscreen=False,
               cameras=["SceneFinal"], save_screens=False, carla_settings=settings, carla_server_settings=config_file)

np.random.seed(123)
nb_actions = 2
window_size = 4
odometry_shape = (7,)

models = Models(image_shape=(carla_config.render_width, carla_config.render_height, 3),
                odometry_shape=odometry_shape, window_length=window_size, nb_actions=nb_actions)

actor = models.build_actor()
critic = models.build_critic()

train_history = None

try:
    processor = MultiInputProcessor(window_length=window_size, nb_inputs=2)
    memory = SequentialMemory(limit=100000, window_length=window_size)

    random_process = OrnsteinUhlenbeckProcess(
        size=nb_actions, theta=.15, mu=0., sigma=.2)

    agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=models.action_input,
                      memory=memory, nb_steps_warmup_critic=216, nb_steps_warmup_actor=216,
                      random_process=random_process, gamma=.99, target_model_update=1e-3, batch_size=16, processor=processor)

    agent.compile([Adam(lr=1e-4), Adam(lr=1e-3)], metrics=['mae'])

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    train_history = agent.fit(env, nb_steps=10**6, visualize=False,
                              verbose=1, action_repetition=4)

    # After training is done, we save the final weights.
    agent.save_weights('ddpg_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

    # Finally, evaluate our algorithm for 5 episodes.
    # agent.test(env, nb_episodes=5, visualize=False, nb_max_episode_steps=200)
except Exception as e:
    tb.print_exc()
    env.close_client_and_server()

finally:
    print(train_history)
