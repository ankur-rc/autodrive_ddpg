'''
Created Date: Sunday December 2nd 2018
Last Modified: Sunday December 2nd 2018 6:17:51 pm
Author: ankurrc
'''
import numpy as np

from rl.core import Processor
from carla.image_converter import to_rgb_array


class MultiInputProcessor(Processor):

    def __init__(self, window_length=None, nb_inputs=None):
        self.window_length = window_length
        self.nb_inputs = nb_inputs

    def process_observation(self, observation):
        # convert image to numoy array and normalise
        # observation[1] = to_rgb_array(observation[1])
        # observation[1] /= 255.

        # print("obsv. min and max:", np.min(
        #     observation[1]), np.max(observation[1]))

        return observation

    def process_state_batch(self, batch):
        input_batches = [[] for x in range(self.nb_inputs)]
        for state in batch:
            processed_state = [[] for x in range(self.nb_inputs)]
            for observation in state:
                assert len(observation) == self.nb_inputs
                for o, s in zip(observation, processed_state):
                    s.append(o)
            for idx, s in enumerate(processed_state):
                input_batches[idx].append(s)
        ob = [np.array(x) for x in input_batches]

        return ob
