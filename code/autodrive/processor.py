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
        observation[1] = to_rgb_array(observation[1])
        observation[1] /= 255.

        # print("obsv. min and max:", np.min(
        #     observation[1]), np.max(observation[1]))

        return observation

    def process_state_batch(self, batch):
        # creates [[], []]; after 1:- [[o_0_0, o_1_0], [o_0_1, o_1_1]]; after 2:- [[o_0_0, o_1_0, o_2_0], [o_0_1, o_1_1, o_2_1]]...
        input_batches = [[] for x in range(self.nb_inputs)]

        # [1, 4, 2] -> [4, 2]
        for state in batch:
            # creates [[], []]
            processed_state = [[] for x in range(self.nb_inputs)]
            # [4, 2] -> [2]
            for observation in state:
                assert len(observation) == self.nb_inputs
                # [(o_t_0, []), (o_t_1, [])]
                for o, s in zip(observation, processed_state):
                    # [(o_t_0, [o_t_0]), (o_t_1, [o_t_1])]
                    s.append(o)

            for idx, s in enumerate(processed_state):
                input_batches[idx].append(s)

        # ob = [ np.array([o_0_0, o_1_0, o_2_0]), np.array([o_0_1, o_1_1, o_2_1]) ]
        ob = [np.array(x) for x in input_batches]

        return ob
