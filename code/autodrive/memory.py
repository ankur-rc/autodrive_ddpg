'''
Created Date: Monday December 3rd 2018
Last Modified: Monday December 3rd 2018 2:36:07 pm
Author: ankurrc
'''
import sys
import os
import math
import random

from collections import deque, namedtuple

import numpy as np

from rl.memory import Memory, zeroed_observation, sample_batch_indexes

Experience = namedtuple(
    'Experience', 'state0, action, reward, state1, terminal1')


class SumTree(object):
    def __init__(self, maxlen=None):
        # buffer size
        self.maxlen = maxlen
        # calculate tree levels
        self.tree_level = math.ceil(math.log(maxlen + 1, 2)) + 1
        # calculate the nodes in the tree
        self.tree_size = 2**self.tree_level - 1
        # allocate memory to the datastructures
        self.tree = [0 for i in range(self.tree_size)]
        self.data = [None for i in range(self.maxlen)]
        # current size of the tree
        self.size = 0
        # pointer to the current location of the tree
        self.cursor = 0

    # add  new experience with 'contents' and new priority with 'value'
    def add(self, contents, value):
        index = self.cursor
        self.cursor = (self.cursor + 1) % self.maxlen
        self.size = min(self.size + 1, self.maxlen)

        # for the experience
        self.data[index] = contents
        # for the priority
        self.val_update(index, value)

    # to get the priority 'value' at index
    def get_val(self, index):
        tree_index = 2**(self.tree_level - 1) - 1 + index
        return self.tree[tree_index]

    # to set the priority 'value' at the 'index'
    def val_update(self, index, value):
        tree_index = 2**(self.tree_level - 1) - 1 + index
        diff = value - self.tree[tree_index]
        self.reconstruct(tree_index, diff)

    # reconstruct the tree after updating it's leaf nodes
    def reconstruct(self, tindex, diff):
        self.tree[tindex] += diff
        if not tindex == 0:
            tindex = int((tindex-1)/2)
            self.reconstruct(tindex, diff)

    def find(self, value, norm=True):
        if norm:
            value *= self.tree[0]
        return self._find(value, 0)

    def _find(self, value, index):
        if 2**(self.tree_level-1)-1 <= index:
            return self.data[index-(2**(self.tree_level-1)-1)], self.tree[index], index-(2**(self.tree_level-1)-1)

        left = self.tree[2*index+1]

        if value <= left:
            return self._find(value, 2*index+1)
        else:
            return self._find(value-left, 2*(index+1))

    def print_tree(self):
        for k in range(1, self.tree_level+1):
            for j in range(2**(k-1)-1, 2**k-1):
                print(self.tree[j], end=' ')
            print()

    def filled_size(self):
        return self.size


class PrioritizedExperience(Memory):
    """ The class represents prioritized experience replay buffer.

    The class has functions: store samples, pick samples with 
    probability in proportion to sample's priority, update 
    each sample's priority, reset alpha.

    see https://arxiv.org/pdf/1511.05952.pdf .

    """

    def __init__(self, memory_size, alpha, beta, **kwargs):
        """ Prioritized experience replay buffer initialization.

        Parameters
        ----------
        memory_size : int
            sample size to be stored
        batch_size : int
            batch size to be selected by `select` method
        alpha: float
            exponent determine how much prioritization.
            Prob_i \sim priority_i**alpha/sum(priority**alpha)
        beta: float
        """
        super(PrioritizedExperience, self).__init__(**kwargs)

        self.tree = SumTree(memory_size)
        self.memory_size = memory_size
        self.alpha = alpha
        self.beta = beta
        self.max_priority = 1.0

    def append(self, observation, action, reward, terminal, training=True):
        """ Add new sample.

        Parameters
        ----------
        observation (dict): Observation returned by environment
        action (int): Action taken to obtain this observation
        reward (float): Reward obtained by taking this action
        terminal (boolean): Is the state terminal

        """
        super(PrioritizedExperience, self).append(observation,
                                                  action, reward, terminal, training=training)

        if training:
            self.tree.add([observation, action, reward,
                           terminal], self.max_priority**self.alpha)

    def sample(self, batch_size, beta=None):
        """Return a randomized batch of experiences

        # Argument
            batch_size (int): Size of the all batch
            beta (float)
        # Returns
            A list of experiences, weights and indices
        """

        assert self.tree.filled_size() >= batch_size + 2, "Not enough entries in memory"

        out = []
        batch_idxs = []
        weights = []
        priorities = []
        for _ in range(batch_size):
            r = random.random()
            data, priority, idx = self.tree.find(r)
            batch_idxs.append(idx)
            out.append(data)
            priorities.append(priority)

        batch_idxs = np.array(batch_idxs) + 1
        assert np.min(batch_idxs) >= self.window_length + 1
        assert np.max(batch_idxs) < self.nb_entries
        assert len(batch_idxs) == batch_size

        # Create experiences
        experiences = []
        for i, idx, priority, data in enumerate(zip(batch_idxs, priorities, out)):
            # [observation, action, reward, terminal]
            # idx - 1 to negate previus step, and terminal flag is in last experience for current observation
            terminal0 = self.tree.data[idx - 2][3]
            while terminal0:
                # Skip this transition because the environment was reset here. Select a new, random
                # transition and use this instead. This may cause the batch to contain the same
                # transition twice.
                r = random.random()
                data, priority, idx = self.tree.find(r)
                terminal0 = self.tree.data[idx - 2][3]
            assert self.window_length + 1 <= idx < self.nb_entries

            # This code is slightly complicated by the fact that subsequent observations might be
            # from different episodes. We ensure that an experience never spans multiple episodes.
            # This is probably not that important in practice but it seems cleaner.
            state0 = [self.tree.data[idx - 1][0]]
            for offset in range(0, self.window_length - 1):
                current_idx = idx - 2 - offset
                assert current_idx >= 1
                current_terminal = self.tree.data[current_idx - 1][3]
                if current_terminal and not self.ignore_episode_boundaries:
                    # The previously handled observation was terminal, don't add the current one.
                    # Otherwise we would leak into a different episode.
                    break
                state0.insert(0, self.tree.data[current_idx][0])
            while len(state0) < self.window_length:
                state0.insert(0, zeroed_observation(state0[0]))
            action = self.tree.data[idx - 1][1]
            reward = self.tree.data[idx - 1][2]
            terminal1 = self.tree.data[idx - 1][3]

            # Okay, now we need to create the follow-up state. This is state0 shifted on timestep
            # to the right. Again, we need to be careful to not include an observation from the next
            # episode if the last state is terminal.
            state1 = [np.copy(x) for x in state0[1:]]
            state1.append(self.tree.data[idx][0])

            # incase these values change during the initial loop where we check for terminal0 to be True
            priorities[i] = priority
            out[i] = data
            batch_idxs[i] = idx

            # add the importance sampling weights
            weights.append((1./self.memory_size/priority) **
                           beta if priority > 1e-16 else 0)

            self.priority_update([idx], [0])  # To avoid duplicating

            assert len(state0) == self.window_length
            assert len(state1) == len(state0)
            experiences.append(Experience(state0=state0, action=action, reward=reward,
                                          state1=state1, terminal1=terminal1))

        self.priority_update(batch_idxs, priorities)  # Revert priorities

        weights /= max(weights)  # Normalize for stability

        assert len(experiences) == batch_size
        return experiences, weights, batch_idxs

    def priority_update(self, indices, priorities):
        """ The methods update samples's priority.

        Parameters
        ----------
        indices : 
            list of sample indices
        """
        for i, p in zip(indices, priorities):
            self.tree.val_update(i, p**self.alpha)

    def reset_alpha(self, alpha):
        """ Reset a exponent alpha.

        Parameters
        ----------
        alpha : float
        """
        self.alpha, old_alpha = alpha, self.alpha
        priorities = [self.tree.get_val(
            i)**-old_alpha for i in range(self.tree.filled_size())]
        self.priority_update(range(self.tree.filled_size()), priorities)

    @property
    def nb_entries(self):
        """Return number of observations

        # Returns
            Number of observations
        """
        return self.tree.filled_size()
