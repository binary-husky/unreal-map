# Created by yingwen at 2019-03-12

import numpy as np
from malib.policies.explorations.base_exploration import ExplorationBase


class RandomExploration(ExplorationBase):
    def __init__(self,action_space):
        self._action_space = action_space

    def get_action(self, t, observation, policy, **kwargs):
        return self._action_space.sample()

    def get_actions(self, t, observations, policy, **kwargs):
        return np.array([self._action_space.sample() for _ in range(len(observations))])
