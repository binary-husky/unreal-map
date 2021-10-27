# Created by yingwen at 2019-03-12
import gym
import numpy as np

from malib.policies.explorations.base_exploration import ExplorationBase


class GaussianExploration(ExplorationBase):
    """
    This strategy adds Gaussian noise to the action taken by the deterministic
    policy.
    """

    def __init__(self,
                 action_space,
                 max_sigma=1.0,
                 min_sigma=0.1,
                 decay_period=1000000):
        assert isinstance(action_space, gym.spaces.Box)
        assert len(action_space.shape) == 1
        self._max_sigma = max_sigma
        self._min_sigma = min_sigma
        self._decay_period = decay_period
        self._action_space = action_space

    def get_action(self, t, observation, policy, **kwargs):
        action, agent_info = policy.get_action(observation)
        sigma = self._max_sigma - (self._max_sigma - self._min_sigma) * min(
            1.0, t * 1.0 / self._decay_period)
        return np.clip(
            action + np.random.normal(size=len(action)) * sigma,
            self._action_space.low,
            self._action_space.high)