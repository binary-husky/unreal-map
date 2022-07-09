# Created by yingwen at 2019-03-12
"""
ϵ-greedy exploration strategy.
Random exploration according to the value of epsilon.
"""
import numpy as np

from malib.policies.explorations.base_exploration import ExplorationBase
# from garage.misc.overrides import overrides
# TODO: maybe follow this to optimize performance: https://github.com/rlworkgroup/garage/blob/master/garage/misc/overrides.py


class EpsilonGreedyExploration(ExplorationBase):
    """
    ϵ-greedy exploration strategy.
    Select action based on the value of ϵ. ϵ will decrease from
    max_epsilon to min_epsilon within decay_ratio * total_timesteps.
    At state s, with probability
    1 − ϵ: select action = argmax Q(s, a)
    ϵ    : select a random action from an uniform distribution.
    Args:
        env_spec: Environment specification
        total_timesteps: Total steps in the training, equivalent to
            max_path_length * n_epochs.
        max_epsilon: The maximum(starting) value of epsilon.
        min_epsilon: The minimum(terminal) value of epsilon.
        decay_ratio: Fraction of total steps for epsilon decay.
    """

    def __init__(self,
                 action_space,
                 total_timesteps,
                 max_epsilon=1.0,
                 min_epsilon=0.02,
                 decay_ratio=0.1):
        self._max_epsilon = max_epsilon
        self._min_epsilon = min_epsilon
        self._decay_period = int(total_timesteps * decay_ratio)
        self._action_space = action_space
        self._epsilon = self._max_epsilon

    def get_action(self, t, observation, policy, **kwargs):
        """
        Get action from this policy for the input observation.
        Args:
            t: Iteration.
            observation: Observation from the environment.
            policy: Policy network to predict action based on the observation.
        Returns:
            opt_action: optimal action from this policy.
        """
        opt_action = policy.get_action(observation)
        self._decay()
        if np.random.random() < self._epsilon:
            opt_action = self._action_space.sample()

        return opt_action

    def get_actions(self, t, observations, policy, **kwargs):
        """
        Get actions from this policy for the input observations.
        Args:
            t: Iteration.
            observation: Observation from the environment.
            policy: Policy network to predict action based on the observation.
        Returns:
            opt_action: optimal actions from this policy.
        """
        opt_actions = policy.get_actions(observations)
        for itr in range(len(opt_actions)):
            self._decay()
            if np.random.random() < self._epsilon:
                opt_actions[itr] = self._action_space.sample()

        return opt_actions

    def _decay(self):
        if self._epsilon > self._min_epsilon:
            self._epsilon -= (
                self._max_epsilon - self._min_epsilon) / self._decay_period