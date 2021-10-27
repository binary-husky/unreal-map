from malib.spaces import MASpace
import numpy as np
from malib.core import Serializable


class EnvSpec:
    def __init__(
            self,
            observation_space,
            action_space):
        """
        :type observation_space: Space
        :type action_space: Space
        """
        self._observation_space = observation_space
        self._action_space = action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space


class MAEnvSpec(Serializable):
    def __init__(
            self,
            observation_spaces,
            action_spaces):
        """
        :type observation_spaces: MASpace
        :type action_spaces: MASpace
        """
        self._Serializable__initialize(locals())
        assert isinstance(observation_spaces, MASpace)
        assert isinstance(action_spaces, MASpace)
        self.agent_num = observation_spaces.agent_num
        self._observation_spaces = observation_spaces
        self._action_spaces = action_spaces
        self._env_specs = np.array(EnvSpec(observation_space, action_space) for observation_space, action_space in zip(observation_spaces, action_spaces))

    @property
    def observation_space(self):
        return self._observation_spaces

    @property
    def action_space(self):
        return self._action_spaces

    def __getitem__(self, i):
        assert (i >= 0) and (i < self.agent_num)
        return self._env_specs[i]