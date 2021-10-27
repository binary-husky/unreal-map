import numpy as np

from malib.environments.base_game import BaseGame
from malib.spaces import Box, MASpace, MAEnvSpec

# Boat state = {x, y, theta, v, w}
# x, y = coords [0, 50], [0, 100]
# theta = boat's angle [-60, 60] degrees
# v = speed, w = angle speed [2, 5], [-1, 1]

# actions a_v = forward acceleration [-1, 2], a_w = angular acceleration [-1, 1]


UPDATE_INTERVAL = 10


class SingleBoatGame(BaseGame):
    def __init__(self):
        self.agent_num = 1
        self.x = 0
        self.y = 50
        self.theta = 0
        self.v = 2
        self.w = 0
        self.stoch = 0
        self.t = 0

        obs_lows = np.array([0, 0, -1. * np.pi / 3., 2., -1.])
        obs_highs = np.array([50, 100, np.pi / 3., 5., 1.])
        self.observation_spaces = MASpace(tuple([Box(low=obs_lows, high=obs_highs)]))
        self.action_spaces = MASpace(tuple([Box(low=-1., high=1., shape=(2,))]))

        self.env_specs = MAEnvSpec(self.observation_spaces, self.action_spaces)
        self.avs = [0] * 10
        self.aws = [0] * 10

        self.rewards = list()  # a list that records rewards obtained in current episode, in sequential order.

    @property
    def observation_space(self):
        return self.observation_spaces[0]

    @property
    def action_space(self):
        return self.action_spaces[0]

    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    def clip(self, v, range):
        return np.clip(v, *range)

    def get_stoch(self):
        """
        Get rid of stochasticity.
        """
        return 4 * (self.x / 50 - (self.x / 50) ** 2)

    # return np.random.normal(4, 1) * (self.x / 50 - (self.x / 50) ** 2)

    def calculate_reward(self):
        x, y = int(self.x), int(self.y)
        if x != 50:
            return 0
        if 25 < y < 35:
            return 15 - 3 * abs(y - 30)
        if 40 < y < 60:
            return 0
        # return 10 - abs(y - 50)
        if 60 < y <= 100:
            return 0
        # return 10 - abs(y - 80) / 2
        return 0

    def step(self, action):  # pylint: disable=E0202
        self.t += 1
        av, aw = action
        av = float(av)
        # av (-1, 2), aw (-1, 1)
        av = av * 1.5 + 0.5  # rescale action to make it (-1, 1) from outside. Convenient for Gaussian policy Squash.
        aw = float(aw)
        self.stoch = self.get_stoch()
        self.x = self.find_nearest(list(range(51)), self.x + self.v * np.cos(self.theta))
        self.y = self.find_nearest(list(range(101)), self.y + self.v * np.sin(self.theta) + self.stoch)
        self.theta = self.find_nearest([(-np.pi / 3 + i * 2 * np.pi / 30) for i in range(11)], self.theta + self.w)
        self.v = self.find_nearest([(2 + i * 3 / 10) for i in range(11)], self.v + av)
        self.w = self.find_nearest([(-1 + i * 2 / 10) for i in range(11)], self.w + aw)
        reward = self.calculate_reward()  # make reward minus to be compatible for maximum entropy RL.
        done = False
        pos = (int(self.x), int(self.y))

        # terminal state
        # terminal state criterion in incorrect.
        if pos[0] == 50 or pos[1] == 0 or pos[1] == 100 or self.t == 25:
            # if pos in [(50, 100), (50, 50), (50, 30)] or self.t == 25:
            # reward -= 0.1 * self.t
            done = True

        reward -= 0.1

        self.rewards.append(reward)

        # return a tuple of (next_obs, rewards, dones, info (timestep)
        # remain np array format in order to maintain compatibility with replay_buffer.
        return (np.array([self.x, self.y, float(self.theta), float(self.v), float(self.w)]),  # next_observation
                np.array([reward]),  # reward
                np.array([done]),  # done
                {"time_step": self.t}  # info
                )

    def reset(self):
        # print("self.t when reset was: ", self.t)
        self.x = 0
        self.y = 50
        self.theta = 0
        self.v = 2
        self.w = 0
        self.stoch = 0
        self.t = 0
        self.rewards.clear()  # clear episodic rewards record.
        return np.array([self.x, self.y, self.theta, self.v, self.w])

    def get_rewards(self):
        """
        Return a list of scalars, which are rewards obtained in current episode, from earlier to later.
        :return: A list of reward scalars.
        """
        return self.rewards

    def get_game_list(self):
        pass

    def render(self):
        pass

    def terminate(self):
        # doo voodoo!
        pass
