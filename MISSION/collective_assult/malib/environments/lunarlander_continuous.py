# actions a_v = forward acceleration [-1, 2], a_w = angular acceleration [-1, 1]
import gym
import numpy as np

from malib.environments.base_game import BaseGame
from malib.spaces import Box, MASpace, MAEnvSpec

# Boat state = {x, y, theta, v, w}
# x, y = coords [0, 50], [0, 100]
# theta = boat's angle [-60, 60] degrees
# v = speed, w = angle speed [2, 5], [-1, 1]


UPDATE_INTERVAL = 10


class LunarLanderContinuous(BaseGame):
    def __init__(self):
        self.env = gym.make('LunarLanderContinuous-v2')

        self.agent_num = 2
        self.observation_spaces = MASpace(tuple(self.env.observation_space for _ in range(self.agent_num)))
        self.action_spaces = MASpace(tuple([Box(low=-1., high=1., shape=(1,)) for _ in range(self.agent_num)]))
        self.env_specs = MAEnvSpec(self.observation_spaces, self.action_spaces)

    def step(self, actions):
        av, aw = actions
        av = float(av)
        aw = float(aw)

        next_ob, reward, done, info = self.env.step([av, aw])

        # next_obs = np.array(
        # 	[[(self.x - 25) / 25., (self.y - 50) / 500., float(self.theta), (float(self.v) - 4) / 2, float(self.w)]
        # 	 for _ in range(self.agent_num)])
        next_obs = np.array([next_ob] * self.agent_num)
        rewards = np.array([reward] * self.agent_num)
        dones = np.array([done] * self.agent_num)

        return next_obs, rewards, dones, info

    # return (np.array(
    # 	[[self.x, self.y, float(self.theta), float(self.v), float(self.w)] for _ in range(self.agent_num)]),
    #         np.array([reward] * self.agent_num), np.array([done] * self.agent_num), {"time_step": self.t})

    def reset(self):
        init_ob = self.env.reset()
        init_obs = np.array([init_ob] * self.agent_num)

        return init_obs

    # return np.array([[self.x, self.y, self.theta, self.v, self.w] for _ in range(self.agent_num)])

    def get_rewards(self):
        """
        Return a list of scalars, which are rewards obtained in current episode, from earlier to later.
        :return: A list of reward scalars.
        """
        pass

    def get_game_list(self):
        pass

    def render(self):
        self.env.render()

    def terminate(self):
        # doo voodoo!
        pass
