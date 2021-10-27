import numpy as np
from malib.spaces import Discrete, Box, MASpace, MAEnvSpec
from malib.environments.base_game import BaseGame
from malib.error import EnvironmentNotFound, WrongNumberOfAgent, WrongActionInputLength

class DifferentialGame(BaseGame):
    def __init__(self, game_name, agent_num, action_range=(-10, 10)):
        self.game_name = game_name
        self.agent_num = agent_num
        self.action_range = action_range

        game_list = DifferentialGame.get_game_list()

        if not self.game_name in game_list:
            raise EnvironmentNotFound(f"The game {self.game_name} doesn't exists")

        expt_num_agent = game_list[self.game_name]['agent_num']
        if expt_num_agent != self.agent_num:
            raise WrongNumberOfAgent(f"The number of agent \
                required for {self.game_name} is {expt_num_agent}")

        self.action_spaces = MASpace(tuple(Box(low=-1., high=1., shape=(1,)) for _ in range(self.agent_num)))
        self.observation_spaces = MASpace(tuple(Box(low=-1., high=1., shape=(1,)) for _ in range(self.agent_num)))
        self.env_specs = MAEnvSpec(self.observation_spaces, self.action_spaces)
        self.t = 0
        self.payoff = {}

        if self.game_name == 'zero_sum':
            self.payoff[0] = lambda a1, a2: a1 * a2
            self.payoff[1] = lambda a1, a2: -a1 * a2
        elif self.game_name == 'trigonometric':
            self.payoff[0] = lambda a1, a2: np.cos(a2) * a1
            self.payoff[1] = lambda a1, a2: np.sin(a1) * a2
        elif self.game_name == 'mataching_pennies':
            self.payoff[0] = lambda a1, a2: (a1-0.5)*(a2-0.5)
            self.payoff[1] = lambda a1, a2: (a1-0.5)*(a2-0.5)
        elif self.game_name == 'rotational':
            self.payoff[0] = lambda a1, a2: 0.5 * a1 * a1 + 10 * a1 * a2
            self.payoff[1] = lambda a1, a2: 0.5 * a2 * a2 - 10 * a1 * a2
        elif self.game_name == 'wolf':
            def V(alpha, beta, payoff):
                u = payoff[(0, 0)] - payoff[(0, 1)] - payoff[(1, 0)] + payoff[(1, 1)]
                return alpha * beta * u + alpha * (payoff[(0, 1)] - payoff[(1, 1)]) + beta * (
                            payoff[(1, 0)] - payoff[(1, 1)]) + payoff[(1, 1)]

            payoff_0 = np.array([[0, 3], [1, 2]])
            payoff_1 = np.array([[3, 2], [0, 1]])

            self.payoff[0] = lambda a1, a2: V(a1, a2, payoff_0)
            self.payoff[1] = lambda a1, a2: V(a1, a2, payoff_1)
        elif self.game_name == 'ma_softq':
            h1 = 0.8
            h2 = 1.
            s1 = 3.
            s2 = 1.
            x1 = -5.
            x2 = 5.
            y1 = -5.
            y2 = 5.
            c = 10.
            def max_f(a1, a2):
                f1 = h1 * (-(np.square(a1 - x1) / s1) - (np.square(a2 - y1) / s1))
                f2 = h2 * (-(np.square(a1 - x2) / s2) - (np.square(a2 - y2) / s2)) + c
                return max(f1, f2)
            self.payoff[0] = lambda a1, a2: max_f(a1, a2)
            self.payoff[1] = lambda a1, a2: max_f(a1, a2)
        else:
            raise EnvironmentNotFound(f"The game {self.game_name} doesn't exists")

        self.rewards = np.zeros((self.agent_num,))

    @staticmethod
    def get_game_list():
        return {
            'zero_sum': {'agent_num': 2, 'action_num': 2},
            'trigonometric': {'agent_num': 2},
            'mataching_pennies': {'agent_num': 2},
            'rotational': {'agent_num': 2},
            'wolf': {'agent_num': 2},
            'ma_softq': {'agent_num': 2},
        }

    def step(self, actions):

        if len(actions) != self.agent_num:
            raise WrongActionInputLength(f"Expected number of actions is {self.agent_num}")

        print('actions', actions)
        actions = np.array(actions).reshape((self.agent_num,)) * self.action_range[1]
        print('scaled', actions)
        reward_n = np.zeros((self.agent_num,))
        for i in range(self.agent_num):
            print('actions', actions)
            reward_n[i] = self.payoff[i](*tuple(actions))
        self.rewards = reward_n
        print(reward_n)
        state_n = np.array(list([[0. * i] for i in range(self.agent_num)]))
        info = {}
        done_n = np.array([True] * self.agent_num)
        self.t += 1
        print('state_n, reward_n, done_n, info', state_n, reward_n, done_n, info)
        return state_n, reward_n, done_n, info

    def reset(self):
        return np.array(list([[0. * i] for i in range(self.agent_num)]))

    def render(self, mode='human', close=False):
        if mode == 'human':
            print(self.__str__())

    def get_rewards(self):
        return self.rewards

    def terminate(self):
        pass

    def __str__(self):
        content = 'Game Name {}, Number of Agent {}, Action Range {}\n'.format(self.game_name, self.agent_num, self.action_range)
        return content


if __name__ == '__main__':
    print(DifferentialGame.get_game_list())
    game = DifferentialGame('zero_sum', agent_num=2)
    print(game)
