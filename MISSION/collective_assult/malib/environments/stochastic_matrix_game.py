import numpy as np

from malib.spaces import Discrete, Box, MASpace, MAEnvSpec
from malib.environments.base_game import BaseGame
from malib.error import EnvironmentNotFound, WrongNumberOfAgent, \
    WrongNumberOfAction, WrongNumberOfState, WrongActionInputLength

class StochasticMatrixGame(BaseGame):
    def __init__(self, game_name, agent_num, action_num, state_num, payoff=None, transition=None):
        self.game_name = game_name
        self.agent_num = agent_num
        self.action_num = action_num
        self.state_num = state_num

        game_list = StochasticMatrixGame.get_game_list()

        if not self.game_name in game_list:
            raise EnvironmentNotFound(f"The game {self.game_name} doesn't exists")

        expt_num_agent = game_list[self.game_name]['agent_num']
        if expt_num_agent != self.agent_num:
            raise WrongNumberOfAgent(f"The number of agent \
                required for {self.game_name} is {expt_num_agent}")

        expt_num_action = game_list[self.game_name]['action_num']
        if expt_num_agent != self.action_num:
            raise WrongNumberOfAction(f"The number of action \
                required for {self.game_name} is {expt_num_action}")

        expt_num_state = game_list[self.game_name]['state_num']
        if expt_num_state != self.state_num:
            raise WrongNumberOfState(f"The number of state \
                required for {self.game_name} is {expt_num_state}")

        self.action_spaces = MASpace(tuple(Box(low=-1., high=1., shape=(1,)) for _ in range(self.agent_num)))
        self.observation_spaces = MASpace(tuple(Discrete(1) for _ in range(self.agent_num)))
        self.env_specs = MAEnvSpec(self.observation_spaces, self.action_spaces)

        self.t = 0
        if payoff is not None:
            payoff = np.array(payoff)
            assert payoff.shape == tuple([state_num, agent_num] + [action_num] * agent_num)
            self.payoff = payoff
        if payoff is None:
            self.payoff = np.zeros(tuple([state_num, agent_num] + [action_num] * agent_num))

        if transition is None:
            self.transition = np.zeros(tuple([state_num] + [action_num] * agent_num + [state_num]))

        if self.game_name == 'PollutionTax':
            self.payoff[0][0] = [[4., 3.],
                                [7., 6.]]
            self.payoff[0][1] = [[5., 8.],
                                [4., 7.]]
            self.payoff[1][0] = [[1., 0.],
                                 [4., 3.]]
            self.payoff[1][1] = [[2., 5.],
                                 [1., 4.]]
            self.transition[0] = [[[1., 0.], [0., 1.]],
                                 [[0., 1.], [0., 1.]]]
            self.transition[1] = [[[1., 0.], [0., 1.]],
                                  [[0., 1.], [0., 1.]]]
        elif self.game_name=='three_matrix_games':
            self.g1 = [[0.,3.], [2.,-1.]]
            self.g2 = [[0., 1.], [4., 3.]]
            self.g = [['g1', 4.], [5., 'g2']]

        self.rewards = np.zeros((self.agent_num,))
        self.state = 0

    def get_three_matrix_games(self, a_n):
        assert len(a_n) == 2
        info = {}
        reward_n = np.zeros((self.agent_num,))
        if self.state == 0:
            if a_n[0] == a_n[1] == 0:
                state_prime = 1
                state_n = np.array([state_prime] * self.agent_num)
                self.state = state_prime
                done_n = np.array([False] * self.agent_num)
            elif a_n[0] == a_n[1] == 1:
                state_prime = 2
                state_n = np.array([state_prime] * self.agent_num)
                self.state = state_prime
                done_n = np.array([False] * self.agent_num)
            else:
                reward_n[0] = self.g[a_n[0]][a_n[1]]
                reward_n[0] = -self.g[a_n[0]][a_n[1]]
                state_prime = 0
                state_n = np.array([state_prime] * self.agent_num)
                self.state = state_prime
                done_n = np.array([True] * self.agent_num)
        if self.state == 1:
            reward_n[0] = self.g1[a_n[0]][a_n[1]]
            reward_n[0] = -self.g1[a_n[0]][a_n[1]]
            state_prime = 0
            state_n = np.array([state_prime] * self.agent_num)
            self.state = state_prime
            done_n = np.array([True] * self.agent_num)
        if self.state == 2:
            reward_n[0] = self.g2[a_n[0]][a_n[1]]
            reward_n[0] = -self.g2[a_n[0]][a_n[1]]
            state_prime = 0
            state_n = np.array([state_prime] * self.agent_num)
            self.state = state_prime
            done_n = np.array([True] * self.agent_num)
        return state_n, reward_n, done_n, info


    @staticmethod
    def get_game_list():
        return {
            'PollutionTax': {'agent_num': 2, 'action_num': 2, 'state_num': 2},
            'three_matrix_games': {'agent_num': 2, 'action_num': 2, 'state_num': 3},
        }

    def step(self, actions):
        if len(actions) != self.agent_num:
            raise WrongActionInputLength(f"Expected number of actions is {self.agent_num}")

        if self.game_name == 'three_matrix_games':
            return self.get_three_matrix_games(actions)

        reward_n = np.zeros((self.agent_num,))
        for i in range(self.agent_num):
            assert actions[i] in range(self.action_num)
            reward_n[i] = self.payoff[self.state][i][tuple(actions)]

        self.rewards = reward_n
        state_prime = np.random.choice(2, 1, p=self.transition[self.state][tuple(actions)])[0]
        print('state_prime', state_prime)
        state_n = np.array([state_prime] * self.agent_num)
        self.state = state_prime
        info = {}
        done_n = np.array([True] * self.agent_num)
        self.t += 1

        return state_n, reward_n, done_n, info

    def reset(self):
        self.state = 0
        return np.array([0] * self.agent_num)

    def render(self, mode='human', close=False):
        if mode == 'human':
            print(self.__str__())

    def get_rewards(self):
        return self.rewards

    def __str__(self):
        content = 'Game Name {}, Number of Agent {}, Number of Action \n'.format(self.game_name, self.agent_num, self.action_num)
        content += 'Payoff Matrixs:\n\n'
        for i in range(self.agent_num):
            content += 'Agent {}, Payoff:\n {} \n\n'.format(i+1, str(self.payoff[i]))
        return content
