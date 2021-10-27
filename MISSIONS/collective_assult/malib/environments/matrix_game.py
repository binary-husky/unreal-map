import numpy as np
from malib.spaces import Discrete, Box, MASpace, MAEnvSpec
from malib.environments.base_game import BaseGame
from malib.error import EnvironmentNotFound, WrongNumberOfAgent, WrongNumberOfAction, \
    WrongActionInputLength

class MatrixGame(BaseGame):
    def __init__(self, game_name, agent_num, action_num, payoff=None, repeated=False, max_step=25, memory=0, discrete_action=True, tuple_obs=True):
        self.game_name = game_name
        self.agent_num = agent_num
        self.action_num = action_num
        self.discrete_action = discrete_action
        self.tuple_obs = tuple_obs

        game_list = MatrixGame.get_game_list()

        if not self.game_name in game_list:
            raise EnvironmentNotFound(f"The game {self.game_name} doesn't exists")

        expt_num_agent = game_list[self.game_name]['agent_num']
        expt_num_action = game_list[self.game_name]['action_num']

        if expt_num_agent != self.agent_num:
            raise WrongNumberOfAgent(f"The number of agent \
                required for {self.game_name} is {expt_num_agent}")

        if expt_num_action != self.action_num:
            raise WrongNumberOfAction(f"The number of action \
                required for {self.game_name} is {expt_num_action}")


        self.action_spaces = MASpace(tuple(Box(low=-1., high=1., shape=(1,)) for _ in range(self.agent_num)))
        self.observation_spaces = MASpace(tuple(Discrete(1) for _ in range(self.agent_num)))

        if self.discrete_action:
            self.action_spaces = MASpace(tuple(Discrete(action_num) for _ in range(self.agent_num)))
            if memory == 0:
                self.observation_spaces = MASpace(tuple(Discrete(1) for _ in range(self.agent_num)))
            elif memory == 1:
                self.observation_spaces = MASpace(tuple(Discrete(5) for _ in range(self.agent_num)))
        else:
            self.action_range = [-1., 1.]
            self.action_spaces = MASpace(tuple(Box(low=-1., high=1., shape=(1,)) for _ in range(self.agent_num)))
            if memory == 0:
                self.observation_spaces = MASpace(tuple(Discrete(1) for _ in range(self.agent_num)))
            elif memory == 1:
                self.observation_spaces =  MASpace(tuple(Box(low=-1., high=1., shape=(12,)) for _ in range(self.agent_num)))

        self.env_specs = MAEnvSpec(self.observation_spaces, self.action_spaces)

        self.t = 0
        self.repeated = repeated
        self.max_step = max_step
        self.memory = memory
        self.previous_action = 0
        self.previous_actions = []
        self.ep_rewards = np.zeros(2)

        if payoff is not None:
            payoff = np.array(payoff)
            assert payoff.shape == tuple([agent_num] + [action_num] * agent_num)
            self.payoff = payoff
        if payoff is None:
            self.payoff = np.zeros(tuple([agent_num] + [action_num] * agent_num))

        if self.game_name == 'coordination_0_0':
            self.payoff[0]=[[1,-1],
                           [-1,-1]]
            self.payoff[1]=[[1,-1],
                           [-1,-1]]
        elif self.game_name == 'coordination_same_action_with_preference':
            self.payoff[0]=[[2, 0],
                           [0, 1]]
            self.payoff[1]=[[1, 0],
                           [0, 2]]
        elif self.game_name == 'zero_sum_nash_0_1':
            # payoff tabular of zero-sum game scenario. nash equilibrium: (Agenat1's action=0,Agent2's action=1)
            self.payoff[0]=[[5,2],
                            [-1,6]]
            self.payoff[1]=[[-5,-2],
                            [1,-6]]
        elif self.game_name == 'matching_pennies':
            # payoff tabular of zero-sumgame scenario. matching pennies
            self.payoff[0]=[[1,-1],
                           [-1,1]]
            self.payoff[1]=[[-1,1],
                           [1,-1]]
        elif self.game_name == 'matching_pennies_3':
            self.payoff[0]=[
                            [ [1,-1],
                              [-1,1] ],
                            [ [1, -1],
                             [-1, 1]]
                            ]
            self.payoff[1]=[
                            [ [1,-1],
                              [1,-1] ],
                            [[-1, 1],
                             [-1, 1]]
                            ]
            self.payoff[2] = [
                            [[-1, -1],
                             [1, 1]],
                            [[1, 1],
                             [-1, -1]]
                            ]
        elif self.game_name =='prison_lola':
            self.payoff[0]=[[-1,-3],
                           [0,-2]]
            self.payoff[1]=[[-1,0],
                           [-3,-2]]
        elif self.game_name =='prison':
            self.payoff[0]=[[3, 1],
                           [4, 2]]
            self.payoff[1]=[[3, 4],
                           [1, 2]]
        elif self.game_name =='stag_hunt':
            self.payoff[0]=[[4, 1],
                           [3, 2]]
            self.payoff[1]=[[4, 3],
                           [1, 2]]
        elif self.game_name =='chicken': # snowdrift
            self.payoff[0]=[[3, 2],
                           [4, 1]]
            self.payoff[1]=[[3, 4],
                           [2, 1]]
        elif self.game_name =='harmony':
            self.payoff[0] = [[4, 3],
                             [2, 1]]
            self.payoff[1] = [[4, 2],
                             [3, 1]]
        elif self.game_name == 'wolf_05_05':
            self.payoff[0] = [[0, 3],
                             [1, 2]]
            self.payoff[1] = [[3, 2],
                             [0, 1]]
            # \alpha, \beta = 0, 0.9, nash is 0.5 0.5
            # Q tables given, matian best response, learn a nash e.
        elif self.game_name == 'climbing':
            self.payoff[0] = [[11, -30, 0],
                              [-30, 7, 6],
                              [0, 0, 5]]
            self.payoff[1] = [[11, -30, 0],
                              [-30, 7, 6],
                              [0, 0, 5]]
        elif self.game_name == 'penalty':
            self.payoff[0] = [[10, 0, 0],
                              [0, 2, 0],
                              [0, 0, 10]]
            self.payoff[1] = [[10, 0, 0],
                              [0, 2, 0],
                              [0, 0, 10]]
        elif self.game_name == 'rock_paper_scissors':
            self.payoff[0] = [[0, -1, 1],
                              [1, 0, -1],
                              [-1, 1, 0]
                              ]
            self.payoff[1] = [[0, 1, -1],
                              [-1, 0, 1],
                              [1, -1, 0]
                              ]

        self.rewards = np.zeros((self.agent_num,))

    @staticmethod
    def get_game_list():
        return {
            'coordination_0_0': {'agent_num': 2, 'action_num': 2},
            'coordination_same_action_with_preference': {'agent_num': 2, 'action_num': 2},
            'zero_sum_nash_0_1': {'agent_num': 2, 'action_num': 2},
            'matching_pennies': {'agent_num': 2, 'action_num': 2},
            'matching_pennies_3': {'agent_num': 3, 'action_num': 2},
            'prison_lola': {'agent_num': 2, 'action_num': 2},
            'prison': {'agent_num': 2, 'action_num': 2},
            'stag_hunt': {'agent_num': 2, 'action_num': 2},
            'chicken': {'agent_num': 2, 'action_num': 2},
            'harmony': {'agent_num': 2, 'action_num': 2},
            'wolf_05_05': {'agent_num': 2, 'action_num': 2},
            'climbing': {'agent_num': 2, 'action_num': 3},
            'penalty': {'agent_num': 2, 'action_num': 3},
            'rock_paper_scissors': {'agent_num': 2, 'action_num': 3},
        }


    def V(self, alpha, beta, payoff):
        u = payoff[(0, 0)] - payoff[(0, 1)] - payoff[(1, 0)] + payoff[(1, 1)]
        return alpha * beta * u + alpha * (payoff[(0, 1)] - payoff[(1, 1)]) + beta * (
                payoff[(1, 0)] - payoff[(1, 1)]) + payoff[(1, 1)]

    def get_rewards(self, actions):
        reward_n = np.zeros((self.agent_num,))
        if self.discrete_action:
            for i in range(self.agent_num):
                assert actions[i] in range(self.action_num)
                reward_n[i] = self.payoff[i][tuple(actions)]
        else:
            actions = (actions + 1.) / 2.
            for i in range(self.agent_num):
                reward_n[i] = self.V(actions[0], actions[1], np.array(self.payoff[i]))
            print(np.array(self.payoff[0]))
            print('actions', actions)
            print('reward', reward_n)
        return reward_n

    def step(self, actions):
        if len(actions) != self.agent_num:
            raise WrongActionInputLength(f"Expected number of actions is {self.agent_num}")

        actions = np.array(actions).reshape((self.agent_num,))
        reward_n = self.get_rewards(actions)
        self.rewards = reward_n
        info = {}
        done_n = np.array([True] * self.agent_num)
        if self.repeated:
            done_n = np.array([False] * self.agent_num)
        self.t += 1
        if self.t >= self.max_step:
            done_n = np.array([True] * self.agent_num)

        state = [0] * (self.action_num * self.agent_num * (self.memory) + 1)
        # state_n = [tuple(state) for _ in range(self.agent_num)]
        if self.memory > 0 and self.t > 0:
            # print('actions', actions)
            if self.discrete_action:
                state[actions[1] + 2 * actions[0] + 1] = 1
            else:
                state = actions

        # tuple for tublar case, which need a hashabe obersrvation
        if self.tuple_obs:
            state_n = [tuple(state) for _ in range(self.agent_num)]
        else:
            state_n = np.array([state for _ in range(self.agent_num)])

        # for i in range(self.agent_num):
        #     state_n[i] = tuple(state_n[i][:])

        self.previous_actions.append(tuple(actions))
        self.ep_rewards += np.array(reward_n)
        # print(state_n, reward_n, done_n, info)
        return state_n, reward_n, done_n, info

    def reset(self):
        # print('reward,', self.ep_rewards / self.t)
        self.ep_rewards = np.zeros(2)
        self.t = 0
        self.previous_action = 0
        self.previous_actions = []
        state = [0] * (self.action_num * self.agent_num * (self.memory)  + 1)
        # state_n = [tuple(state) for _ in range(self.agent_num)]
        if self.memory > 0:
            state = [0., 0.]
        if self.tuple_obs:
            state_n = [tuple(state) for _ in range(self.agent_num)]
        else:
            state_n = np.array([state for _ in range(self.agent_num)])
        # print(state_n)

        return state_n

    def render(self, mode='human', close=False):
        if mode == 'human':
            print(self.__str__())

    def terminate(self):
        pass

    def get_joint_reward(self):
        return self.rewards

    def __str__(self):
        content = 'Game Name {}, Number of Agent {}, Number of Action \n'.format(self.game_name, self.agent_num, self.action_num)
        content += 'Payoff Matrixs:\n\n'
        for i in range(self.agent_num):
            content += 'Agent {}, Payoff:\n {} \n\n'.format(i+1, str(self.payoff[i]))
        return content


if __name__ == '__main__':
    print(MatrixGame.get_game_list())
    game = MatrixGame('matching_pennies_3', agent_num=3, action_num=2)
    print(game)
