import random
import os
from abc import abstractmethod
from collections import defaultdict
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from malib.agents.tabular.q_learning.base_tabular_agent import Agent, StationaryAgent
import maci.utils as utils
from copy import deepcopy


class BaseQAgent(Agent):
    def __init__(self, name, id_, action_num, env, alpha_decay_steps=10000., alpha=0.01, gamma=0.95, episilon=0.1, verbose=True, **kwargs):
        super().__init__(name, id_, action_num, env, **kwargs)
        self.episilon = episilon
        self.alpha_decay_steps = alpha_decay_steps
        self.gamma = gamma
        self.alpha = alpha
        self.epoch = 0
        self.Q = None
        self.pi = defaultdict(partial(np.random.dirichlet, [1.0] * self.action_num))
        self.record = defaultdict(list)
        self.verbose = verbose
        self.pi_history = [deepcopy(self.pi)]

    def done(self, env):
        if self.verbose:
            utils.pv('self.full_name(game)')
            utils.pv('self.Q')
            utils.pv('self.pi')

        numplots = env.numplots if env.numplots >= 0 else len(self.record)
        for s, record in sorted(
                self.record.items(), key=lambda x: -len(x[1]))[:numplots]:
            self.plot_record(s, record, env)
        self.record.clear()

    # learning rate decay
    def step_decay(self):
        # drop = 0.5
        # epochs_drop = 10000
        # decay_alpha = self.alpha * math.pow(drop, math.floor((1 + self.epoch) / epochs_drop))
        # return 1 / (1 / self.alpha + self.epoch * 1e-4)
        return self.alpha_decay_steps / (self.alpha_decay_steps + self.epoch)
        # return decay_alpha
    # def alpha(self, t):
    #     return self.alpha_decay_steps / (self.alpha_decay_steps + t)

    def act(self, s, exploration, game):
        if exploration and random.random() < self.episilon:
            return random.randint(0, self.action_num - 1)
        else:
            if self.verbose:
                for s in self.Q.keys():
                    print('{}--------------'.format(self.id_))
                    print('Q of agent {}: state {}: {}'.format(self.id_, s, str(self.Q[s])))
                    # print('QAof agent {}: state {}: {}'.format(self.id_, s, str(self.Q_A[s])))
                    # self.Q_A
                    print('pi of agent {}: state {}: {}'.format(self.id_, s, self.pi[s]))
                    # print('pi of opponent agent {}: state{}: {}'.format(self.id_, s, self.opponent_best_pi[s]))
                    print('{}--------------'.format(self.id_))
            # print()
            return StationaryAgent.sample(self.pi[s])

    @abstractmethod
    def update(self, s, a, o, r, s2, env, done=False):
        pass

    @abstractmethod
    def update_policy(self, s, a, env):
        pass

    def plot_record(self, s, record, env):
        os.makedirs('policy/', exist_ok=True)
        fig = plt.figure(figsize=(18, 10))
        n = self.action_num
        for a in range(n):
            plt.subplot(n, 1, a + 1)
            plt.tight_layout()
            plt.gca().set_ylim([-0.05, 1.05])
            plt.gca().set_xlim([1.0, env.t + 1.0])
            plt.title('player: {}: state: {}, action: {}'.format(self.full_name(env), s, a))
            plt.xlabel('step')
            plt.ylabel('pi[a]')
            plt.grid()
            x, y = list(zip(*((t, pi[a]) for t, pi in record)))
            x, y = list(x) + [env.t + 1.0], list(y) + [y[-1]]
            plt.plot(x, y, 'r-')
        fig.savefig('policy/{}_{}.pdf'.format(self.full_name(env), s))
        plt.close(fig)

    def record_policy(self, s, env):
        pass
        # if env.numplots != 0:
        #     if s in self.record:
        #         self.record[s].append((env.t - 0.01, self.record[s][-1][1]))
        #     self.record[s].append((env.t, np.copy(self.pi[s])))


class QAgent(BaseQAgent):
    def __init__(self, id_, action_num, env, **kwargs):
        super().__init__('q', id_, action_num, env, **kwargs)
        self.Q = defaultdict(partial(np.random.rand, self.action_num))
        self.R = defaultdict(partial(np.zeros, self.action_num))
        self.count_R = defaultdict(partial(np.zeros, self.action_num))

    def done(self, env):
        self.R.clear()
        self.count_R.clear()
        super().done(env)

    def update(self, s, a, o, r, s2, env, done=False):
        self.count_R[s][a] += 1.0
        self.R[s][a] += (r - self.R[s][a]) / self.count_R[s][a]
        Q = self.Q[s]
        V = self.val(s2)
        decay_alpha = self.step_decay()
        if done:
            Q[a] = Q[a] + decay_alpha * (r - Q[a])
        else:
            Q[a] = Q[a] + decay_alpha * (r + self.gamma * V - Q[a])
        if self.verbose:
            print(self.epoch)
        self.update_policy(s, a, env)
        self.record_policy(s, env)
        self.epoch += 1

    def val(self, s):
        return np.max(self.Q[s])

    def update_policy(self, s, a, env):
        Q = self.Q[s]
        self.pi[s] = (Q == np.max(Q)).astype(np.double)






