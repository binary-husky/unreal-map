# Created by yingwen at 2019-03-10

import numpy as np
from copy import deepcopy
from collections import defaultdict
from functools import partial
from malib.agents.tabular.q_learning.base_q import QAgent
from malib.agents.tabular.utils import softmax



class OMQAgent(QAgent):
    def __init__(self, id_, action_num, env, **kwargs):
        super().__init__(id_, action_num, env, **kwargs)
        self.name = 'omq'
        self.count_SO = defaultdict(partial(np.zeros, self.action_num))
        self.opponent_pi = defaultdict(partial(np.random.dirichlet, [1.0] * self.action_num))
        self.pi_history = [deepcopy(self.pi)]
        self.opponent_pi_history = [deepcopy(self.opponent_pi)]
        self.Q = defaultdict(partial(np.random.rand, *(self.action_num, self.action_num)))
        self.R = defaultdict(partial(np.zeros, (self.action_num, self.action_num)))
        self.count_R = defaultdict(partial(np.zeros, (self.action_num, self.action_num)))


    def update(self, s, a, o, r, s2, env, done=False):
        self.count_SO[s][o] += 1.
        self.opponent_pi[s] = self.count_SO[s] / np.sum(self.count_SO[s])
        self.count_R[s][a][o] += 1.0
        self.R[s][a][o] += (r - self.R[s][a][o]) / self.count_R[s][a][o]
        Q = self.Q[s]
        V = self.val(s2)
        decay_alpha = self.step_decay()
        if done:
            Q[a][o] = Q[a][o] + decay_alpha * (r - Q[a])
        else:
            Q[a][o] = Q[a][o] + decay_alpha * (r + self.gamma * V - Q[a][o])
        if self.verbose:
            print(self.epoch)
        self.update_policy(s, a, env)
        self.record_policy(s, env)
        self.epoch += 1

    def val(self, s):
        return np.max(np.dot(self.Q[s], self.opponent_pi[s]))

    def update_policy(self, s, a, game):
        # print('Qs {}'.format(self.Q[s]))
        # print('OPI {}'.format(self.opponent_best_pi[s]))
        # print('pis: ' + str(np.dot(self.Q[s], self.opponent_best_pi[s])))
        self.pi[s] = softmax(np.dot(self.Q[s], self.opponent_pi[s]))

        # print('pis: ' + str(np.sum(np.dot(self.Q[s], self.opponent_best_pi[s]))))
        self.pi_history.append(deepcopy(self.pi))
        self.opponent_pi_history.append(deepcopy(self.opponent_pi))
        if self.verbose:
            print('opponent pi of {}: {}'.format(self.id_, self.opponent_pi[s]))