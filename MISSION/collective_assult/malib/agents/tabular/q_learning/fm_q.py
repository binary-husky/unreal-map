# Created by yingwen at 2019-03-10


import numpy as np
from copy import deepcopy
from collections import defaultdict
from functools import partial
from malib.agents.tabular.q_learning.base_q import QAgent


class FMQAgent(QAgent):
    def __init__(self, id_, action_num, env, tau=500, c=10,
                 **kwargs):
        super().__init__(id_, action_num, env, **kwargs)
        self.name = 'FMQ'
        self.Q = defaultdict(partial(np.random.rand, self.action_num))
        self.tau = tau
        self.init_tau = tau
        self.max_R = defaultdict(partial(np.zeros, self.action_num))
        self.max_R_a_counter = defaultdict(partial(np.zeros, self.action_num))
        self.max_R_freq = defaultdict(partial(np.zeros, self.action_num))
        self.c = c

    def update_max_R_freq(self, s, a, r):
        if r >= self.max_R[s][a]:
            self.max_R[s][a] = r
            self.max_R_a_counter[s][a] += 1.0
        self.max_R_freq[s][a] = self.max_R_a_counter[s][a]/self.count_R[s][a]

    def update(self, s, a, o, r, s2, env, done=False):
        self.count_R[s][a] += 1.0
        self.update_max_R_freq(s, a, r)
        Q = self.Q[s]
        V = self.val(s2)
        decay_alpha = self.step_decay()
        if done:
            Q[a] = Q[a] + 0.9 * (r - Q[a])
        else:
            Q[a] = Q[a] + decay_alpha * (r + self.gamma * V - Q[a])
        if self.verbose:
            print(self.epoch)
        self.update_policy(s, a, env)
        self.record_policy(s, env)
        self.pi_history.append(deepcopy(self.pi))
        self.epoch += 1

    def val(self, s):
        return np.max(self.Q[s])

    def update_policy(self, s, a, env):
        tau = np.exp(-0.1*self.epoch)*self.init_tau+1.0
        Q = self.Q[s]
        pi = Q + self.c * self.max_R_freq[s]*self.max_R[s]
        self.pi[s] = np.exp(pi/tau)/np.sum(np.exp(pi/tau))
