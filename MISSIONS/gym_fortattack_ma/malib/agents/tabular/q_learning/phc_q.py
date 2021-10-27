from collections import defaultdict
from functools import partial

import numpy as np
from copy import deepcopy
from malib.agents.tabular.q_learning.base_q import QAgent
from malib.agents.tabular.q_learning.base_tabular_agent import StationaryAgent


class PHCAgent(QAgent):
    def __init__(self, id_, action_num, env, delta=0.02, **kwargs):
        super().__init__(id_, action_num, env, **kwargs)
        self.name = 'phc'
        self.delta = delta
        self.pi_history = [deepcopy(self.pi)]

    def update_policy(self, s, a, game):
        delta = self.delta
        if a == np.argmax(self.Q[s]):
            self.pi[s][a] += delta
        else:
            self.pi[s][a] -= delta / (self.action_num - 1)
        StationaryAgent.normalize(self.pi[s])
        self.pi_history.append(deepcopy(self.pi))


class WoLFPHCAgent(PHCAgent):
    def __init__(self, _id, action_num, env, delta_w=0.0025, delta_l=0.005, **kwargs):
        super().__init__(_id, action_num, env, **kwargs)
        self.name = 'wolf'
        self.delta_w = delta_w
        self.delta_l = delta_l
        self.pi_ = defaultdict(partial(np.random.dirichlet, [1.0] * self.action_num))
        self.count_pi = defaultdict(int)

    def done(self, env):
        self.pi_.clear()
        self.count_pi.clear()
        super().done(env)

    def update_policy(self, s, a, env):
        self.count_pi[s] += 1
        self.pi_[s] += (self.pi[s] - self.pi_[s]) / self.count_pi[s]
        self.delta = self.delta_w \
            if np.dot(self.pi[s], self.Q[s]) \
               > np.dot(self.pi_[s], self.Q[s]) \
            else self.delta_l
        super().update_policy(s, a, env)
