# Created by yingwen at 2019-03-10

import numpy as np
from copy import deepcopy
from malib.agents.tabular.q_learning.base_tabular_agent import StationaryAgent
from malib.agents.tabular.q_learning.base_q import QAgent


class EMAQAgent(QAgent):
    def __init__(self, id_, action_num, env, delta1=0.001, delta2=0.002, **kwargs):
        super().__init__(id_, action_num, env, **kwargs)
        self.name = 'emaq'
        self.delta1 = delta1
        self.delta2 = delta2
        self.pi_history = [deepcopy(self.pi)]

    def update_policy(self, s, a, game):
        if a == np.argmax(self.Q[s]):
            delta = self.delta1
            vi = np.zeros(self.action_num)
            vi[a] = 1.
        else:
            delta = self.delta2
            vi = np.zeros(self.action_num)
            vi[a] = 0.

        self.pi[s] = (1 - delta) * self.pi[s] + delta * vi
        StationaryAgent.normalize(self.pi[s])
        self.pi_history.append(deepcopy(self.pi))