import numpy as np
from copy import deepcopy
from malib.agents.tabular.q_learning.base_tabular_agent import StationaryAgent
from malib.agents.tabular.q_learning.base_q import QAgent


class PGAAPPAgent(QAgent):
    def __init__(self, id_, action_num, env, eta=0.01, **kwargs):
        super().__init__(id_, action_num, env, **kwargs)
        self.name = 'pha-app'
        self.eta = eta
        self.pi_history = [deepcopy(self.pi)]

    def update_policy(self, s, a, game):
        V = np.dot(self.pi[s], self.Q[s])
        delta_hat_A = np.zeros(self.action_num)
        delta_A = np.zeros(self.action_num)
        for ai in range(self.action_num):
            if self.pi[s][ai] == 1:
                delta_hat_A[ai]= self.Q[s][ai] - V
            else:
                delta_hat_A[ai] = (self.Q[s][ai] - V) / (1 - self.pi[s][ai])
            delta_A[ai] = delta_hat_A[ai] - self.gamma * abs(delta_hat_A[ai]) *self.pi[s][ai]
        self.pi[s] += self.eta * delta_A
        StationaryAgent.normalize(self.pi[s])
        self.pi_history.append(deepcopy(self.pi))

