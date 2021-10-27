# Created by yingwen at 2019-03-10

import numpy as np
from copy import deepcopy
from collections import defaultdict
from functools import partial
from malib.agents.tabular.q_learning.base_q import QAgent
from malib.agents.tabular.utils import softmax


class RRQAgent(QAgent):
    def __init__(self, id_, action_num, env, phi_type='count', a_policy='softmax', **kwargs):
        super().__init__(id_, action_num, env, **kwargs)
        self.name = 'RR2Q'
        self.phi_type = phi_type
        self.a_policy = a_policy
        self.count_AOS = defaultdict(partial(np.zeros, (self.action_num, self.action_num)))
        self.count_OS = defaultdict(partial(np.zeros, (self.action_num, )))
        self.opponent_best_pi = defaultdict(partial(np.random.dirichlet, [1.0] * self.action_num))
        self.pi_history = [deepcopy(self.pi)]
        self.opponent_best_pi_history = [deepcopy(self.opponent_best_pi)]
        self.Q = defaultdict(partial(np.random.rand, *(self.action_num, self.action_num)))
        self.Q_A = defaultdict(partial(np.random.rand, self.action_num))
        self.R = defaultdict(partial(np.zeros, (self.action_num, self.action_num)))
        self.count_R = defaultdict(partial(np.zeros, (self.action_num, self.action_num)))

    def update(self, s, a, o, r, s2, env, done=False, tau=0.5):
        self.count_AOS[s][a][o] += 1.0
        self.count_OS[s][o] += 1.
        decay_alpha = self.step_decay()
        if self.phi_type == 'count':
            count_sum = np.reshape(np.repeat(np.sum(self.count_AOS[s], 1), self.action_num), (self.action_num, self.action_num))
            self.opponent_best_pi[s] = self.count_AOS[s] / (count_sum + 0.1)
            self.opponent_best_pi[s] = self.opponent_best_pi[s] / (np.sum(self.opponent_best_pi[s]) + 0.1)
        elif self.phi_type == 'norm-exp':
            self.Q_A_reshaped = np.reshape(np.repeat(self.Q_A[s], self.action_num), (self.action_num, self.action_num))
            self.opponent_best_pi[s] = np.log(np.exp((self.Q[s] - self.Q_A_reshaped)))
            self.opponent_best_pi[s] = self.opponent_best_pi[s] / np.reshape(
                np.repeat(np.sum(self.opponent_best_pi[s], 1), self.action_num), (self.action_num, self.action_num))

        self.count_R[s][a][o] += 1.0
        self.R[s][a][o] += (r - self.R[s][a][o]) / self.count_R[s][a][o]
        Q = self.Q[s]
        V = self.val(s2)
        if done:
            Q[a][o] = Q[a][o] + decay_alpha * (r - Q[a][o])
            self.Q_A[s][a] = self.Q_A[s][a] + decay_alpha * (r - self.Q_A[s][a])
        else:
            Q[a][o] = Q[a][o] + decay_alpha * (r + self.gamma * V - Q[a][o])
            self.Q_A[s][a] = self.Q_A[s][a] + decay_alpha * (r + self.gamma * V - self.Q_A[s][a])
        if self.verbose:
            print(self.epoch)
        self.update_policy(s, a, env)
        self.record_policy(s, env)
        self.epoch += 1

    def val(self, s):
        return np.max(np.sum(np.multiply(self.Q[s], self.opponent_best_pi[s]), 1))

    def update_policy(self, s, a, game):
        if self.a_policy == 'softmax':
            self.pi[s] = softmax(np.sum(np.multiply(self.Q[s], self.opponent_best_pi[s]), 1))
        else:
            Q = np.sum(np.multiply(self.Q[s], self.opponent_best_pi[s]), 1)
            self.pi[s] = (Q == np.max(Q)).astype(np.double)
        self.pi_history.append(deepcopy(self.pi))
        self.opponent_best_pi_history.append(deepcopy(self.opponent_best_pi))
        if self.verbose:
            print('opponent pi of {}: {}'.format(self.id_, self.opponent_best_pi))




class GRRQAgent(QAgent):
    def __init__(self, id_, action_num, env, k=0, phi_type='count', a_policy='softmax', **kwargs):
        super().__init__(id_, action_num, env, **kwargs)
        self.name = 'GRRQ'
        self.k = k
        self.phi_type = phi_type
        self.a_policy = a_policy
        self.count_AOS = defaultdict(partial(np.zeros, (self.action_num, self.action_num)))
        self.opponent_best_pi = defaultdict(partial(np.random.dirichlet, [1.0] * self.action_num))
        self.pi_history = [deepcopy(self.pi)]
        self.opponent_best_pi_history = [deepcopy(self.opponent_best_pi)]
        self.Q = defaultdict(partial(np.random.rand, *(self.action_num, self.action_num)))
        self.Q_A = defaultdict(partial(np.random.rand, self.action_num))
        self.R = defaultdict(partial(np.zeros, (self.action_num, self.action_num)))
        self.count_R = defaultdict(partial(np.zeros, (self.action_num, self.action_num)))


    def update(self, s, a, o, r, s2, env, done=False):
        self.count_AOS[s][a][o] += 1.0
        decay_alpha = self.step_decay()
        if self.phi_type == 'count':
            count_sum = np.reshape(np.repeat(np.sum(self.count_AOS[s], 1), self.action_num), (self.action_num, self.action_num))
            self.opponent_best_pi[s] = self.count_AOS[s] / (count_sum + 0.1)
        elif self.phi_type == 'norm-exp':
            self.Q_A_reshaped = np.reshape(np.repeat(self.Q_A[s], self.action_num), (self.action_num, self.action_num))
            self.opponent_best_pi[s] = np.log(np.exp(self.Q[s] - self.Q_A_reshaped))
            self.opponent_best_pi[s] = self.opponent_best_pi[s] / np.reshape(
                np.repeat(np.sum(self.opponent_best_pi[s], 1), self.action_num), (self.action_num, self.action_num))

        self.count_R[s][a][o] += 1.0
        self.R[s][a][o] += (r - self.R[s][a][o]) / self.count_R[s][a][o]
        Q = self.Q[s]
        V = self.val(s2)
        if done:
            Q[a][o] = Q[a][o] + decay_alpha * (r - Q[a][o])
            self.Q_A[s][a] = self.Q_A[s][a] + decay_alpha * (r - self.Q_A[s][a])
        else:
            Q[a][o] = Q[a][o] + decay_alpha * (r + self.gamma * V - Q[a][o])
            self.Q_A[s][a] = self.Q_A[s][a] + decay_alpha * (r + self.gamma * V - self.Q_A[s][a])
        print(self.epoch)
        self.update_policy(s, a, env)
        self.record_policy(s, env)
        self.epoch += 1

    def val(self, s):
        return np.max(np.sum(np.multiply(self.Q[s], self.opponent_best_pi[s]), 1))

    def update_policy(self, s, a, game):
        if self.a_policy == 'softmax':
            self.pi[s] = softmax(np.sum(np.multiply(self.Q[s], self.opponent_best_pi[s]), 1))
        else:
            Q = np.sum(np.multiply(self.Q[s], self.opponent_best_pi[s]), 1)
            self.pi[s] = (Q == np.max(Q)).astype(np.double)
        self.pi_history.append(deepcopy(self.pi))
        self.opponent_best_pi_history.append(deepcopy(self.opponent_best_pi))
        print('opponent pi of {}: {}'.format(self.id_, self.opponent_best_pi))
