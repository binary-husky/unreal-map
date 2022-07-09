# Created by yingwen at 2019-03-10

import random
import numpy as np
from copy import deepcopy
from collections import defaultdict
from functools import partial
from malib.agents.tabular.q_learning.base_q import QAgent


class JALAgent(QAgent):
    def __init__(self, id_, action_num, env, sliding_wnd_size=50, exploration=True,
                 **kwargs):
        super().__init__(id_, action_num, env, **kwargs)
        self.name = 'JAL'
        self.exploration = exploration
        self.episilon_init = np.copy(self.episilon)

        self.pi_history = [deepcopy(self.pi)]
        self.Q = defaultdict(partial(np.random.rand, *(self.action_num, self.action_num)))
        self.sliding_wnd_size = sliding_wnd_size

        self.pi_neg_i = defaultdict(
            partial(np.random.dirichlet, [1.0] * self.action_num)
        )
        self.opponent_best_pi_history = [deepcopy(self.pi_neg_i)]
        self.replay_buffer = []

    def update_opponent_action_prob(self, s, a_i, a_neg_i, s_prime, r):
        self.replay_buffer.append((s, a_i, a_neg_i, s_prime, r))
        sliding_wnd_size = min(self.sliding_wnd_size, len(self.replay_buffer))
        sliding_window = self.replay_buffer[-sliding_wnd_size:]
        a_neg_i_map = Counter()
        # update opponent action probability
        denominator = 0
        for exp in sliding_window:
            if exp[0] != s:  # state
                continue
            denominator += 1
            a_neg_i_map[exp[2]] += 1
        self.pi_neg_i[s] = np.array(
            [a_neg_i_map[i] / denominator for i in range(self.action_num)])

    def update(self, s, a, o, r, s2, env, done=False):
        self.update_opponent_action_prob(s=s, a_i=a, a_neg_i=o, s_prime=s2, r=r)
        self.update_policy(done=done)
        self.pi_history.append(deepcopy(self.pi))
        self.opponent_best_pi_history.append(deepcopy(self.pi_neg_i))
        self.epoch += 1

    def compute_marginal_pi(self, s, one_hot=True):
        self.pi[s] = np.sum(np.multiply(self.Q[s], self.pi_neg_i[s]), 1)
        if one_hot == True:
            self.pi[s] = (self.pi[s] == np.max(self.pi[s])).astype(int)
        else:
            self.pi[s] /= np.sum(self.pi[s])
        return self.pi[s]

    def update_policy(self, k=1, gamma=0.95, done=True):
        # sliding_wnd_size = min(self.sliding_wnd_size, len(self.replay_buffer))
        # sliding_window = self.replay_buffer[-sliding_wnd_size:]
        # sample = sliding_window[np.random.choice(len(sliding_window), size=1)[0]]
        sample = self.replay_buffer[-1]
        s, a_i, a_neg_i, s_prime, r = sample
        decay_alpha = self.step_decay()
        self.episilon = self.episilon_init*self.step_decay()
        if not done:
            v_s_prime = np.max(self.compute_marginal_pi(s_prime))
            y = r + gamma * v_s_prime * (1 - done)
        else:
            y = r
        self.Q[s][a_i, a_neg_i] = (
            (1 - decay_alpha) * self.Q[s][a_i, a_neg_i] +
            decay_alpha * y
        )
        self.compute_marginal_pi(s)

    def act(self, s, exploration, game):
        """
        Function act sample actions from pi for a given state.
        Input:
            s: Int representing a state.
        Returns:
            Int: Sampled action for agent i,
                 Sampled action for the opponent according to our
                 belief.
        """
        agent_p = self.compute_marginal_pi(s, one_hot=False)
        if self.exploration and random.random() < self.episilon:
            agent_action = random.randint(0, self.action_num - 1)
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
            agent_action = np.argmax(agent_p)
        return agent_action

