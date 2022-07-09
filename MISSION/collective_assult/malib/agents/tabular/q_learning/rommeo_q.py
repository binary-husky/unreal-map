# Created by yingwen at 2019-03-10

import numpy as np
from copy import deepcopy
from collections import defaultdict, Counter
from functools import partial
from malib.agents.tabular.q_learning.base_q import QAgent, StationaryAgent
import random

class ROMMEOAgent(QAgent):
    def __init__(self, id_, action_num, env, phi_type='rommeo', tau=1, temperature_decay='alpha', sliding_wnd_size=50,
                 **kwargs):
        super().__init__(id_, action_num, env, **kwargs)
        self.name = 'ROMMEO'
        self.phi_type = phi_type
        self.pi_history = [deepcopy(self.pi)]
        self.Q = defaultdict(partial(np.random.rand, *(self.action_num, self.action_num)))
        self.tau = tau
        self.init_tau = tau
        self.sliding_wnd_size = sliding_wnd_size
        self.temperature_decay = temperature_decay

        self.pi_neg_i = defaultdict(
            partial(np.random.dirichlet, [1.0] * self.action_num)
        )
        self.opponent_pi_history = [deepcopy(self.pi_neg_i)]
        self.replay_buffer = []
        self.cond_pi = defaultdict(
            partial(np.random.rand, *(self.action_num, action_num))
        )
        self.rho = defaultdict(
            partial(np.random.dirichlet, [1.0] * self.action_num)
        )
        self.rho_history = [deepcopy(self.rho)]

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
        self.opponent_pi_history.append(deepcopy(self.pi_neg_i))
        self.rho_history.append(deepcopy(self.rho))
        self.epoch += 1

    def compute_conditional_pi(self, s):
        self.cond_pi[s] = np.exp(self.Q[s] / self.tau)
        self.cond_pi[s] /= np.sum(np.exp(self.Q[s] / self.tau), axis=0)
        return self.cond_pi[s]

    def compute_opponent_model(self, s):
        # self.rho[s] = np.multiply(self.pi_neg_i[s], np.sum(np.exp(self.Q[s] / self.tau), axis=0))
        # self.rho[s] = np.power(np.multiply(self.pi_neg_i[s], np.sum(np.exp(self.Q[s] / self.tau), axis=0)), self.tau)
        self.rho[s] = np.multiply(self.pi_neg_i[s], np.power(np.sum(np.exp(self.Q[s] / self.tau), axis=0), self.tau))
        self.rho[s] /= self.rho[s].sum()
        return self.rho[s]

    def compute_marginal_pi(self, s):
        pi = self.compute_conditional_pi(s)
        if self.phi_type == 'independent':
            rho = np.ones(self.action_num) / np.sum(np.ones(self.action_num))
        elif self.phi_type == 'vanilla':
            rho = self.pi_neg_i[s]
        elif self.phi_type == 'rommeo':
            rho = self.compute_opponent_model(s)
        else:
            raise ValueError('Unrecognized opponent model learning type')
        self.pi[s] = np.sum(np.multiply(pi, rho), 1)
        return self.pi[s]

    def update_policy(self, k=1, gamma=0.95, done=True):
        # sliding_wnd_size = min(self.sliding_wnd_size, len(self.replay_buffer))
        # sliding_window = self.replay_buffer[-sliding_wnd_size:]
        # sample = sliding_window[np.random.choice(len(sliding_window), size=1)[0]]

        sample = self.replay_buffer[-1]
        s, a_i, a_neg_i, s_prime, r = sample
        numerator, denominator = 0, 0
        decay_alpha = self.step_decay()
        if self.temperature_decay == 'alpha':
            self.tau = self.init_tau * self.step_decay()
        elif self.temperature_decay == 'exp':
            self.tau = np.exp(-0.1*self.epoch)*self.init_tau+0.1
        elif self.temperature_decay == 'constant':
            self.tau = self.init_tau
        else:
            raise ValueError('unrecognized decay style for temperature')
        if not done:
            for _ in range(k):
                sampled_a_i, sampled_a_neg_i = self.act(s,exploration=False, game='game', return_pred_opp=True)
                numerator += (
                    self.pi_neg_i[s_prime][sampled_a_neg_i] *
                    np.exp(self.Q[s_prime][sampled_a_i, sampled_a_neg_i])
                )
                pi = self.compute_conditional_pi(s_prime)[sampled_a_i, sampled_a_neg_i]
                rho = self.compute_opponent_model(s_prime)[sampled_a_neg_i]
                denominator += (pi * rho)

            v_s_prime = np.log((1 / k) * (numerator / denominator))

            y = r + gamma * v_s_prime * (1 - done)
        else:
            y = r
        self.Q[s][a_i, a_neg_i] = (
            (1 - decay_alpha) * self.Q[s][a_i, a_neg_i] +
            decay_alpha * y
        )
        self.compute_marginal_pi(s)

    def act(self, s, exploration, game, return_pred_opp=False):
        """
        Function act sample actions from pi for a given state.
        Input:
            s: Int representing a state.
        Returns:
            Int: Sampled action for agent i,
                 Sampled action for the opponent according to our
                 belief.
        """
        opponent_p = self.compute_opponent_model(s)
        # print(opponent_p)
        opponent_action = np.random.choice(
            opponent_p.size, size=1, p=opponent_p)[0]
        # agent_p = np.exp(self.Q[s][:, opponent_action])
        agent_p = self.compute_marginal_pi(s)
        if exploration and random.random() < self.episilon:
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
            agent_action = StationaryAgent.sample(agent_p)
        if return_pred_opp:
            return agent_action, opponent_action
        else:
            return agent_action