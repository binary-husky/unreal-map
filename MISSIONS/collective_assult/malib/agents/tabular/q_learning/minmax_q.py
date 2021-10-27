# Created by yingwen at 2019-03-10

import numpy as np
from copy import deepcopy
from collections import defaultdict
from functools import partial
from malib.agents.tabular.q_learning.base_q import BaseQAgent, QAgent
from malib.agents.tabular.q_learning.base_tabular_agent import StationaryAgent, Agent
import importlib


class MinMaxQAgent(BaseQAgent):
    def __init__(self, id_, action_num, env, opp_action_num):
        super().__init__('minmax', id_, action_num, env)
        self.solvers = []
        self.opp_action_num = opp_action_num
        self.pi_history = [deepcopy(self.pi)]
        self.Q = defaultdict(partial(np.random.rand, self.action_num, self.opp_action_num))

    def done(self, env):
        self.solvers.clear()
        super().done(env)

    def val(self, s):
        Q = self.Q[s]
        pi = self.pi[s]
        return min(np.dot(pi, Q[:, o]) for o in range(self.opp_action_num))

    def update(self, s, a, o, r, s2, env):
        Q = self.Q[s]
        V = self.val(s2)
        decay_alpha = self.step_decay()
        Q[a, o] = Q[a, o] + decay_alpha * (r + self.gamma * V - Q[a, o])
        self.update_policy(s, a, env)
        self.record_policy(s, env)

    def update_policy(self, s, a, env):
        self.initialize_solvers()
        for solver, lib in self.solvers:
            try:
                self.pi[s] = self.lp_solve(self.Q[s], solver, lib)
                StationaryAgent.normalize(self.pi[s])
                self.pi_history.append(deepcopy(self.pi))
            except Exception as e:
                print('optimization using {} failed: {}'.format(solver, e))
                continue
            else: break

    def initialize_solvers(self):
        if not self.solvers:
            for lib in ['gurobipy', 'scipy.optimize', 'pulp']:
                try: self.solvers.append((lib, importlib.import_module(lib)))
                except: pass

    def lp_solve(self, Q, solver, lib):
        ret = None

        if solver == 'scipy.optimize':
            c = np.append(np.zeros(self.action_num), -1.0)
            A_ub = np.c_[-Q.T, np.ones(self.opp_action_num)]
            b_ub = np.zeros(self.opp_action_num)
            A_eq = np.array([np.append(np.ones(self.action_num), 0.0)])
            b_eq = np.array([1.0])
            bounds = [(0.0, 1.0) for _ in range(self.action_num)] + [(-np.inf, np.inf)]
            res = lib.linprog(
                c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
            ret = res.x[:-1]
        elif solver == 'gurobipy':
            m = lib.Model('LP')
            m.setParam('OutputFlag', 0)
            m.setParam('LogFile', '')
            m.setParam('LogToConsole', 0)
            v = m.addVar(name='v')
            pi = {}
            for a in range(self.action_num):
                pi[a] = m.addVar(lb=0.0, ub=1.0, name='pi_{}'.format(a))
            m.update()
            m.setObjective(v, sense=lib.GRB.MAXIMIZE)
            for o in range(self.opp_action_num):
                m.addConstr(
                    lib.quicksum(pi[a] * Q[a, o] for a in range(self.action_num)) >= v,
                    name='c_o{}'.format(o))
            m.addConstr(lib.quicksum(pi[a] for a in range(self.action_num)) == 1, name='c_pi')
            m.optimize()
            ret = np.array([pi[a].X for a in range(self.action_num)])
        elif solver == 'pulp':
            v = lib.LpVariable('v')
            pi = lib.LpVariable.dicts('pi', list(range(self.action_num)), 0, 1)
            prob = lib.LpProblem('LP', lib.LpMaximize)
            prob += v
            for o in range(self.opp_action_num):
                prob += lib.lpSum(pi[a] * Q[a, o] for a in range(self.action_num)) >= v
            prob += lib.lpSum(pi[a] for a in range(self.action_num)) == 1
            prob.solve(lib.GLPK_CMD(msg=0))
            ret = np.array([lib.value(pi[a]) for a in range(self.action_num)])

        if not (ret >= 0.0).all():
            raise Exception('{} - negative probability error: {}'.format(solver, ret))

        return ret


class MetaControlAgent(Agent):
    def __init__(self, id_, action_num, env, opp_action_num):
        super().__init__('metacontrol', id_, action_num, env)
        self.agents = [QAgent(id_, action_num, env), MinimaxQAgent(id_, action_num, env, opp_action_num)]
        self.n = np.zeros(len(self.agents))
        self.controller = None

    def act(self, s, exploration, env):
        print([self.val(i, s) for i in range(len(self.agents))])
        self.controller = np.argmax([self.val(i, s) for i in range(len(self.agents))])
        return self.agents[self.controller].act(s, exploration, env)

    def done(self, env):
        for agent in self.agents:
            agent.done(env)

    def val(self, i, s):
        return self.agents[i].val(s)

    def update(self, s, a, o, r, s2, env):
        for agent in self.agents:
            agent.update(s, a, o, r, s2, env)

        self.n[self.controller] += 1
        print('id: {}, n: {} ({}%)'.format(self.id_, self.n, 100.0 * self.n / np.sum(self.n)))
