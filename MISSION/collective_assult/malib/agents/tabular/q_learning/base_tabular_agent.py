from abc import ABCMeta, abstractmethod
import numpy as np


class Agent(object):
    __metaclass__ = ABCMeta

    def __init__(self, name, id_, action_num, env):
        self.name = name
        self.id_ = id_
        self.action_num = action_num
        # len(env.action_space[id_])
        # self.opp_action_space = env.action_space[0:id_] + env.action_space[id_:-1]

    def set_pi(self, pi):
        # assert len(pi) == self.actin_num
        self.pi = pi

    def done(self, env):
        pass

    @abstractmethod
    def act(self, s, exploration, env):
        pass

    def update(self, s, a, o, r, s2, env):
        pass

    @staticmethod
    def format_time(n):
        return ''
        # s = humanfriendly.format_size(n)
        # return s.replace(' ', '').replace('bytes', '').replace('byte', '').rstrip('B')

    def full_name(self, env):
        return '{}_{}_{}'.format(env.name, self.name, self.id_)


class StationaryAgent(Agent):
    def __init__(self, id_, action_num, env, pi=None):
        super().__init__('stationary', id_, action_num, env)
        if pi is None:
            pi = np.random.dirichlet([1.0] * self.action_num)
        self.pi = np.array(pi, dtype=np.double)
        StationaryAgent.normalize(self.pi)

    def act(self, s, exploration, env):
        if self.verbose:
            print('pi of agent {}: {}'.format(self.id_, self.pi))
        return StationaryAgent.sample(self.pi)

    @staticmethod
    def normalize(pi):
        minprob = np.min(pi)
        if minprob < 0.0:
            pi -= minprob
        pi /= np.sum(pi)

    @staticmethod
    def sample(pi):
        return np.random.choice(pi.size, size=1, p=pi)[0]


class RandomAgent(StationaryAgent):
    def __init__(self, id_, action_num, env):
        assert action_num > 0
        super().__init__(id_, env, action_num, pi=[1.0 / action_num] * action_num)
        self.name = 'random'