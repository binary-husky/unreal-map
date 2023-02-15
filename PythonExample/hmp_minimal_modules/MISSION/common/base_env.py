import numpy as np
class BaseEnv(object):
    def __init__(self, rank) -> None:
        self.observation_space = None
        self.action_space = None
        self.rank = rank

    def step(self, act):
        # obs: a Tensor with shape (n_agent, ...)
        # reward: a Tensor with shape (n_agent, 1) or (n_team, 1)
        # done: a Bool
        # info: a dict
        raise NotImplementedError
        # Warning: if you have only one team and RewardAsUnity, 
        # you must make sure that reward has shape=[n_team=1, 1]
        # e.g. 
        # >> RewardForTheOnlyTeam = +1
        # >> RewardForAllTeams = np.array([RewardForTheOnlyTeam, ])
        # >> return (ob, RewardForAllTeams, done, info)
        return (ob, RewardForAllTeams,  done, info)  # choose this if RewardAsUnity
        return (ob, RewardForAllAgents, done, info)  # choose this if not RewardAsUnity

    def reset(self):
        # obs: a Tensor with shape (n_agent, ...)
        # info: a dict
        raise NotImplementedError
        return ob, info


class RawObsArray(object):
    raw_obs_size = {}   # shared
    def __init__(self, key='default'):
        self.key = key
        if self.key not in self.raw_obs_size:
            self.guards_group = []
            self.nosize = True
        else:
            self.guards_group = np.zeros(shape=(self.raw_obs_size[self.key]), dtype=np.float32)
            self.nosize = False
            self.p = 0

    def append(self, buf):
        if self.nosize:
            self.guards_group.append(buf)
        else:
            L = len(buf)
            self.guards_group[self.p:self.p+L] = buf[:]
            self.p += L

    def get(self):
        if self.nosize:
            self.guards_group = np.concatenate(self.guards_group)
            self.raw_obs_size[self.key] = len(self.guards_group)
        return self.guards_group
        
    def get_raw_obs_size(self):
        assert self.key in self.raw_obs_size > 0
        return self.raw_obs_size[self.key]