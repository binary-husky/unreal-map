# Created by yingwen at 2019-03-10
import gym
import numpy as np


def flat_dim(space):
    if isinstance(space, gym.spaces.Box):
        return np.prod(space.low.shape)
    elif isinstance(space, gym.spaces.Discrete):
        return space.n
    elif isinstance(space, gym.spaces.Dict):
        return np.sum([flat_dim(v) for v in space.spaces.values()])
    else:
        return np.sum([flat_dim(x) for x in space.spaces])


def flatten(space, obs):
    if isinstance(space, gym.spaces.Box):
        return np.asarray(obs).flatten()
    elif isinstance(space, gym.spaces.Discrete):
        if space.n == 2:
            obs = int(obs)
        return to_onehot(obs, space.n)
    elif isinstance(space, gym.spaces.Dict):
        return np.concatenate(
            [flatten(space.spaces[key], obs[key]) for key in space.spaces.keys()])
    else:
        return np.concatenate(
            [flatten(c, xi) for c, xi in zip(space.spaces, obs)])


def flatten_n(space, obs):
    if isinstance(space, gym.spaces.Box):
        obs = np.asarray(obs)
        return obs.reshape((obs.shape[0], -1))
    elif isinstance(space, gym.spaces.Discrete):
        return to_onehot_n(obs, space.n)
    else:
        obs_regrouped = [[obs[i] for o in obs] for i in range(len(obs[0]))]
        flat_regrouped = [
            flatten_n(c, oi) for c, oi in zip(space.spaces, obs_regrouped)
        ]
        return np.concatenate(flat_regrouped, axis=-1)


def unflatten(space, obs):
    if isinstance(space, gym.spaces.Box):
        return np.asarray(obs).reshape(space.shape)
    elif isinstance(space, gym.spaces.Discrete):
        return from_onehot(obs)
    else:
        dims = [flat_dim(c) for c in space.spaces]
        flat_xs = np.split(obs, np.cumsum(dims)[:-1])
        return tuple(unflatten(c, xi) for c, xi in zip(space.spaces, flat_xs))


def unflatten_n(space, obs):
    if isinstance(space, gym.spaces.Box):
        obs = np.asarray(obs)
        return obs.reshape((obs.shape[0], ) + space.shape)
    elif isinstance(space, gym.spaces.Discrete):
        return from_onehot_n(obs)
    else:
        dims = [flat_dim(c) for c in space.spaces]
        flat_xs = np.split(obs, np.cumsum(dims)[:-1], axis=-1)
        unflat_xs = [
            unflatten_n(c, xi) for c, xi in zip(space.spaces, flat_xs)
        ]
        unflat_xs_grouped = list(zip(*unflat_xs))
        return unflat_xs_grouped


def to_onehot(ind, dim):
    ret = np.zeros(dim)
    ret[ind] = 1
    return ret


def to_onehot_n(inds, dim):
    ret = np.zeros((len(inds), dim))
    ret[np.arange(len(inds)), inds] = 1
    return ret


def from_onehot(v):
    return np.nonzero(v)[0][0]


def from_onehot_n(v):
    if ((isinstance(v, np.ndarray) and not v.size)
            or (isinstance(v, list) and not v)):
        return []
    return np.nonzero(v)[1]