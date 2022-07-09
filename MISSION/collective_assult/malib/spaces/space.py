import numpy as np
from gym.spaces import Space
from malib.spaces import utils


class MASpace(Space):
    """
    A multiagent tuple (i.e., product) of simpler spaces
    Example usage:
    self.observation_space = MASpace((spaces.Discrete(2), spaces.Discrete(3)))
    """
    def __init__(self, spaces):
        self.spaces = spaces
        self.agent_num = len(spaces)

    def seed(self, seed):
        [space.seed(seed) for space in self.spaces]

    def sample(self):
        return np.array([space.sample() for space in self.spaces])

    def contains(self, x):
        if isinstance(x, list):
            x = tuple(x)
        return isinstance(x, tuple) and len(x) == len(self.spaces) and all(
            space.contains(part) for (space, part) in zip(self.spaces, x))


    def to_jsonable(self, sample_n):
        # serialize as list-repr of tuple of vectors
        return [space.to_jsonable([sample[i] for sample in sample_n]) \
                for i, space in enumerate(self.spaces)]

    def from_jsonable(self, sample_n):
        return [sample for sample in zip(*[space.from_jsonable(sample_n[i]) for i, space in enumerate(self.spaces)])]


    def flatten(self, x):
        assert len(x) == self.agent_num
        return np.array([utils.flatten(space, x_i) for x_i, space in zip(x, self.spaces)])

    def unflatten(self, x):
        assert len(x) == self.agent_num
        return np.array([utils.unflatten(space, x_i) for x_i, space in zip(x, self.spaces)])

    def flatten_n(self, xs):
        assert len(xs) == self.agent_num
        return np.array([utils.flatten_n(space, xs_i) for xs_i, space in zip(xs, self.spaces)])

    def unflatten_n(self, xs):
        assert len(xs) == self.agent_num
        return np.array([utils.unflatten_n(space, xs_i) for xs_i, space in zip(xs, self.spaces)])

    def __getitem__(self, i):
        assert (i >= 0) and (i < self.agent_num)
        return self.spaces[i]

    @property
    def flat_dim(self):
        """
        The dimension of the flattened vector of the tensor representation
        """
        return np.sum([utils.flat_dim(x) for x in self.spaces])

    @property
    def shape(self):
        return tuple(x.shape for x in self.spaces)

    def agent_shape(self, i):
        assert i in range(self.agent_num)
        return self.spaces[i].shape

    def opponent_shape(self, i):
        assert i in range(self.agent_num)
        shapes = []
        for agent, space in enumerate(self.spaces):
            if agent != i:
                shapes.append(space.shape)
        return tuple(shapes)

    def opponent_flat_dim(self, i):
        assert i in range(self.agent_num)
        return self.flat_dim - utils.flat_dim(self.spaces[i])

    def agent_flat_dim(self, i):
        assert i in range(self.agent_num)
        return utils.flat_dim(self.spaces[i])

    def __len__(self):
        return len(self.spaces)

    def __eq__(self, other):
        return self.spaces == other.spaces

    def __repr__(self):
        return '\n'.join(["Agent {}, {}".format(i, space.__repr__()) for i, space in zip(range(self.agent_num), self.spaces)])

