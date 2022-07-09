import numpy as np

from malib.core import Serializable


class IndexedReplayBuffer(Serializable):
    def __init__(self, observation_dim, action_dim, opponent_action_dim=None, max_replay_buffer_size=10e4):
        self._Serializable__initialize(locals())

        self._max_buffer_size = int(max_replay_buffer_size)
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._opponent_action_dim = opponent_action_dim

        self._observations = np.zeros((self._max_buffer_size, self._observation_dim))
        self._next_obs = np.zeros((self._max_buffer_size, self._observation_dim))
        self._actions = np.zeros((self._max_buffer_size, self._action_dim))
        self._rewards = np.zeros(self._max_buffer_size)
        self._terminals = np.zeros(self._max_buffer_size, dtype='uint8')

        if self._opponent_action_dim is not None:
            self._opponent_actions = np.zeros((self._max_buffer_size, self._opponent_action_dim))

        self._top = 0
        self._size = 0

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        self._observations[self._top] = observation
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._next_obs[self._top] = next_observation
        if 'opponent_action' in kwargs and self._opponent_action_dim is not None:
            self._opponent_actions[self._top] = kwargs['opponent_action']
        self._advance()

    def terminate_episode(self):
        pass

    def _advance(self):
        self._top = (self._top + 1) % self._max_buffer_size
        if self._size < self._max_buffer_size:
            self._size += 1

    def random_indices(self, batch_size):
        self.indices = np.random.randint(0, self._size, batch_size)
        return self.indices

    def random_batch(self, batch_size):
        self.indices = np.random.randint(0, self._size, batch_size)
        return self.batch_by_indices(self.indices)

    def recent_batch(self, batch_size):
        self.indices = np.array(list(range(self._top - batch_size, self._top)))
        return self.batch_by_indices(self.indices)

    def batch_by_indices(self, indices):
        batch = dict(
            observations=self._observations[indices].astype(np.float32),
            actions=self._actions[indices].astype(np.float32),
            rewards=self._rewards[indices].astype(np.float32),
            terminals=self._terminals[indices].astype(np.float32),
            next_observations=self._next_obs[indices].astype(np.float32),
        )
        if self._opponent_action_dim is not None:
            batch['opponent_actions'] = self._opponent_actions[indices].astype(np.float32)
        return batch

    @property
    def size(self):
        return self._size

    def __getstate__(self):
        d = super(IndexedReplayBuffer, self).__getstate__()
        d.update(dict(
            o=self._observations.tobytes(),
            a=self._actions.tobytes(),
            r=self._rewards.tobytes(),
            t=self._terminals.tobytes(),
            no=self._next_obs.tobytes(),
            top=self._top,
            size=self._size,
        ))
        if self._opponent_action_dim is not None:
            d.update((dict(o_a=self._opponent_actions.tobytes())))
        return d

    def __setstate__(self, d):
        super(IndexedReplayBuffer, self).__setstate__(d)
        self._observations = np.fromstring(d['o']).reshape(
            self._max_buffer_size, -1
        )
        self._next_obs = np.fromstring(d['no']).reshape(
            self._max_buffer_size, -1
        )
        self._actions = np.fromstring(d['a']).reshape(self._max_buffer_size, -1)
        self._rewards = np.fromstring(d['r']).reshape(self._max_buffer_size)
        self._terminals = np.fromstring(d['t'], dtype=np.uint8)
        self._top = d['top']
        self._size = d['size']
        if self._opponent_action_dim is not None:
            self._opponent_actions = np.fromstring(d['o_a']).reshape(self._max_buffer_size, -1)
