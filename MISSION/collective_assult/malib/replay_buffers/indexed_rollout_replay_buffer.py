import numpy as np

from malib.core import Serializable


class IndexedRolloutReplayBuffer(Serializable):
    def __init__(self, observation_dim, action_dim, opponent_action_dim=None, max_rollout_length=50,
                 max_replay_buffer_size=100):
        self._Serializable__initialize(locals())

        self._max_buffer_size = int(max_replay_buffer_size)
        self._max_rollout_length = int(max_rollout_length)
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._opponent_action_dim = opponent_action_dim

        self._observations = np.zeros((self._max_buffer_size, self._max_rollout_length, self._observation_dim),
                                      dtype=np.float32)
        self._next_obs = np.zeros((self._max_buffer_size, self._max_rollout_length, self._observation_dim),
                                  dtype=np.float32)
        self._actions = np.zeros((self._max_buffer_size, self._max_rollout_length, self._action_dim), dtype=np.float32)
        self._rewards = np.zeros((self._max_buffer_size, self._max_rollout_length), dtype=np.float32)
        self._terminals = np.zeros((self._max_buffer_size, self._max_rollout_length), dtype='uint8')
        if self._opponent_action_dim is not None:
            self._opponent_actions = np.zeros(
                (self._max_buffer_size, self._max_rollout_length, self._opponent_action_dim), dtype=np.float32)

        # current episode
        self._cur_observations = np.zeros((self._max_rollout_length, self._observation_dim), dtype=np.float32)
        self._cur_next_obs = np.zeros((self._max_rollout_length, self._observation_dim), dtype=np.float32)
        self._cur_actions = np.zeros((self._max_rollout_length, self._action_dim), dtype=np.float32)
        self._cur_rewards = np.zeros(self._max_rollout_length, dtype=np.float32)
        self._cur_terminals = np.zeros(self._max_rollout_length, dtype='uint8')
        if self._opponent_action_dim is not None:
            self._cur_opponent_actions = np.zeros((self._max_rollout_length, self._opponent_action_dim),
                                                  dtype=np.float32)

        self._top = 0
        self._size = 0
        self._cur_top = 0

    def add_sample(self, observation, action, reward, terminal, next_observation, **kwargs):
        self._cur_observations[self._cur_top] = observation
        self._cur_actions[self._cur_top] = action
        self._cur_rewards[self._cur_top] = reward
        self._cur_terminals[self._cur_top] = terminal
        self._cur_next_obs[self._cur_top] = next_observation
        if 'opponent_action' in kwargs and self._opponent_action_dim is not None:
            self._cur_opponent_actions[self._cur_top] = kwargs['opponent_action']
        self._cur_advance()

    def _add_rollout(self, observations, actions, rewards, terminals, next_observations, **kwargs):
        # episode length check.
        assert observations.shape[0] == actions.shape[0] == rewards.shape[0] == terminals.shape[0] == \
               next_observations.shape[0]
        opponent_actions = None
        if 'opponent_actions' in kwargs and self._opponent_action_dim is not None:
            opponent_actions = kwargs['opponent_actions']
            assert opponent_actions.shape[0] == observations.shape[0]

        rollout_length = observations.shape[0]
        self._observations[self._top, :rollout_length, :] = observations
        self._actions[self._top, :rollout_length, :] = actions
        self._rewards[self._top, :rollout_length] = rewards
        self._terminals[self._top, :rollout_length] = terminals
        self._next_obs[self._top, :rollout_length, :] = next_observations
        if opponent_actions is not None:
            self._opponent_actions[self._top, :rollout_length, :] = opponent_actions
        self._advance()

    def terminate_episode(self):
        pass

    def _advance(self):
        self._top = (self._top + 1) % self._max_buffer_size
        if self._size < self._max_buffer_size:
            self._size += 1

    def _cur_advance(self):
        # if current rollout terminates
        if self._cur_terminals[self._cur_top] == 1 or self._cur_top == self._max_rollout_length - 1:
            self._add_rollout(observations=self._cur_observations[:self._cur_top, ...],
                              actions=self._cur_actions[:self._cur_top, ...],
                              rewards=self._cur_rewards[:self._cur_top, ...],
                              terminals=self._cur_terminals[:self._cur_top, ...],
                              next_observations=self._cur_next_obs[:self._cur_top, ...],
                              opponent_actions=self._cur_opponent_actions[:self._cur_top, ...])
            self._cur_top = 0
        else:
            self._cur_top += 1

    def random_indices(self, batch_size):
        self.indices = np.random.randint(0, self._size, batch_size)
        return self.indices

    def batch_rollout_by_indices(self, indices):
        batch = dict(observations_rollout=self._observations[indices], actions_rollout=self._actions[indices],
                     rewards_rollout=self._rewards[indices], terminals_rollout=self._terminals[indices],
                     next_observations_rollout=self._next_obs[indices], )
        if self._opponent_action_dim is not None:
            batch['opponent_actions'] = self._opponent_actions[indices]
        return batch

    def random_batch_rollout(self, batch_size):
        self.indices = np.random.randint(0, self._size, batch_size)
        return self.batch_rollout_by_indices(self.indices)

    def recent_rollout_batch(self, batch_size):
        self.indices = np.array(list(range(self._top - batch_size, self._top)))
        return self.batch_rollout_by_indices(self.indices)

    def flatten_rollout_batch(self, rollout_batch):
        batch = {}
        for key in rollout_batch.keys():
            val = rollout_batch[key]
            if len(val.shape) == 2:
                flat_val = np.reshape(val, newshape=(-1,))
            else:
                assert len(val.shape) == 3
                flat_val = np.reshape(val, newshape=(-1, val.shape[2]))
            batch[key.replace('_rollout', '')] = flat_val
        return batch

    @property
    def size(self):
        return self._size
