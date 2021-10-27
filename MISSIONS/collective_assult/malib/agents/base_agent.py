import tensorflow as tf
from malib.core import Serializable
from malib.utils import tf_utils


class RLAgent:
    def act(self, observation):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError


class OffPolicyAgent(RLAgent, Serializable):
    """This class implements OffPolicyRLAgents."""

    def __init__(
            self,
            observation_space,
            action_space,
            policy,
            qf,
            replay_buffer,
            train_sequence_length, # for rnn policy
            name='off_policy_agent',
            *args,
            **kwargs):
        self._Serializable__initialize(locals())
        self._observation_space = observation_space
        self._action_space = action_space
        self._policy = policy
        self._qf = qf
        self._replay_buffer = replay_buffer
        self._train_sequence_length = train_sequence_length
        self._name = name
        self.init_opt()

    def log_diagnostics(self, batch):
        """Log diagnostic information on current batch."""
        self.policy.log_diagnostics(batch)
        self.qf.log_diagnostics(batch)

    def init_opt(self):
        """
        Initialize the optimization procedure.
        If using tensorflow, this may
        include declaring all the variables and compiling functions.
        """
        raise NotImplementedError

    def init_eval(self):
        """
        Initialize the evaluation procedure.
        """
        raise NotImplementedError


    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def replay_buffer(self):
        return self._replay_buffer

    @property
    def policy(self):
        """Return the current policy held by the agent.
        Returns:
          A Policy object.
        """
        return self._policy

    @property
    def qf(self):
        """Return the current policy held by the agent.
        Returns:
          A Policy object.
        """
        return self._qf

    @property
    def train_sequence_length(self):
        return self._train_sequence_length

    def train(self, batch, weights=None):
        if self.train_sequence_length is not None:
            if batch['observations'].shape[1] != self.train_sequence_length:
                raise ValueError('Invalid sequence length in batch.')
        loss_info = self._train(batch=batch, weights=weights)
        return loss_info

    def _train(self, batch, weights):
        raise NotImplementedError

    # def __getstate__(self):
    #     state = Serializable.__getstate__(self)
    #     # state['pickled_qf'] = self._qf
    #     # state['pickled_policy'] = self._policy
    #     # state['pickled_qf'] = self._qf
    #     # state['pickled_policy'] = self._policy
    #     return state
    #
    # def __setstate__(self, state):
    #     Serializable.__setstate__(self, state)
    #     # self._qf = state['pickled_qf']
    #     # self._policy = state['pickled_policy']
    #     # self.set_weights(state['pickled_weights'])

