# Created by yingwen at 2019-03-10

from collections import OrderedDict
from contextlib import contextmanager
import numpy as np
from scipy.stats import poisson
import tensorflow as tf
import tensorflow_probability as tfp
from malib.distributions.squash_bijector import SquashBijector

from .base_policy import Policy


SCALE_DIAG_MIN_MAX = (-20, 2)
LOG_STD_MAX = 2
LOG_STD_MIN = -20


class LevelKPolicy(Policy):
    def __init__(self,
                 main_policy,
                 secondary_policy,
                 prior_policy,
                 secondary_prior_policy,
                 k=2,
                 k_weights=[1., 1.],
                 k_sampler=lambda x: np.random.randint(0, x+1, 1)[0],
                 preprocessor=None,
                 mu=0, # mean for possion
                 deterministic=False,
                 name='level_k_policy',
                 *args,
                 **kwargs):
        self._Serializable__initialize(locals())

        self._main_policy = main_policy
        self._secondary_policy = secondary_policy
        self._prior_policy = prior_policy
        self._secondary_prior_policy = secondary_prior_policy
        self._k = k
        self._mu = mu
        self._k_weights = k_weights
        self._k_sampler = k_sampler
        self._deterministic = deterministic
        self._name = name
        self._preprocessor = preprocessor

        super(LevelKPolicy, self).__init__(*args, **kwargs)

    @contextmanager
    def set_k(self, k=1):
        """Context manager for changing the determinism of the policy.
        Args:
            set_deterministic (`bool`): Value to set the self._is_deterministic
                to during the context. The value will be reset back to the
                previous value when the context exits.
        """
        was_k = self._k
        self._dk = k
        yield
        self._k = was_k

    def level_distribution(self, k, mu):
        _dists = np.array([poisson.pmf(kk, mu) for kk in range(1, k+1)])
        return _dists / np.sum(_dists)

    def sample_k(self, k, mu):
        return np.random.choice(range(1, k+1), 1, p=self.level_distribution(k, mu))[0]

    def get_weights(self):
        return self._main_policy.get_weights()

    def set_weights(self, *args, **kwargs):
        return self._main_policy.set_weights(*args, **kwargs)

    @property
    def trainable_variables(self):
        return self._main_policy.trainable_variables

    @property
    def non_trainable_weights(self):
        """Due to our nested model structure, we need to filter duplicates."""
        return []

    def get_policy_stack(self, k):
        policy_stack = []
        if k % 2 == 0:
            policy = self._secondary_prior_policy
        else:
            policy = self._prior_policy
        policy_stack.append(policy)
        for i in range(1, k + 1):
            if k % 2 == 0:
                if i % 2 == 1:
                    policy = self._secondary_policy
                else:
                    policy = self._main_policy
            else:
                if i % 2 == 1:
                    policy = self._main_policy
                else:
                    policy = self._secondary_policy
            policy_stack.append(policy)
        return policy_stack

    def get_actions(self, conditions, k):
        all_actions = []
        policy_stack = self.get_policy_stack(k)
        actions = policy_stack[0].get_actions(conditions)
        all_actions.append(actions)
        if k == 0:
            return [actions]
        for policy in policy_stack[1:]:
            with policy.set_deterministic(True):
                actions = policy.get_actions([conditions, actions])
            all_actions.append(actions)
        return all_actions

    def get_action(self, condition, k):
        return self.get_actions(condition[None], k)[0]

    def log_pis(self, conditions, k, all_actions):
        all_log_pis = []
        policy_stack = self.get_policy_stack(k)
        for i, policy in enumerate(policy_stack):
            actions = all_actions[i]
            if i == 0:
                log_pis = policy.log_pis_model([conditions, actions])
                all_log_pis.append(log_pis)
                continue
            input_actions = all_actions[i - 1]
            conditions = [conditions, input_actions]
            log_pis = policy.log_pis_model([*conditions, actions])
            all_log_pis.append(log_pis)
        return all_log_pis

    def get_action_np(self, condition, k):
        all_action = self.get_actions_np(condition[None], k)
        return list([actions[0] for actions in all_action])

    def get_actions_np(self, conditions, k):
        all_actions = self.get_actions(conditions, k)
        all_actions = list([actions.numpy() for actions in all_actions])
        return all_actions

    def log_pis_np(self, conditions, k, all_actions):
        all_log_pis_np = list(log_pis.numpy() for log_pis in self.log_pis(conditions, k, all_actions))
        return all_log_pis_np

    def get_diagnostics(self, conditions):
        """Return diagnostic information of the policy.
        Returns the mean, min, max, and standard deviation of means and
        covariances.
        """
        pass
        # (shifts_np,
        #  log_scale_diags_np,
        #  log_pis_np,
        #  raw_actions_np,
        #  actions_np) = self.diagnostics_model.predict(conditions)
        #
        # return OrderedDict({
        #     '{}/shifts-mean'.format(self._name): np.mean(shifts_np),
        #     '{}/shifts-std'.format(self._name): np.std(shifts_np),
        #
        #     '{}/log_scale_diags-mean'.format(self._name): np.mean(log_scale_diags_np),
        #     '{}/log_scale_diags-std'.format(self._name): np.std(log_scale_diags_np),
        #
        #     '{}/-log-pis-mean'.format(self._name): np.mean(-log_pis_np),
        #     '{}/-log-pis-std'.format(self._name): np.std(-log_pis_np),
        #
        #     '{}/raw-actions-mean'.format(self._name): np.mean(raw_actions_np),
        #     '{}/raw-actions-std'.format(self._name): np.std(raw_actions_np),
        #
        #     '{}/actions-mean'.format(self._name): np.mean(actions_np),
        #     '{}/actions-std'.format(self._name): np.std(actions_np),
        # })