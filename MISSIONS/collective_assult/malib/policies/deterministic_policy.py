# Created by yingwen at 2019-03-10


from collections import OrderedDict

import numpy as np
import tensorflow as tf

from malib.networks.mlp import MLP
from malib.policies.base_policy import Policy


class DeterministicPolicy(Policy):
    def __init__(self,
                 input_shapes,
                 output_shape,
                 squash=True,
                 preprocessor=None,
                 name='DeterministicPolicy',
                 *args,
                 **kwargs):
        self._Serializable__initialize(locals())
        self._input_shapes = input_shapes
        self._output_shape = output_shape
        self._squash = squash
        self._name = name
        self._preprocessor = preprocessor
        super(DeterministicPolicy, self).__init__(*args, **kwargs)

        self.condition_inputs = [
            tf.keras.layers.Input(shape=input_shape)
            for input_shape in input_shapes
        ]

        conditions = tf.keras.layers.Lambda(
            lambda x: tf.concat(x, axis=-1)
        )(self.condition_inputs)

        if preprocessor is not None:
            conditions = preprocessor(conditions)
        raw_actions = self._policy_net(
            input_shapes=(conditions.shape[1:],),
            output_size=output_shape[0],
        )(conditions)
        actions = raw_actions if self._squash else tf.nn.tanh(raw_actions)
        self.actions_model = tf.keras.Model(self.condition_inputs, actions)
        self.diagnostics_model = tf.keras.Model(self.condition_inputs, (raw_actions, actions))

    def get_actions(self, conditions):
        return self.actions_model(conditions)

    def get_action(self, condition, extend_dim=True):
        if extend_dim:
            condition = condition[None]
        return self.get_actions(condition)[0]

    def get_action_np(self, condition, extend_dim=True):
        # if type(condition) is list:
        #     extend_dim = False
        if extend_dim:
            condition = condition[None]
        return self.get_actions_np(condition)[0]

    def get_actions_np(self, conditions):
        return self.actions_model.predict(conditions)

    def _policy_net(self, input_shapes, output_size):
        raise NotImplementedError

    def reset(self):
        pass

    def get_weights(self):
        return self.actions_model.get_weights()

    def set_weights(self, *args, **kwargs):
        return self.actions_model.set_weights(*args, **kwargs)

    @property
    def trainable_variables(self):
        return self.actions_model.trainable_variables

    @property
    def non_trainable_weights(self):
        """Due to our nested model structure, we need to filter duplicates."""
        return list(set(super(DeterministicPolicy, self).non_trainable_weights))

    def get_diagnostics(self, conditions):
        """Return diagnostic information of the policy.
        Returns the mean, min, max, and standard deviation of means and
        covariances.
        """
        (raw_actions_np,
         actions_np) = self.diagnostics_model.predict(conditions)

        return OrderedDict({
            '{}/raw-actions-mean'.format(self._name): np.mean(raw_actions_np),
            '{}/raw-actions-std'.format(self._name): np.std(raw_actions_np),

            '{}/actions-mean'.format(self._name): np.mean(actions_np),
            '{}/actions-std'.format(self._name): np.std(actions_np),
        })


class DeterministicMLPPolicy(DeterministicPolicy):
    def __init__(self,
                 hidden_layer_sizes,
                 activation='relu',
                 output_activation='linear',
                 *args, **kwargs):
        self._hidden_layer_sizes = hidden_layer_sizes
        self._activation = activation
        self._output_activation = output_activation

        self._Serializable__initialize(locals())
        super(DeterministicMLPPolicy, self).__init__(*args, **kwargs)

    def _policy_net(self, input_shapes, output_size):
        raw_actions = MLP(
            input_shapes=input_shapes,
            hidden_layer_sizes=self._hidden_layer_sizes,
            output_size=output_size,
            activation=self._activation,
            output_activation=self._output_activation,
            name='{}/GaussianMLPPolicy'.format(self._name)
        )
        return raw_actions