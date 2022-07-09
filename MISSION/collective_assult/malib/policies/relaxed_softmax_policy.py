# Created by yingwen at 2019-03-10

# TODO: gumble softmax policy

# Created by yingwen at 2019-03-10

from collections import OrderedDict

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from malib.networks.mlp import MLP
from .base_policy import StochasticPolicy

class RelaxedSoftmaxPolicy(StochasticPolicy):
    def __init__(self, input_shapes, output_shape, temperature=.5, preprocessor=None, name='gaussian_policy', repara=True,
                 *args,
                 **kwargs):
        """

        Args:
            input_shapes:
            output_shape:
            squash:
            preprocessor:
            name:
            *args:
            **kwargs:
        """
        self._Serializable__initialize(locals())

        self._input_shapes = input_shapes
        self._output_shape = output_shape
        self._temperature = tf.constant(temperature, dtype=tf.float32)
        self._name = name
        self._preprocessor = preprocessor
        self._repara = repara

        super(RelaxedSoftmaxPolicy, self).__init__(*args, **kwargs)

        self.condition_inputs = [tf.keras.layers.Input(shape=input_shape) for input_shape in input_shapes]

        conditions = tf.keras.layers.Lambda(lambda x: tf.concat(x, axis=-1))(self.condition_inputs)

        if preprocessor is not None:
            conditions = preprocessor(conditions)

        self._logits_model = self._logits_net(
            input_shapes=(conditions.shape[1:],), output_size=output_shape[0], )

        logits = self._logits_model(conditions)

        self.logits_model = tf.keras.Model(self.condition_inputs, logits)


        def action_fn(inputs):
            temperature, logits =  inputs
            return tfp.distributions.RelaxedOneHotCategorical(temperature, logits=logits).sample()
        # actions_dist =

        actions = tf.keras.layers.Lambda(action_fn)((self._temperature, logits))

        self.actions_model = tf.keras.Model(self.condition_inputs, actions, name=self._name)

        deterministic_actions = tf.keras.layers.Lambda(lambda logits: tf.math.softmax(logits))(logits)

        self.deterministic_actions_model = tf.keras.Model(self.condition_inputs, deterministic_actions)

        def log_pis_fn(inputs):
            temperature, logits, actions = inputs
            actions_dist = tfp.distributions.RelaxedOneHotCategorical(temperature, logits=logits)
            log_pis = actions_dist.log_prob(actions)[:, None]
            return log_pis

        self.actions_input = tf.keras.layers.Input(shape=output_shape)

        log_pis = tf.keras.layers.Lambda(log_pis_fn)([self._temperature, logits, actions])

        log_pis_for_action_input = tf.keras.layers.Lambda(log_pis_fn)([self._temperature,  logits, self.actions_input])

        self.log_pis_model = tf.keras.Model((*self.condition_inputs, self.actions_input), log_pis_for_action_input)

        self.diagnostics_model = tf.keras.Model(self.condition_inputs, (logits, log_pis, actions))

    def _logits_net(self, input_shapes, output_size):
        raise NotImplementedError

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
        return list(set(super(RelaxedSoftmaxPolicy, self).non_trainable_weights))

    def get_raw_and_actions(self, conditions):
        if self._deterministic:
            return self.deterministic_raw_and_actions_model(conditions)
        return self.raw_and_actions_model(conditions)

    def get_actions(self, conditions):
        if self._deterministic:
            return self.deterministic_actions_model(conditions)
        return self.actions_model(conditions)

    def get_action(self, condition, extend_dim=True):
        if extend_dim:
            if isinstance(condition, list):
                for i in range(len(condition)):
                    condition[i] = condition[i][None]
            else:
                condition = condition[None]
        return self.get_actions(condition)[0]

    def log_pis(self, conditions, actions):
        assert not self._deterministic
        # print(conditions, actions)
        return self.log_pis_model([*conditions, actions])

    def get_action_np(self, condition, extend_dim=True):
        if extend_dim:
            if isinstance(condition, list):
                for i in range(len(condition)):
                    condition[i] = condition[i][None]
            else:
                condition = condition[None]
        return self.get_actions_np(condition)[0]

    def get_actions_np(self, conditions):
        if self._deterministic:
            actions = self.deterministic_actions_model.predict(conditions)
            # print('get_actions_np _deterministic'+self._name, actions)
            return actions
        actions = self.actions_model.predict(conditions)
        # print('get_actions_np non-deterministic'+self._name, actions)
        return actions

    def log_pis_np(self, conditions, actions):
        return self.log_pis_model.predict([*conditions, actions])

    def get_model_regularization_loss(self):
        return tf.reduce_mean(self.logits_model.losses)

    def get_diagnostics(self, conditions):
        """Return diagnostic information of the policy.
        Returns the mean, min, max, and standard deviation of means and
        covariances.
        """
        (logits_np, log_pis_np, actions_np) = self.diagnostics_model.predict(
            conditions)

        return OrderedDict({'{}/logits-mean'.format(self._name): np.mean(logits_np),
                            '{}/logits-std'.format(self._name): np.std(logits_np),

                            '{}/-log-pis-mean'.format(self._name): np.mean(-log_pis_np),
                            '{}/-log-pis-std'.format(self._name): np.std(-log_pis_np),

                            '{}/actions-mean'.format(self._name): np.mean(actions_np),
                            '{}/actions-std'.format(self._name): np.std(actions_np), })


class RelaxedSoftmaxMLPPolicy(RelaxedSoftmaxPolicy):
    def __init__(self, hidden_layer_sizes, activation=tf.keras.layers.ReLU(), output_activation='linear',
                 pi_std=None, *args, **kwargs):
        self._hidden_layer_sizes = hidden_layer_sizes
        self._activation = activation
        self._output_activation = output_activation

        self._pi_std = pi_std

        self._Serializable__initialize(locals())
        super(RelaxedSoftmaxMLPPolicy, self).__init__(*args, **kwargs)

    def _logits_net(self, input_shapes, output_size):
        logits = MLP(input_shapes=input_shapes, hidden_layer_sizes=self._hidden_layer_sizes,
                                           output_size=output_size, activation=self._activation,
                                           output_activation=self._output_activation,
                                           name='{}/GaussianMLPPolicy'.format(self._name),
                                           kernel_regularizer=tf.keras.regularizers.l2(0.001),
                                           bias_regularizer=tf.keras.regularizers.l2(0.001), )
        return logits
