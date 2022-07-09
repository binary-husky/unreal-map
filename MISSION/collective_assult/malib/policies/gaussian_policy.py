# Created by yingwen at 2019-03-10

from collections import OrderedDict

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from malib.distributions.squash_bijector import SquashBijector
from malib.networks.mlp import MLP
from .base_policy import LatentSpacePolicy, StochasticPolicy

# SCALE_DIAG_MIN_MAX = (tf.constant([-20.]), tf.constant([2.]))
# LOG_STD_MAX = tf.constant([2.])
# LOG_STD_MIN = tf.constant([-20.])

SCALE_DIAG_MIN_MAX = (-20., 2.)
LOG_STD_MAX = 2.
LOG_STD_MIN = -20.


class GaussianPolicy(LatentSpacePolicy):
    def __init__(self, input_shapes, output_shape, squash=True, preprocessor=None, name='gaussian_policy', repara=True,
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
        self._squash = squash
        self._name = name
        self._preprocessor = preprocessor
        self._repara = repara

        super(GaussianPolicy, self).__init__(*args, **kwargs)

        self.condition_inputs = [tf.keras.layers.Input(shape=input_shape) for input_shape in input_shapes]

        conditions = tf.keras.layers.Lambda(lambda x: tf.concat(x, axis=-1))(self.condition_inputs)

        if preprocessor is not None:
            conditions = preprocessor(conditions)

        self._shift_and_log_scale_diag_net_model = self._shift_and_log_scale_diag_net(
            input_shapes=(conditions.shape[1:],), output_size=output_shape[0] * 2, )

        # shift_and_log_scale_diag = self._shift_and_log_scale_diag_net(input_shapes=(conditions.shape[1:],),
        # 															  output_size=output_shape[0] * 2, )(conditions)
        shift_and_log_scale_diag = self._shift_and_log_scale_diag_net_model(conditions)

        shift, log_scale_diag = tf.keras.layers.Lambda(
            lambda shift_and_log_scale_diag: tf.split(shift_and_log_scale_diag, num_or_size_splits=2, axis=-1))(
            shift_and_log_scale_diag)

        log_scale_diag = tf.keras.layers.Lambda(
            lambda log_scale_diag: tf.clip_by_value(log_scale_diag, *SCALE_DIAG_MIN_MAX))(log_scale_diag)

        self.shift_scale_model = tf.keras.Model(self.condition_inputs, (shift, log_scale_diag))

        batch_size = tf.keras.layers.Lambda(lambda x: tf.shape(x)[0])(conditions)

        base_distribution = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(output_shape),
                                                                     scale_diag=tf.ones(output_shape))

        latents = tf.keras.layers.Lambda(lambda batch_size: base_distribution.sample(batch_size))(batch_size)

        self.latents_model = tf.keras.Model(self.condition_inputs, latents)
        self.latents_input = tf.keras.layers.Input(shape=output_shape)

        def raw_actions_fn(inputs):
            shift, log_scale_diag, latents = inputs
            if self._repara:
                bijector = tfp.bijectors.Affine(shift=shift, scale_diag=tf.math.exp(log_scale_diag))
                actions = bijector.forward(latents)
            else:
                actions = tfp.distributions.MultivariateNormalDiag(loc=shift,
                                                                   scale_diag=tf.math.exp(log_scale_diag)).sample()
            return actions

        raw_actions = tf.keras.layers.Lambda(raw_actions_fn)((shift, log_scale_diag, latents))

        raw_actions_for_fixed_latents = tf.keras.layers.Lambda(raw_actions_fn)(
            (shift, log_scale_diag, self.latents_input))

        squash_bijector = (SquashBijector() if self._squash else tfp.bijectors.Identity())

        actions = tf.keras.layers.Lambda(lambda raw_actions: squash_bijector.forward(raw_actions))(raw_actions)
        self.actions_model = tf.keras.Model(self.condition_inputs, actions, name=self._name)
        self.raw_and_actions_model = tf.keras.Model(self.condition_inputs, [raw_actions, actions])

        actions_for_fixed_latents = tf.keras.layers.Lambda(lambda raw_actions: squash_bijector.forward(raw_actions))(
            raw_actions_for_fixed_latents)
        self.actions_model_for_fixed_latents = tf.keras.Model((*self.condition_inputs, self.latents_input),
                                                              actions_for_fixed_latents)

        deterministic_actions = tf.keras.layers.Lambda(lambda shift: squash_bijector.forward(shift))(shift)

        self.deterministic_actions_model = tf.keras.Model(self.condition_inputs, deterministic_actions)
        self.deterministic_raw_and_actions_model = tf.keras.Model(self.condition_inputs, [raw_actions, actions])

        def log_pis_fn(inputs):
            shift, log_scale_diag, actions = inputs
            if self._repara:
                base_distribution = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(output_shape),
                                                                             scale_diag=tf.ones(output_shape))
                bijector = tfp.bijectors.Chain(
                    (squash_bijector, tfp.bijectors.Affine(shift=shift, scale_diag=tf.math.exp(log_scale_diag)),))
                distribution = (
                    tfp.distributions.TransformedDistribution(distribution=base_distribution, bijector=bijector))
                log_pis = distribution.log_prob(actions)[:, None]
            else:
                base_distribution = tfp.distributions.MultivariateNormalDiag(loc=shift,
                                                                             scale_diag=tf.math.exp(log_scale_diag))
                distribution = (
                    tfp.distributions.TransformedDistribution(distribution=base_distribution, bijector=squash_bijector))
                log_pis = distribution.log_prob(actions)[:, None]
            return log_pis

        self.raw_actions_input = tf.keras.layers.Input(shape=output_shape)
        self.actions_input = tf.keras.layers.Input(shape=output_shape)

        log_pis = tf.keras.layers.Lambda(log_pis_fn)([shift, log_scale_diag, actions])

        log_pis_for_action_input = tf.keras.layers.Lambda(log_pis_fn)([shift, log_scale_diag, self.actions_input])

        log_pis_for_raw_action_input = tf.keras.layers.Lambda(log_pis_fn)(
            [shift, log_scale_diag, tf.tanh(self.raw_actions_input)])

        self.log_pis_model = tf.keras.Model((*self.condition_inputs, self.actions_input), log_pis_for_action_input)
        self.raw_log_pis_model = tf.keras.Model((*self.condition_inputs, self.raw_actions_input),
                                                log_pis_for_raw_action_input)
        self.diagnostics_model = tf.keras.Model(self.condition_inputs,
                                                (shift, log_scale_diag, log_pis, raw_actions, actions))

    def _shift_and_log_scale_diag_net(self, input_shapes, output_size):
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
        return list(set(super(GaussianPolicy, self).non_trainable_weights))

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

    def log_pis_for_raw(self, conditions, raw_actions):
        assert not self._deterministic
        return self.raw_log_pis_model([*conditions, raw_actions])

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
        return tf.reduce_mean(self._shift_and_log_scale_diag_net_model.losses)

    def get_diagnostics(self, conditions):
        """Return diagnostic information of the policy.
        Returns the mean, min, max, and standard deviation of means and
        covariances.
        """
        (shifts_np, log_scale_diags_np, log_pis_np, raw_actions_np, actions_np) = self.diagnostics_model.predict(
            conditions)

        return OrderedDict({'{}/shifts-mean'.format(self._name): np.mean(shifts_np),
                            '{}/shifts-std'.format(self._name): np.std(shifts_np),

                            '{}/log_scale_diags-mean'.format(self._name): np.mean(log_scale_diags_np),
                            '{}/log_scale_diags-std'.format(self._name): np.std(log_scale_diags_np),

                            '{}/-log-pis-mean'.format(self._name): np.mean(-log_pis_np),
                            '{}/-log-pis-std'.format(self._name): np.std(-log_pis_np),

                            '{}/raw-actions-mean'.format(self._name): np.mean(raw_actions_np),
                            '{}/raw-actions-std'.format(self._name): np.std(raw_actions_np),

                            '{}/actions-mean'.format(self._name): np.mean(actions_np),
                            '{}/actions-std'.format(self._name): np.std(actions_np), })


class GaussianMLPPolicy(GaussianPolicy):
    def __init__(self, hidden_layer_sizes, activation=tf.keras.layers.ReLU(), output_activation='linear',
                 pi_std=None, *args, **kwargs):
        self._hidden_layer_sizes = hidden_layer_sizes
        self._activation = activation
        self._output_activation = output_activation

        self._pi_std = pi_std

        self._Serializable__initialize(locals())
        super(GaussianMLPPolicy, self).__init__(*args, **kwargs)

    def _shift_and_log_scale_diag_net(self, input_shapes, output_size):
        shift_and_log_scale_diag_net = MLP(input_shapes=input_shapes, hidden_layer_sizes=self._hidden_layer_sizes,
                                           output_size=output_size, activation=self._activation,
                                           output_activation=self._output_activation,
                                           name='{}/GaussianMLPPolicy'.format(self._name),
                                           kernel_regularizer=tf.keras.regularizers.l2(0.001),
                                           bias_regularizer=tf.keras.regularizers.l2(0.001), )
        return shift_and_log_scale_diag_net
