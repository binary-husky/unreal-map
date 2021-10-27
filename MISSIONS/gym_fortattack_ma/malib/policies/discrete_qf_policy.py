# Created by yingwen at 2019-03-11

from collections import OrderedDict

import tensorflow as tf
import numpy as np
from .base_policy import Policy


class DiscreteQfDerivedPolicy(Policy):
    """
    Discrete Q-Function Derived policy.
    Args:
        input_shapes: Input shapes.
        qf: The q-function used.
    """

    def __init__(self, input_shapes, qf, preprocessor=None):
        super(DiscreteQfDerivedPolicy, self).__init__()
        self._Serializable__initialize(locals())
        self._qf = qf

        self.condition_inputs = [
            tf.keras.layers.Input(shape=input_shape)
            for input_shape in input_shapes
        ]

        conditions = tf.keras.layers.Lambda(
            lambda x: tf.concat(x, axis=-1)
        )(self.condition_inputs)

        if preprocessor is not None:
            conditions = preprocessor(conditions)

        q_vals = self._qf(conditions)
        actions = tf.math.argmax(q_vals)
        self.actions_model = tf.keras.Model(self.condition_inputs, actions)

        self.diagnostics_model = tf.keras.Model(
            self.condition_inputs, (q_vals, actions))

    def get_weights(self):
        return self.actions_model.get_weights()

    def set_weights(self, *args, **kwargs):
        return self.actions_model.set_weights(*args, **kwargs)

    @property
    def trainable_variables(self):
        return []

    def reset(self):
        pass

    def get_actions(self, conditions):
        return self.actions_model(conditions)

    def get_action(self, condition):
        return self.actions_model(np.array([condition]))[0]

    def get_action_np(self, condition):
        return self.actions_model.predict(np.array([condition]))[0]

    def get_actions_np(self, conditions):
        return self.actions_model.predict(conditions)


    def get_diagnostics(self, conditions):
        """Return diagnostic information of the policy.
                Returns the mean, min, max, and standard deviation of means and
                covariances.
                """
        (q_vals,
         actions_np) = self.diagnostics_model.predict(conditions)

        return OrderedDict({
            '{}/raw-actions-mean'.format(self._name): np.mean(q_vals),
            '{}/raw-actions-std'.format(self._name): np.std(q_vals),

            '{}/actions'.format(self._name): np.mean(actions_np),
        })
