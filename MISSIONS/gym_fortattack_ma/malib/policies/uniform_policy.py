# Created by yingwen at 2019-03-10

from collections import OrderedDict

import tensorflow as tf
import numpy as np
from malib.policies.base_policy import Policy


class UniformPolicy(Policy):
    def __init__(self, input_shapes, output_shape, action_range=(-1.0, 1.0)):
        super(UniformPolicy, self).__init__()
        self._Serializable__initialize(locals())

        self.inputs = [
            tf.keras.layers.Input(shape=input_shape)
            for input_shape in input_shapes
        ]
        self._action_range = action_range

        x = tf.keras.layers.Lambda(
            lambda x: tf.concat(x, axis=-1)
        )(self.inputs)

        actions = tf.keras.layers.Lambda(
            lambda x: tf.random.uniform(
                (tf.shape(x)[0], output_shape[0]),
                *action_range)
        )(x)

        self.actions_model = tf.keras.Model(self.inputs, actions)

        self.actions_input = tf.keras.Input(shape=output_shape)

        log_pis = tf.keras.layers.Lambda(
            lambda x: tf.tile(tf.math.log([
                (action_range[1] - action_range[0]) / 2.0
            ])[None], (tf.shape(x)[0], 1))
        )(self.actions_input)

        self.log_pis_model = tf.keras.Model(
            (*self.inputs, self.actions_input), log_pis)

    def get_weights(self):
        return []

    def set_weights(self, *args, **kwargs):
        return

    @property
    def trainable_variables(self):
        return []

    def reset(self):
        pass

    def get_actions(self, conditions):
        return self.actions_model(conditions)

    def get_action(self, condition):
        return self.actions_model(np.array([condition]))[0]

    def log_pis(self, conditions, actions):
        return self.log_pis_model([*conditions, actions])

    def get_actions_np(self, conditions):
        return self.actions_model.predict(conditions)

    def log_pis_np(self, conditions, actions):
        return self.log_pis_model.predict([*conditions, actions])

    def get_diagnostics(self, conditions):
        return OrderedDict({})