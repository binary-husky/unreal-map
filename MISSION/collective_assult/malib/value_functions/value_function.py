# Created by yingwen at 2019-03-10

from collections import OrderedDict
from malib.networks.mlp import MLP
from malib.value_functions.base_value_function import BaseValueFunction
import tensorflow as tf
import numpy as np


class ValueFunction(BaseValueFunction):
    def __init__(self,
                 input_shapes,
                 output_shape,
                 preprocessor=None,
                 name=None,
                 *args,
                 **kwargs):
        self._Serializable__initialize(locals())

        self._input_shapes = input_shapes
        self._output_shape = output_shape
        self._name = name
        self._preprocessor = preprocessor

        super(ValueFunction, self).__init__(*args, **kwargs)

        self.condition_inputs = [
            tf.keras.layers.Input(shape=input_shape)
            for input_shape in input_shapes
        ]

        conditions = tf.keras.layers.Lambda(
            lambda x: tf.concat(x, axis=-1)
        )(self.condition_inputs)

        if preprocessor is not None:
            conditions = preprocessor(conditions)

        #
        # concated_shape = tuple(np.sum(self._input_shapes, axis=-1))
        # print('concated_shape', concated_shape)
        values = self._value_net(
            input_shapes=(conditions.shape[1:], ),
            output_size=output_shape[0],
        )(conditions)

        self.values_model = tf.keras.Model(self.condition_inputs, values)
        self.diagnostics_model = tf.keras.Model(
            self.condition_inputs, values)

    def _value_net(self, input_shapes, output_size):
        raise NotImplementedError

    def get_values(self, conditions):
        return self.values_model(conditions)

    def get_values_np(self, conditions):
        return self.values_model.predict(conditions)

    def get_weights(self):
        return self.values_model.get_weights()

    def set_weights(self, *args, **kwargs):
        return self.values_model.set_weights(*args, **kwargs)

    @property
    def trainable_variables(self):
        return self.values_model.trainable_variables

    @property
    def non_trainable_weights(self):
        """Due to our nested model structure, we need to filter duplicates."""
        return list(set(super(ValueFunction, self).non_trainable_weights))

    def get_diagnostics(self, conditions):
        """Return diagnostic information of the policy.
        Returns the mean, min, max, and standard deviation of means and
        covariances.
        """
        values = self.diagnostics_model.predict(conditions)

        return OrderedDict({
            'values-mean': np.mean(values),
            'values-std': np.std(values),
            'conditions': conditions,
        })


class MLPValueFunction(ValueFunction):
    def __init__(self,
                 hidden_layer_sizes,
                 activation='relu',
                 output_activation='linear',
                 *args, **kwargs):
        self._hidden_layer_sizes = hidden_layer_sizes
        self._activation = activation
        self._output_activation = output_activation

        self._Serializable__initialize(locals())
        super(MLPValueFunction, self).__init__(*args, **kwargs)

    def _value_net(self, input_shapes, output_size):
        values = MLP(
            input_shapes=input_shapes,
            hidden_layer_sizes=self._hidden_layer_sizes,
            output_size=output_size,
            activation=self._activation,
            output_activation=self._output_activation)
        return values
