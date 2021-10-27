# Created by yingwen at 2019-03-10

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
# return tf.math.log(tf.constant(1. + 1e-6, dtype=tf.float32) - tf.square(tf.math.tanh(x)))

class SquashBijector(tfp.bijectors.Bijector):
    def __init__(self, validate_args=False, name="tanh"):
        super(SquashBijector, self).__init__(
            forward_min_event_ndims=0,
            validate_args=validate_args,
            name=name)

    def _forward(self, x):
        return tf.nn.tanh(x)

    def _inverse(self, y):
        return tf.atanh(y)

    def _forward_log_det_jacobian(self, x):
        # return tf.math.log(1. + 1e-6 - tf.square(tf.math.tanh(x)))
        return 2. * (np.log(2.) - x - tf.nn.softplus(-2. * x))