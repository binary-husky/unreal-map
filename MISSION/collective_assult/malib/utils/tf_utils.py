# Created by yingwen at 2019-03-12
import tensorflow as tf

EPS = 1e-6

def soft_variables_update(source_variables, target_variables, tau=1.0,
                          sort_variables_by_name=False, name=None):
    """Performs a soft/hard update of variables from the source to the target.
    For each variable v_t in target variables and its corresponding variable v_s
    in source variables, a soft update is:
    v_t = (1 - tau) * v_t + tau * v_s
    When tau is 1.0 (the default), then it does a hard update:
    v_t = v_s
    Args:
      source_variables: list of source variables.
      target_variables: list of target variables.
      tau: A float scalar in [0, 1]. When tau is 1.0 (the default), we do a hard
        update.
      sort_variables_by_name: A bool, when True would sort the variables by name
        before doing the update.
      name: A string, name.
    Returns:
      An operation that updates target variables from source variables.
    Raises:
      ValueError: if tau is not in [0, 1].
    """
    if tau < 0 or tau > 1:
        raise ValueError('Input `tau` should be in [0, 1].')
    updates = []

    op_name = 'soft_variables_update'
    if name is not None:
        op_name = '{}_{}'.format(name, op_name)
    if tau == 0.0 or not source_variables or not target_variables:
        return tf.no_op(name=op_name)
    if sort_variables_by_name:
        source_variables = sorted(source_variables, key=lambda x: x.name)
        target_variables = sorted(target_variables, key=lambda x: x.name)
    for (v_s, v_t) in zip(source_variables, target_variables):
        v_t.shape.assert_is_compatible_with(v_s.shape)
        if tau == 1.0:
            update = v_t.assign(v_s)
        else:
            update = v_t.assign((1 - tau) * v_t + tau * v_s)
        updates.append(update)
    return tf.group(*updates, name=op_name)


def clip_gradient_norms(gradients_to_variables, max_norm):
    """Clips the gradients by the given value.
    Args:
      gradients_to_variables: A list of gradient to variable pairs (tuples).
      max_norm: the maximum norm value.
    Returns:
      A list of clipped gradient to variable pairs.
    """
    clipped_grads_and_vars = []
    for grad, var in gradients_to_variables:
        if grad is not None:
            if isinstance(grad, tf.IndexedSlices):
                tmp = tf.clip_by_norm(grad.values, max_norm)
                grad = tf.IndexedSlices(tmp, grad.indices, grad.dense_shape)
            else:
                grad = tf.clip_by_norm(grad, max_norm)
        clipped_grads_and_vars.append((grad, var))
    return clipped_grads_and_vars


def apply_gradients(gradients, variables, optimizer, gradient_clipping_max_norm=None):
    # Tuple is used for py3, where zip is a generator producing values once.
    grads_and_vars = tuple(zip(gradients, variables))
    if gradient_clipping_max_norm is not None:
        grads_and_vars = clip_gradient_norms(grads_and_vars, gradient_clipping_max_norm)
    optimizer.apply_gradients(grads_and_vars)


def squash_correction(actions, squashed=True):
    if squashed:
        actions = tf.atanh(actions)
    return tf.reduce_sum(tf.math.log(1 - tf.tanh(actions) ** 2 + EPS), axis=1)


def assert_shape(tensor, expected_shape):
    tensor_shape = tensor.shape
    assert len(tensor_shape) == len(expected_shape)
    assert all([a == b for a, b in zip(tensor_shape[1:], expected_shape[1:])])


def remove_inf_nan(tensor):
    tensor = tf.where(tf.math.is_nan(tensor), tf.zeros_like(tensor), tensor)
    tensor = tf.where(tf.math.is_inf(tensor), tf.zeros_like(tensor), tensor)
    return tensor