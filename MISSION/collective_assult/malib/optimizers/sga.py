import tensorflow as tf

def fwd_gradients(ys, xs, grad_xs=None, stop_gradients=None,
                  colocate_gradients_with_ops=True):

    us = [tf.zeros_like(y) + float("nan") for y in ys]
    dydxs = tf.gradients(ys=ys, xs=xs, grad_ys=us, stop_gradients=stop_gradients)
    # Deal with strange types that tf.gradients returns but can't
    dydxs = [
        tf.convert_to_tensor(value=dydx) if isinstance(dydx, tf.IndexedSlices) else dydx
        for dydx in dydxs
    ]
    dydxs = [
        tf.zeros_like(x) if dydx is None else dydx for x, dydx in zip(xs, dydxs)
    ]
    dydxs = tf.gradients(ys=dydxs, xs=us, grad_ys=grad_xs)
    return dydxs

def list_divide_scalar(xs, y):
    return [x / y for x in xs]


def list_subtract(xs, ys):
    return [x - y for (x, y) in zip(xs, ys)]


def jacobian_vec(ys, xs, vs):
    return fwd_gradients(
        ys, xs, grad_xs=vs, stop_gradients=xs)


def jacobian_transpose_vec(ys, xs, vs):
    dydxs = tf.gradients(ys=ys, xs=xs, grad_ys=vs, stop_gradients=xs)
    dydxs = [
        tf.zeros_like(x) if dydx is None else dydx for x, dydx in zip(xs, dydxs)
    ]
    return dydxs


def _dot(x, y):
    dot_list = []
    for xx, yy in zip(x, y):
        dot_list.append(tf.reduce_sum(input_tensor=xx * yy))
    return tf.add_n(dot_list)

class SymplecticOptimizer(tf.train.Optimizer):
    """Optimizer that corrects for rotational components in gradients."""

    def __init__(self,
                 learning_rate,
                 reg_params=1.,
                 use_signs=True,
                 use_locking=False,
                 name='symplectic_optimizer'):
        super(SymplecticOptimizer, self).__init__(
            use_locking=use_locking, name=name)
        self._gd = tf.train.RMSPropOptimizer(learning_rate)
        self._reg_params = reg_params
        self._use_signs = use_signs

    def compute_gradients(self,
                          loss,
                          var_list=None,
                          gate_gradients=tf.train.Optimizer.GATE_OP,
                          aggregation_method=None,
                          colocate_gradients_with_ops=False,
                          grad_loss=None):
        return self._gd.compute_gradients(loss, var_list, gate_gradients,
                                          aggregation_method,
                                          colocate_gradients_with_ops, grad_loss)

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        grads, vars_ = zip(*grads_and_vars)
        n = len(vars_)
        h_v = jacobian_vec(grads, vars_, grads)
        ht_v = jacobian_transpose_vec(grads, vars_, grads)
        at_v = list_divide_scalar(list_subtract(ht_v, h_v), 2.)
        if self._use_signs:
            grad_dot_h = _dot(grads, ht_v)
            at_v_dot_h = _dot(at_v, ht_v)
            mult = grad_dot_h * at_v_dot_h
            lambda_ = tf.sign(mult / n + 0.1) * self._reg_params
        else:
            lambda_ = self._reg_params
        apply_vec = [(g + lambda_ * ag, x)
                     for (g, ag, x) in zip(grads, at_v, vars_)
                     if at_v is not None]
        return self._gd.apply_gradients(apply_vec, global_step, name)