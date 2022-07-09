# Created by yingwen at 2019-03-11

from collections import OrderedDict
from malib.core import Serializable


class BaseValueFunction(Serializable):

    def get_values(self, conditions):
        raise NotImplementedError

    def get_values_np(self, conditions):
        raise NotImplementedError

    def get_weights(self):
        raise NotImplementedError

    def set_weights(self, *args, **kwargs):
        raise NotImplementedError

    def get_diagnostics(self, conditions):
        """Return diagnostic information of the policy.
        Arguments:
            conditions: Observations to run the diagnostics for.
        Returns:
            diagnostics: OrderedDict of diagnostic information.
        """
        diagnostics = OrderedDict({})
        return diagnostics

    def terminate(self):
        """
        Clean up operation
        """
        pass

    @property
    def recurrent(self):
        """
        Indicates whether the policy is recurrent.
        :return:
        """
        return False

    def __getstate__(self):
        state = Serializable.__getstate__(self)
        state['pickled_weights'] = self.get_weights()
        return state

    def __setstate__(self, state):
        Serializable.__setstate__(self, state)
        self.set_weights(state['pickled_weights'])