# Created by yingwen at 2019-03-12

class ExplorationBase:
    def get_action(self, t, observation, policy, **kwargs):
        raise NotImplementedError

    def get_actions(self, t, observations, policy, **kwargs):
        raise NotImplementedError

    def reset(self):
        pass