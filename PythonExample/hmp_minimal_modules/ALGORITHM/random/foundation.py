import numpy as np
from UTIL.colorful import *
from UTIL.tensor_ops import my_view, __hash__
from config import GlobalConfig

class AlgorithmConfig:
    preserve = ''


# 改变自身颜色的动作 ChangeColor(color_index)
# 运动动作 MoveToDirection(x, y, z)
# 提高一段时间的加速度的动作 AccHighLevel4
# 发射武器动作 FireToWaterDrop(water_drop_uid)


class RandomController(object):
    def __init__(self, n_agent, n_thread, space, mcv=None, team=None):
        self.n_agent = n_agent
        self.n_thread = n_thread
        self.space = space
        self.mcv = mcv
        self.n_action = GlobalConfig.ScenarioConfig.n_actions

    def interact_with_env(self, StateRecall):
        obs = StateRecall['Latest-Obs']
        P = StateRecall['ENV-PAUSE']
        active_thread_obs = obs[~P]
        actions = np.random.randint(low=0,high=self.n_action, size=(self.n_thread, self.n_agent, 1))
        StateRecall['_hook_'] = None
        return actions, StateRecall 


class RandomControllerWithActionSetV2(object):

    def __init__(self, n_agent, n_thread, space, mcv=None, team=None):
        from .actionset import ActionConvertV2
        self.n_agent = n_agent
        self.n_thread = n_thread
        self.space = space
        self.mcv = mcv
        self.actions_set = ActionConvertV2(
            SELF_TEAM_ASSUME=team, 
            OPP_TEAM_ASSUME=(1-team), 
            OPP_NUM_ASSUME=GlobalConfig.ScenarioConfig.N_AGENT_EACH_TEAM[1-team]
        )
        self.n_action = self.actions_set.n_act

    def interact_with_env(self, StateRecall):
        obs = StateRecall['Latest-Obs']
        P = StateRecall['ENV-PAUSE']

        active_thread_obs = obs[~P]
        actions = np.random.randint(low=0,high=self.n_action, size=(self.n_thread, self.n_agent, 1))

        act_converted = np.array(
                            list(map(lambda x: self.actions_set.convert_act_arr(None, x), 
                            actions.flatten()))).reshape(self.n_thread, self.n_agent, self.actions_set.ActDigitLen)

        StateRecall['_hook_'] = None
        return act_converted, StateRecall 


class RandomControllerWithActionSetV4(object):

    def __init__(self, n_agent, n_thread, space, mcv=None, team=None):
        from .actionset import ActionConvertMovingV4
        self.n_agent = n_agent
        self.n_thread = n_thread
        self.space = space
        self.mcv = mcv
        self.actions_set = ActionConvertMovingV4(
            SELF_TEAM_ASSUME=team, 
            OPP_TEAM_ASSUME=(1-team), 
            OPP_NUM_ASSUME=GlobalConfig.ScenarioConfig.N_AGENT_EACH_TEAM[1-team]
        )
        self.n_action = self.actions_set.n_act

    def interact_with_env(self, StateRecall):
        obs = StateRecall['Latest-Obs']
        P = StateRecall['ENV-PAUSE']

        active_thread_obs = obs[~P]
        actions = np.random.randint(low=0,high=self.n_action, size=(self.n_thread, self.n_agent, 1))

        act_converted = np.array(
                            list(map(lambda x: self.actions_set.convert_act_arr(None, x), 
                            actions.flatten()))).reshape(self.n_thread, self.n_agent, self.actions_set.ActDigitLen)

        StateRecall['_hook_'] = None
        return act_converted, StateRecall 



class RandomControllerWithActionSetV1(object):

    def __init__(self, n_agent, n_thread, space, mcv=None, team=None):
        from .actionset import ActionConvertV1Carrier
        self.n_agent = n_agent
        self.n_thread = n_thread
        self.space = space
        self.mcv = mcv
        self.actions_set = ActionConvertV1Carrier(
            SELF_TEAM_ASSUME=team, 
            OPP_TEAM_ASSUME=(1-team), 
            OPP_NUM_ASSUME=GlobalConfig.ScenarioConfig.N_AGENT_EACH_TEAM[1-team]
        )
        self.n_action = self.actions_set.n_act

    def interact_with_env(self, StateRecall):
        obs = StateRecall['Latest-Obs']
        P = StateRecall['ENV-PAUSE']

        active_thread_obs = obs[~P]
        actions = np.random.randint(low=0,high=self.n_action, size=(self.n_thread, self.n_agent, 1))

        act_converted = np.array(
                            list(map(lambda x: self.actions_set.convert_act_arr(None, x), 
                            actions.flatten()))).reshape(self.n_thread, self.n_agent, self.actions_set.ActDigitLen)

        StateRecall['_hook_'] = None
        return act_converted, StateRecall 
    

class RandomControllerWithMomentumAgent(object):

    def __init__(self, n_agent, n_thread, space, mcv=None, team=None):
        from .actionset import ActionConvertV1Momentum
        self.n_agent = n_agent
        self.n_thread = n_thread
        self.space = space
        self.mcv = mcv
        self.actions_set = ActionConvertV1Momentum(
            SELF_TEAM_ASSUME=team, 
            OPP_TEAM_ASSUME=(1-team), 
            OPP_NUM_ASSUME=GlobalConfig.ScenarioConfig.N_AGENT_EACH_TEAM[1-team]
        )
        self.n_action = self.actions_set.n_act

    def interact_with_env(self, StateRecall):
        obs = StateRecall['Latest-Obs']
        P = StateRecall['ENV-PAUSE']

        active_thread_obs = obs[~P]
        actions = np.random.randint(low=0,high=self.n_action, size=(self.n_thread, self.n_agent, 1))

        act_converted = np.array(
                            list(map(lambda x: self.actions_set.convert_act_arr(None, x), 
                            actions.flatten()))).reshape(self.n_thread, self.n_agent, self.actions_set.ActDigitLen)

        StateRecall['_hook_'] = None
        return act_converted, StateRecall 