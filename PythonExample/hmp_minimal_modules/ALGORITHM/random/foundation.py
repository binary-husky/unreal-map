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
