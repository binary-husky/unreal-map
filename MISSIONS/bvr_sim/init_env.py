import os,sys
import numpy as np
from .env.xsim_manager import XSimManager
from .env.communication_service import CommunicationService
from .env.env_runner import EnvRunner
class ChainVar(object):
    def __init__(self, chain_func, chained_with):
        self.chain_func = chain_func
        self.chained_with = chained_with

class ScenarioConfig(object): # ADD_TO_CONF_SYSTEM 加入参数搜索路径 do not remove this comment !!!

    ################ needed by the core of hmp ################
    N_TEAM = 1  
    N_AGENT_EACH_TEAM = [5,]
    AGENT_ID_EACH_TEAM = [range(0,5),]
    TEAM_NAMES = ['ALGORITHM.None->None',] 

    ## If the length of action array != the number of agents, set ActAsUnity to True
    ## If the length of action array == the number of agents, set ActAsUnity to False
    ActAsUnity = False  

    ## If the length of reward array == the number of agents, set RewardAsUnity to False
    ## If the length of reward array == 1, set RewardAsUnity to True
    RewardAsUnity = True


    ################ needed by env itself ################
    render = False
    max_steps_episode = 1000

    ################ needed by some ALGORITHM ################
    state_provided = False
    avail_act_provided = False


    # n_actions = 6 + map_param_registry[map_]['n_enemies']
    # n_actions_cv = ChainVar(
    #     lambda map_:6 + map_param_registry[map_]['n_enemies'], 
    #     chained_with=['map_']
    # )
    obs_vec_length = 6
    return_mat = False
    block_invalid_action = True # sc2 中，需要始终屏蔽掉不可用的动作

class BASE_ENV(object):
    def __init__(self, rank):
        # xsim引擎控制器
        self.xsim_manager = XSimManager(time_ratio=100)
        # 与xsim引擎交互通信服务
        self.communication_service = CommunicationService(self.xsim_manager.address)
        import time
        time.sleep(100)

    def step(self, act):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def __del__(self):
        self.xsim_manager.__del__()

class BVR_ENV(BASE_ENV):
    def __init__(self, rank):
        super(BVR_ENV, self).__init__(rank)




def make_bvr_env(env_id, rank):
    return BVR_ENV(rank)
