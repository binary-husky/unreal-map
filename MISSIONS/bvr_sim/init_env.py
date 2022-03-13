import os, sys, math
import numpy as np
from .env.xsim_manager import XSimManager
from .env.communication_service import CommunicationService
from .env.env_runner import EnvRunner
from .agent.env_cmd import CmdEnv
from .render import RenderBridge
from .converter import Converter
from .endgame import EndGame

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

    ## If the length of action array == the number of teams, set ActAsUnity to True
    ## If the length of action array == the number of agents, set ActAsUnity to False
    ActAsUnity = False  

    ## If the length of reward array == the number of agents, set RewardAsUnity to False
    ## If the length of reward array == 1, set RewardAsUnity to True
    RewardAsUnity = True

    ## If the length of obs array == the number of agents, set ObsAsUnity to False
    ## If the length of obs array == the number of teams, set ObsAsUnity to True
    ObsAsUnity = True

    ################ needed by env itself ################
    render = False
    max_steps_episode = 1000

    SpeedLevels = 5 # the speed is divided as 5 discrete speed
    # meanning [MinSpeed, , MaxSpeed]  
    # MinSpeed + (MaxSpeed - MinSpeed)*( i/(SpeedLevels-1) ), i = [0,1,2,3,4]
    HeightLevels = 5 # the speed is divided as 5 discrete speed
    # meanning [MinHeight , MaxHeight]  
    # MinHeight + (MaxHeight - MinHeight)*( i/(HeightLevels-1) ), i = [0,1,2,3,4]

    internal_step = 1

    ################ needed by some ALGORITHM ################
    state_provided = False
    avail_act_provided = False
    n_actions = 4


    obs_vec_length = 6
    return_mat = False
    block_invalid_action = True # sc2 中，需要始终屏蔽掉不可用的动作


class BaseEnv(object):
    def __init__(self, rank) -> None:
        self.observation_space = None
        self.action_space = None

    def step(self, act):
        # obs: a Tensor with shape (n_agent, ...)
        # reward: a Tensor with shape (n_agent, 1) or (n_team, 1) or (1,)
        # done: a Bool
        # info: a dict
        raise NotImplementedError
        return (ob, reward, done, info)

    def reset(self):
        # obs: a Tensor with shape (n_agent, ...)
        # done: a Bool
        raise NotImplementedError
        return ob, info


def make_bvr_env(env_id, rank):
    return BVR_PVE(rank)
    # return BVR_EVE(rank)
# class BASE_ENV_PVP(BVR):
#     def __init__(self) -> None:
#         super().__init__()
#     pass    # raise NotImplementedError
    

class BVR(BaseEnv, RenderBridge, EndGame):
    def __init__(self, rank) -> None:
        super(BVR, self).__init__(rank)

        from .agent.dummy_observer.base import Baseclass
        # an observer keeps track of agent info such as distence
        self.OBSERVER_CLASS = Baseclass
        # xsim引擎控制器
        self.xsim_manager = XSimManager(time_ratio=100)
        # 与xsim引擎交互通信服务
        self.communication_service = CommunicationService(self.xsim_manager.address)

        # score of episode
        self.red_score = 0
        self.blue_score = 0

        # self.n_agents
        self.n_agents = 5
        self.n_opp = 5

        self.ScenarioConfig = ScenarioConfig

        # the id of env among parallel envs, only rank=0 thread is allowed to render
        self.rank = rank

        
    def bvr_step(self, cmd_list):
        self.raw_obs = self.communication_service.step(cmd_list)
        # an observer keeps track of agent info such as distence
        self.observer.observe(self.raw_obs["sim_time"], self.raw_obs['red'])
        done, red_win, blue_win = self.get_done(self.raw_obs)
        done = (done>0); 
        win = {'red':(red_win>0), 'blue':(blue_win>0), 'done':done}
        return self.raw_obs, done, win

    def bvr_reset(self):
        self._end_()
        self.communication_service.reset()
        self.raw_obs = self.communication_service.step([])
        # observer's first info feedin
        self.observer = self.OBSERVER_CLASS('red', {"side": 'red'})
        self.observer.observe(self.raw_obs["sim_time"], self.raw_obs['red'])
        return self.raw_obs

    def _end_(self):
        return self.communication_service.end()

    def __del__(self):
        print('[init_env.py] bvr class del')
        self.xsim_manager.__del__()

class BVR_PVE(BVR, Converter):
    def __init__(self, rank):
        super(BVR_PVE, self).__init__(rank)
        # opponent AI
        from .agent.yiteam_final_commit_v7.Yi_team import Yi_team
        self.OPP_CONTROLLER_CLASS = Yi_team

        self.player_color = "red"
        self.opp_color = "blue"
        self.internal_step = self.ScenarioConfig.internal_step



    def step(self, raw_action):
        # convert raw_action Tensor from RL to actions recognized by the game
        player_action = self.parse_raw_action(raw_action)
        
        for _ in range(self.internal_step):
            # move the opponents
            opp_action = self.opp_controller.step(self.raw_obs["sim_time"], self.raw_obs[self.opp_color])

            # commmit actions to the environment
            cmd_list = player_action + opp_action
            self.raw_obs, done, win = self.bvr_step(cmd_list)
            
            if done:  break
            player_action = []

        # use observer's info to render
        if self.ScenarioConfig.render and self.rank==0: self.advanced_render(self.raw_obs["sim_time"], win, self.observer)

        # system get done
        assert self.raw_obs is not None, ("obs is None")
        converted_obs = self.convert_obs(self.raw_obs)
        reward = 0

        info = {'win':win[self.player_color]}
        return converted_obs, reward, done, info

    def reset(self):
        # randomly switch sides
        randx = np.random.rand() 
        if randx < 0.5: self.player_color, self.opp_color = ("red", "blue")
        else: self.player_color, self.opp_color = ("blue", "red")

        # initialize controllers
        # controller 1
        self.opp_controller = self.OPP_CONTROLLER_CLASS(self.opp_color, {"side": self.opp_color})


        # reset signal to base
        self.raw_obs = self.bvr_reset()

        # read obs after reset
        converted_obs = self.convert_obs(self.raw_obs)
        
        info = self.raw_obs[self.player_color]
        return converted_obs, info




class BVR_EVE(BVR, Converter):
    def __init__(self, rank):
        super().__init__(rank)
        # super(BVR_PVE, self).__init__(rank)
        # opponent AI
        from .agent.yiteam_final_commit_v7.Yi_team import Yi_team
        self.OPP_CONTROLLER_CLASS = Yi_team
        self.player_color = "red"
        self.opp_color = "blue"

    def step(self, raw_action):
        # ignore raw_action 
        
        # move the opponents
        fakeplayer_action = self.fakeplayer_controller.step(self.raw_obs["sim_time"], self.raw_obs[self.player_color])
        opp_action = self.opp_controller.step(self.raw_obs["sim_time"], self.raw_obs[self.opp_color])
        # commmit actions to the environment
        self.raw_obs, done, win = self.bvr_step(fakeplayer_action + opp_action)
        # use observer's info to render
        if self.ScenarioConfig.render and self.rank==0: self.advanced_render(self.raw_obs["sim_time"], win, self.observer)
        # system get done
        assert self.raw_obs is not None, ("obs is None")
        converted_obs = self.convert_obs(self.raw_obs)
        reward = 0
        info = {'win':win[self.player_color]}
        return converted_obs, reward, done, info

    def reset(self):
        # randomly switch sides
        if np.random.rand() < 0.5: self.player_color, self.opp_color = ("red", "blue")
        else: self.player_color, self.opp_color = ("blue", "red")

        # initialize controllers
        # controller 1
        self.opp_controller = self.OPP_CONTROLLER_CLASS(self.opp_color, {"side": self.opp_color})
        # controller 2
        self.fakeplayer_controller = self.OPP_CONTROLLER_CLASS(self.player_color, {"side": self.player_color})

        # reset signal to base
        self.raw_obs = self.bvr_reset()

        # read obs after reset
        converted_obs = self.convert_obs(self.raw_obs)
        
        info = self.raw_obs[self.player_color]
        return converted_obs, info

