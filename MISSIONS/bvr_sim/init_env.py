import os, sys, math
import numpy as np
from .env.xsim_manager import XSimManager
from .env.communication_service import CommunicationService
from .env.env_runner import EnvRunner
from .agent.env_cmd import CmdEnv
from .render import RenderBridge
from .converter import Converter
from .endgame import EndGame
from UTILS.tensor_ops import MayGoWrong, repeat_at
from UTILS.config_args import ChainVar

class ScenarioConfig(object):  
    '''
        ScenarioConfig: This config class will be 'injected' with new settings from JSONC.
        (E.g., override configs with ```python main.py --cfg example.jsonc```)
        (As the name indicated, ChainVars will change WITH vars it 'chained_with' during config injection)
        (please see UTILS.config_args to find out how this advanced trick works out.)
    '''
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
    ObsAsUnity = False
    # ObsAsUnity = True

    ################ needed by env itself ################
    logging_enable = False
    render = False
    MaxEpisodeStep = 1000

    # SpeedLevels = 5 # the speed is divided as 5 discrete speed
    # meanning [MinSpeed, , MaxSpeed]  
    # MinSpeed + (MaxSpeed - MinSpeed)*( i/(SpeedLevels-1) ), i = [0,1,2,3,4]
    # HeightLevels = 5 # the speed is divided as 5 discrete speed
    # meanning [MinHeight , MaxHeight]  
    # MinHeight + (MaxHeight - MinHeight)*( i/(HeightLevels-1) ), i = [0,1,2,3,4]

    internal_step = 3

    ################ needed by some ALGORITHM ################
    StateProvided = True
    AvailActProvided = True
    EntityOriented = False

    put_availact_into_obs = True
    '''
        raw_action= [AT, TT, HT, SP]
    '''
    use_simple_action_space = True
    n_action_dimension = 2

    n_switch_actions = 7
    n_target_actions = 10
    n_actions = n_switch_actions +  n_target_actions
    # n_target = 10 # 暂时只有5+5，测试中
    # str_action_description = "[gym.spaces.Discrete(%d), gym.spaces.Discrete(%d), gym.spaces.Discrete(%d), gym.spaces.Discrete(%d)]"%(

    reduce_ipc_io = False

    obs_vec_length = 6
    return_mat = False
    AvailActProvided = True # sc2 中，需要始终屏蔽掉不可用的动作

    time_ratio = 200

    Disable_AT0 = False     # idle
    Disable_AT1 = False     # fly to target
    Disable_AT2 = True      # fly away from target
    Disable_AT3 = True      # fly 3 clock from target
    Disable_AT4 = True      # fly 9 clock from target
    Disable_AT5 = False     # fire to target
    Disable_AT6 = False      # change speed

    Oppo_Bits_Mask = np.array([False]*n_switch_actions+[True]*5+[False]*5)
    Ally_Bits_Mask = np.array([False]*n_switch_actions+[False]*5+[True]*5)

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
        self.xsim_manager = XSimManager(time_ratio=ScenarioConfig.time_ratio)
        # 与xsim引擎交互通信服务
        self.communication_service = CommunicationService(self.xsim_manager.address)

        # score of episode
        self.red_score = 0
        self.blue_score = 0

        # self.n_agents
        self.n_agents = 5
        self.n_opp = 5

        self.ScenarioConfig = ScenarioConfig
        self.n_actions = ScenarioConfig.n_actions
        self.n_switch_actions = ScenarioConfig.n_switch_actions
        self.n_target_actions = ScenarioConfig.n_target_actions
        # self.HeightLevels = ScenarioConfig.HeightLevels
        # self.SpeedLevels = ScenarioConfig.SpeedLevels
        # the id of env among parallel envs, only rank=0 thread is allowed to render
        self.rank = rank

        self.RewardAsUnity = ScenarioConfig.RewardAsUnity
        self.ObsAsUnity = ScenarioConfig.ObsAsUnity

        self.AT_SEL = 0
        self.TT_SEL = 1

        self.n_action_dimension = ScenarioConfig.n_action_dimension

        self.observation_space = {'state_shape': self.convert_obs_as_unity(obs=None, get_size=True), 
                                  'obs_shape':   self.convert_obs_individual(obs=None, get_size=True)}

        self.action_space =  {'n_actions': ScenarioConfig.n_actions,
                              'n_agents':  self.n_agents}

    def bvr_step(self, cmd_list):
        self.raw_obs = self.communication_service.step(cmd_list)
        # an observer keeps track of agent info such as distence
        reward_related_info = self.observer.observe(self.raw_obs["sim_time"], self.raw_obs['red'])
        done, red_win, blue_win = self.get_done(self.raw_obs)
        done = (done>0); 
        win = {'red':(red_win>0), 'blue':(blue_win>0), 'done':done}
        reward_dict = self.get_reward(reward_related_info, win)
        return self.raw_obs, done, win, reward_dict

    # @MayGoWrong
    def get_reward(self, reward_related_info, win):
        '''
            my = red
            op = blue
            'my_plane_fallen':0
            'op_plane_fallen':0
            'my_ms_miss_dead':0
            'op_ms_miss_dead':0
            'my_ms_miss_alive':0
            'op_ms_miss_alive':0
            'my_ms_hit':0
            'op_ms_hit':0
        '''
        R1_self_plane_fallen = -0.1
        R2_enem_plane_fallen = +1
        R3_self_ms_hit = 0
        R4_enem_ms_hit = -0.1
        R5_self_ms_miss_alive = -0.1
        R6_enem_ms_miss_alive = 0

        R7_win = 5
        R8_lose = -5

        red_reward = (
            reward_related_info['my_plane_fallen']  * R1_self_plane_fallen + 
            reward_related_info['op_plane_fallen']  * R2_enem_plane_fallen + 
            reward_related_info['my_ms_hit']        * R3_self_ms_hit + 
            reward_related_info['op_ms_hit']        * R4_enem_ms_hit + 
            reward_related_info['my_ms_miss_alive'] * R5_self_ms_miss_alive + 
            reward_related_info['op_ms_miss_alive'] * R6_enem_ms_miss_alive
        )


        blue_reward = (
            reward_related_info['op_plane_fallen']  * R1_self_plane_fallen + 
            reward_related_info['my_plane_fallen']  * R2_enem_plane_fallen + 
            reward_related_info['op_ms_hit']        * R3_self_ms_hit + 
            reward_related_info['my_ms_hit']        * R4_enem_ms_hit + 
            reward_related_info['op_ms_miss_alive'] * R5_self_ms_miss_alive + 
            reward_related_info['my_ms_miss_alive'] * R6_enem_ms_miss_alive
        )
        if win['done']:
            red_reward  += R7_win if win['red']  else R8_lose
            blue_reward += R7_win if win['blue'] else R8_lose
        # obs[self.player_color], obs[self.opp_color]
        return {'red': red_reward, 'blue': blue_reward}

    def bvr_reset(self):
        try:
            self._end_()
            self.communication_service.reset()
            self.raw_obs = self.communication_service.step([])
            # observer's first info feedin
            self.observer = self.OBSERVER_CLASS('red', {"side": 'red'})
            self.observer.observe(self.raw_obs["sim_time"], self.raw_obs['red'])
        except:
            self.xsim_manager.__del__()
            self.xsim_manager = None
            self.communication_service = None
            import gc
            gc.collect()
            
            # xsim引擎控制器
            self.xsim_manager = XSimManager(time_ratio=ScenarioConfig.time_ratio)
            # 与xsim引擎交互通信服务
            self.communication_service = CommunicationService(self.xsim_manager.address)

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
        self.check_avail_act = None
        self.advanced_action_context = {}

    def step(self, raw_action):
        # convert raw_action Tensor from RL to actions recognized by the game
        player_action, next_avail_act = self.parse_raw_action(raw_action)
        self.check_avail_act = next_avail_act.copy()
        
        reward_accu_internal = 0
        for _ in range(self.internal_step):
            # import time, cProfile, pstats
            # tic = time.time()
            # def x():
            #     for i in range(500):
            #         opp_action = self.opp_controller.step(i, self.raw_obs[self.opp_color])
            # cProfile.runctx("x()", globals(), locals(), filename="result.prof")
            # print('time：：：：', time.time()-tic)
            # raise 'done profile'

            # # move the opponents
            opp_action = self.opp_controller.step(self.raw_obs["sim_time"], self.raw_obs[self.opp_color])

            # commmit actions to the environment
            cmd_list = player_action + opp_action
            self.raw_obs, done, win, reward_dict = self.bvr_step(cmd_list)

            # accumulate player reward
            reward_accu_internal += reward_dict[self.player_color]
            
            if done:  break
            player_action = []

        # use observer's info to render
        # if self.ScenarioConfig.render and self.rank==0: self.advanced_render(self.raw_obs["sim_time"], win, self.observer)
        if self.ScenarioConfig.render: self.advanced_render(self.raw_obs["sim_time"], win, self.observer)

        # system get done
        assert self.raw_obs is not None, ("obs is None")

        if self.ObsAsUnity:
            converted_obs = self.convert_obs_as_unity(self.raw_obs)
        else:
            converted_obs = self.convert_obs_individual(self.raw_obs, next_avail_act=next_avail_act)

        info = {
            'win':win[self.player_color],
            'state':self.convert_obs_as_unity(self.raw_obs),
            'avail-act': next_avail_act
        }
        if not ScenarioConfig.reduce_ipc_io:
            info.update(self.raw_obs[self.player_color])
        if self.RewardAsUnity:
            return_reward = np.array([reward_accu_internal,])
        if self.ObsAsUnity:
            return_obs = np.array([converted_obs, ])
        else:
            return_obs = converted_obs
        return return_obs, return_reward, done, info




    def reset(self):
        
        # initialize action context if using simple action space
        self.action_context = np.zeros(shape=(self.n_agents, self.n_action_dimension))
        for i in range(self.n_agents): self.action_context[i, self.AT_SEL] = 0
        for i in range(self.n_agents): self.action_context[i, self.TT_SEL] = i
        self.advanced_action_context = {}

        # randomly switch sides
        randx = np.random.rand() 
        if randx < 0.5: self.player_color, self.opp_color = ("red", "blue")
        else: self.player_color, self.opp_color = ("blue", "red")

        # initialize controllers
        self.opp_controller = self.OPP_CONTROLLER_CLASS(self.opp_color, {"side": self.opp_color})

        # reset signal to base
        self.raw_obs = self.bvr_reset()

        # info = {'win':win[self.player_color]}info['avail-act']
        action_dim_corr_dict, _ = self.get_action_dim_corr_dict()
        next_avail_act = repeat_at(1-action_dim_corr_dict, 0, n_times=self.n_agents)
        self.check_avail_act = next_avail_act.copy()

        # read obs after reset
        if self.ObsAsUnity:
            converted_obs = self.convert_obs_as_unity(self.raw_obs)
        else:
            converted_obs = self.convert_obs_individual(self.raw_obs, next_avail_act=next_avail_act)

        info = {
            'state':self.convert_obs_as_unity(self.raw_obs),
            # first action is limited in AT
            'avail-act': next_avail_act
        }
        if not ScenarioConfig.reduce_ipc_io:
            info.update(self.raw_obs[self.player_color])

        if hasattr(self, '可视化桥'): self.可视化桥.刚刚复位 = True

        if self.ObsAsUnity:
            return_obs = np.array([converted_obs, ])
        else:
            return_obs = converted_obs
        return return_obs, info



'''
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

        self.可视化桥.刚刚复位 = True
        return converted_obs, info

'''


