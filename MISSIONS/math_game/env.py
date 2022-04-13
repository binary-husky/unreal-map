from lib2to3.pytree import Base
import os, sys, math
from cv2 import Algorithm
import numpy as np
from UTILS.tensor_ops import repeat_at, np_one_hot
from UTILS.config_args import ChainVar

'''
    ScenarioConfig: This config class will be 'injected' with new settings from JSONC.
    (E.g., override configs with ```python main.py --cfg example.jsonc```)
    (As the name indicated, ChainVars will change WITH vars it 'chained_with' during config injection)
    (please see UTILS.config_args to find out how this advanced trick works out.)
'''
class ScenarioConfig(object): # ADD_TO_CONF_SYSTEM 加入参数搜索路径 do not remove this comment !!!
    # <Part 1> Needed by the hmp core #
    n_team1agent = 5

    N_TEAM = 1  

    N_AGENT_EACH_TEAM = [n_team1agent,]
    AGENT_ID_EACH_TEAM = [range(0,n_team1agent),]

    # Establish chain reaction
    N_AGENT_EACH_TEAM_cv = ChainVar(lambda n_team1agent: [n_team1agent,], chained_with=['n_team1agent'])
    AGENT_ID_EACH_TEAM_cv = ChainVar(lambda n_team1agent: [range(0,n_team1agent),], chained_with=['n_team1agent'])

    TEAM_NAMES = ['ALGORITHM.None->None',]
    '''
        ## If the length of action array == the number of teams, set ActAsUnity to True
        ## If the length of action array == the number of agents, set ActAsUnity to False
    '''
    ActAsUnity = False

    '''
        ## If the length of reward array == the number of agents, set RewardAsUnity to False
        ## If the length of reward array == 1, set RewardAsUnity to True
    '''
    RewardAsUnity = True

    '''
        ## If the length of obs array == the number of agents, set ObsAsUnity to False
        ## If the length of obs array == the number of teams, set ObsAsUnity to True
    '''
    ObsAsUnity = False

    # <Part 2> Needed by env itself #
    MaxEpisodeStep = 10

    # <Part 3> Needed by some ALGORITHM #
    StateProvided = False
    AvailActProvided = False
    EntityOriented = False

    n_actions = 2

    #
    show_details = False

    # MathGame Setting
    one_hot_id = True


class BaseEnv(object):
    def __init__(self, rank) -> None:
        self.observation_space = None
        self.action_space = None
        self.rank = rank

    def step(self, act):
        # obs: a Tensor with shape (n_agent, ...)
        # reward: a Tensor with shape (n_agent, 1) or (n_team, 1)
        # done: a Bool
        # info: a dict
        raise NotImplementedError
        # Warning: if you have only one team and RewardAsUnity, 
        # you must make sure that reward has shape=[n_team=1, 1]
        # e.g. 
        # >> RewardForTheOnlyTeam = +1
        # >> RewardForAllTeams = np.array([RewardForTheOnlyTeam, ])
        # >> return (ob, RewardForAllTeams, done, info)
        return (ob, RewardForAllTeams,  done, info)  # choose this if RewardAsUnity
        return (ob, RewardForAllAgents, done, info)  # choose this if not RewardAsUnity

    def reset(self):
        # obs: a Tensor with shape (n_agent, ...)
        # done: a Bool
        raise NotImplementedError
        return ob, info

class MathEnv(BaseEnv):
    def __init__(self, rank) -> None:
        super().__init__(rank)
        self.n_agents = ScenarioConfig.n_team1agent

        self.observation_space = {
            'state_shape': None, 
            'obs_shape': self.get_obs(TS=None, get_size=True)}

        self.action_space = {
            'n_actions': ScenarioConfig.n_actions,
            'n_agents':  self.n_agents}
        
        if ScenarioConfig.StateProvided:
            self.observation_space['state_shape'] = 1
        
        self.show_details = (ScenarioConfig.show_details and rank==0)
        if self.show_details:
            from VISUALIZE.mcom import mcom
            from config import GlobalConfig as cfg
            logdir = cfg.logdir
            self.mcv = mcom(
                        path='%s/logger_mathgame/'%logdir,
                        digit=16,
                        rapid_flush=True,
                        draw_mode=cfg.draw_mode,
                        tag='[mathgame/env.py]',
                        image_path='%s/rec_mathgame.jpg'%logdir)
            self.mcv.rec_init('g')

    def reset(self):
        self.TS = 0
        ob = self.get_obs(TS=0)
        info = {}
        if ScenarioConfig.StateProvided:
            info['state'] = np.array([self.TS])
        return ob, info

    def get_obs(self, TS, get_size=False):
        core_dim = self.n_agents + 1
        if get_size: return core_dim

        ob = ob_part1 = np.zeros((self.n_agents, 1), dtype=np.float32) + TS
        if ScenarioConfig.one_hot_id:
            ob_part2 = np.eye(self.n_agents, dtype=np.float32)
            ob = np.concatenate((ob, ob_part2), -1)

        return ob

    def step(self, act):
        nLevel = 3
        info = {}
        if self.show_details:

            for i in range(ScenarioConfig.n_actions):
                self.mcv.rec(sum(act==i), 'Ts=%d, Act=%d'%(self.TS, i))
            self.mcv.rec_show()

        if self.TS==0:  # Level 1
            # we expect all but one agents choose act1, only one agent choose act0
            # reward: 
            #   - Any agent choose act0: Team Reward +1
            #   - No agent choose act0: Team Reward +0
            n_chosen_act0 = sum(act==0)
            reward = n_chosen_act0/self.n_agents

            self.TS = 1
            ob = self.get_obs(TS=1)
            if ScenarioConfig.StateProvided:
                info['state'] = np.array([1])
        elif self.TS==1:  # Level 2
            # 尽可能多的智能体选择act1
            # 但如果没有智能体选择act0，奖励-1
            n_chosen_act1 = sum(act==1)
            reward = n_chosen_act1/self.n_agents
            # if (self.n_agents-n_chosen_act1)==0:
            #     reward = -1

            self.TS = 2
            ob = self.get_obs(TS=2)
            if ScenarioConfig.StateProvided:
                info['state'] = np.array([2])

        elif self.TS==2:  # Level 3
            # 尽可能多的智能体选择act0
            # 但如果没有智能体选择act1，奖励-1
            n_chosen_act0 = sum(act==0)
            reward = n_chosen_act0/self.n_agents
            # if (self.n_agents-n_chosen_act0)==0:
            #     reward = -1
                
            self.TS = 3
            ob = self.get_obs(TS=3) # Terminal obs, won't be accepted
            if ScenarioConfig.StateProvided:
                info['state'] = np.array([2])
        else:
            assert False, 'Should not be here !'

        # obs: a Tensor with shape (n_agent, ...)
        if self.TS >= nLevel: 
            done = True
        else: 
            done = False
        # 
        info.update({'win':False})
        reward_allteam = np.array([reward])
        return (ob, reward_allteam, done, info)
    

def make_math_env(env_id, rank):
    return MathEnv(rank)

'''

    if self.TS==0:
        # we expect all but one agents choose act1, only one agent choose act0
        # reward: 
        #   - Any agent choose act0: Team Reward +1
        #   - No agent choose act0: Team Reward +0
        reward = n_chosen_act0/self.n_agents

    Results:
          act0    act1
        [[9.9981e-01, 1.9052e-04],
         [9.9962e-01, 3.7964e-04],
         [9.9970e-01, 2.9723e-04],
         [9.9981e-01, 1.9268e-04],
         [9.9986e-01, 1.4075e-04]],
     _____________________________________________________________________________

        elif self.TS==1:  # Level 2
            # we expect all but one agents choose act1, only one agent choose act0
            # reward: 
            #   - Any agent choose act0: Team Reward +1
            #   - No agent choose act0: Team Reward +0
            reward = -1 if n_chosen_act0==0 else +1
            self.TS = 2
            ob = self.get_obs(TS=2)
        [[0.7883, 0.2117],
         [0.5876, 0.4124],
         [0.7635, 0.2365],
         [0.8501, 0.1499],
         [0.8389, 0.1611]],

     _____________________________________________________________________________

        elif self.TS==2:  # Level 3
            # we expect all but one agents choose act0, only one agent choose act1
            # reward: 
            #   - Any agent choose act1: Team Reward +1
            #   - No agent choose act1: Team Reward +0
            reward = -1 if n_chosen_act1==0 else +1
            self.TS = 3

          act0    act1
        [[0.3068, 0.6932],
         [0.3439, 0.6561],
         [0.3616, 0.6384],
         [0.3519, 0.6481],
         [0.3549, 0.6451]],
'''