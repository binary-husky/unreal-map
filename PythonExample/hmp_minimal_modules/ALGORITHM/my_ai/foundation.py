import numpy as np
from UTIL.colorful import *
from UTIL.tensor_ops import my_view, __hash__
from config import GlobalConfig
from MISSION.uhmap.actionset import strActionToDigits, ActDigitLen
class AlgorithmConfig:
    preserve = ''



class ReinforceAlgorithmFoundation(object):
    def __init__(self, n_agent, n_thread, space, mcv=None, team=None):
        self.n_agent = n_agent
        self.n_thread = n_thread
        self.space = space
        self.mcv = mcv

    def interact_with_env(self, StateRecall):
        obs = StateRecall['Latest-Obs']
        P = StateRecall['ENV-PAUSE']
        active_thread_obs = obs[~P]
        actions = np.zeros(shape=(self.n_thread, self.n_agent, ActDigitLen))

        for env_index in range(self.n_thread):
            for agent_index in range(self.n_agent):
                if np.random.rand() < 0.5:
                    color_index = np.random.randint(low=0, high=4)
                    actions[env_index, agent_index] = strActionToDigits(f'ActionSetDemo::ChangeColor;{color_index}')
                else:
                    uid = 11 if agent_index % 2 == 0 else 10
                    actions[env_index, agent_index] = strActionToDigits(f'ActionSetDemo::FireToWaterdrop;{uid}')


        StateRecall['_hook_'] = None
        return actions, StateRecall 




class DiscreteRLFoundation(object):
    def __init__(self, n_agent, n_thread, space, mcv=None, team=None):
        self.n_agent = n_agent
        self.n_thread = n_thread
        self.space = space
        self.mcv = mcv
        self.action_list = [
            strActionToDigits('ActionSetDemo::ChangeColor;0'),
            strActionToDigits('ActionSetDemo::ChangeColor;1'),
            strActionToDigits('ActionSetDemo::ChangeColor;2'),
            strActionToDigits('ActionSetDemo::ChangeColor;3'),
            strActionToDigits('ActionSetDemo::FireToWaterdrop;10'),
            strActionToDigits('ActionSetDemo::FireToWaterdrop;11'),
            strActionToDigits('ActionSetDemo::MoveToDirection;X=1.0 Y=0.0 Z=0.0'),
            strActionToDigits('ActionSetDemo::MoveToDirection;X=1.0 Y=1.0 Z=0.0'),
            strActionToDigits('ActionSetDemo::MoveToDirection;X=0.0 Y=1.0 Z=0.0'),
            strActionToDigits('ActionSetDemo::MoveToDirection;X=-1.0 Y=1.0 Z=0.0'),
            strActionToDigits('ActionSetDemo::MoveToDirection;X=-1.0 Y=0.0 Z=0.0'),
            strActionToDigits('ActionSetDemo::MoveToDirection;X=-1.0 Y=-1.0 Z=0.0'),
            strActionToDigits('ActionSetDemo::MoveToDirection;X=0.0 Y=-1.0 Z=0.0'),
            strActionToDigits('ActionSetDemo::MoveToDirection;X=1.0 Y=-1.0 Z=0.0'),
            strActionToDigits('ActionSetDemo::MoveToDirection;X=0.0 Y=0.0 Z=1.0'),
            strActionToDigits('ActionSetDemo::MoveToDirection;X=0.0 Y=0.0 Z=-1.0'),
        ]
        self.how_many_actions = len(self.action_list)

    def interact_with_env(self, StateRecall):
        obs = StateRecall['Latest-Obs']
        P = StateRecall['ENV-PAUSE']
        active_thread_obs = obs[~P]
        actions = np.zeros(shape=(self.n_thread, self.n_agent, ActDigitLen))

        for env_index in range(self.n_thread):
            for agent_index in range(self.n_agent):
                action_x = np.random.randint(low=0,high=self.how_many_actions)
                actions[env_index, agent_index] = self.action_list[action_x]


        StateRecall['_hook_'] = None
        return actions, StateRecall 
