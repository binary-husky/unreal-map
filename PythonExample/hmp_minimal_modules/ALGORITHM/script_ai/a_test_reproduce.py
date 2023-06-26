import numpy as np
from UTIL.colorful import *
from UTIL.tensor_ops import my_view, __hash__
from config import GlobalConfig
from MISSION.uhmap.actionset_v3 import strActionToDigits, ActDigitLen

class AlgorithmConfig:
    preserve = ''

class DummyAlgorithmBase():
    def __init__(self, n_agent, n_thread, space, mcv=None, team=None):
        self.n_agent = n_agent
        self.n_thread = n_thread
        self.ScenarioConfig = GlobalConfig.ScenarioConfig
        self.attack_order = {}
        self.team = team

    def forward(self, inp, state, mask=None):
        raise NotImplementedError

    def to(self, device):
        return self
    
    def interact_with_env(self, State_Recall):
        assert State_Recall['Latest-Obs'] is not None, ('make sure obs is ok')

        ENV_PAUSE = State_Recall['ENV-PAUSE']
        ENV_ACTIVE = ~ENV_PAUSE

        assert self.n_thread == len(ENV_ACTIVE), ('the number of thread is wrong?')
        
        n_active_thread = sum(ENV_ACTIVE)

        # assert len(State_Recall['Latest-Obs']) == n_active_thread, ('make sure we have the right batch of obs')
        actions = np.zeros(shape=(self.n_thread, self.n_agent, ActDigitLen))

        # set actions of in-active threads to NaN (will be done again in multi_team.py, this line is not necessary)
        actions[ENV_PAUSE] = np.nan

        # swap (self.n_thread, self.n_agent) -> (self.n_agent, self.n_thread) 
        actions = np.swapaxes(actions, 0, 1)
        return actions, {}


pre_def_color = [
    '(R=1,G=0,B=0,A=1)', 
    '(R=0,G=1,B=0,A=1)', 
    '(R=0,G=0,B=1,A=1)', 
]
# sel_l = [-8, -7, -8, -5, -4, -4, -4, -2, 0, -3, -1, -1, 3, 2, 2, 5, 4, 5, 6, 6, -8, -6, -7, -3, -4, -5, -2, -3, -2, 0, 0, 0, 3, 4, 2, 2, 3, 4, 8, 7, -7, -6, -7, -5, -4, -3, -4, -2, -1, -2, -1, 2, 2, 1, 1, 4, 4, 5, 8, 4, -7, -5, -4, -5, -3, -3, -2, -2, -3, 0, -1, 0, 1, 2, 3, 3, 5, 5, 4, 7, -8, -5, -6, -5, -3, -4, -3, -4, -3, -2, 0, 0, 3, 2, 4, 5, 4, 5, 5, 8, -7, -6, -4, -4, -4, -5, -3, -3, -3, -1, 0, 1, 0, 3, 3, 5, 3, 6, 7, 6, -7, -5, -6, -6, -1, -4, -2, -1, -1, 0, -1, 1, 1, 2, 3, 4, 5, 4, 4, 5, -7, -4, -6, -6, -3, -3, -1, -2, 0, -1, 0, -1, 3, 1, 2, 3, 3, 1, 7, 7, -5, -7, -6, -4, -3, -3, -2, 0, 0, -2, 0, 1, 1, 0, 3, 1, 2, 3, 4, 6, -7, -5, -4, -5, -4, -3, -3, -2, -2, 1, 0, 2, 0, 2, 1, 4, 5, 4, 4, 5]
sel_l = [-8, -7, -8, -5, -4, -4, -4, -2, 0, -3, -1, -1, 3, 2, 2, 5, 4, 5, 6, 6, -8, -6, -7, -3, -4, -5, -2, -3, -2, 0, 0, 0, 3, 4, 2, 2, 3, 4, 8, 7, -7, -6, -7, -5, -4, -3, -4, -2, -1, -2, -1, 2, 2, 1, 1, 4, 4, 5, 8, 4, -7, -5, -4, -5, -3, -3, -2, -2, -3, 0, -1, 0, 1, 2, 3, 3, 5, 5, 4, 7, -8, -5, -6, -5, -3, -4, -3, -4, -3, -2, 0, 0, 3, 2, 4, 5, 4, 5, 5, 8, -7, -6, -4, -4, -4, -5, -3, -3, -3, -1, 0, 1, 0, 3, 3, 5, 3, 6, 7, 6, -7, -5, -6, -6, -1, -4, -2, -1, -1, 0, -1, 1, 1, 2, 3, 4, 5, 4, 4, 5, -7, -4, -6, -6, -3, -3, -1, -2, 0, -1, 0, -1, 3, 1, 2, 3, 3, 1, 7, 7, -5, -7, -6, -4, -3, -3, -2, 0, 0, -2, 0, 1, 1, 0, 3, 1, 2, 3, 4, 6, -7, -5, -4, -5, -4, -3, -3, -2, -2, 1, 0, 2, 0, 2, 1, 4, 5, 4, 4, 5]
class TestReproduce(DummyAlgorithmBase):
    def interact_with_env(self, State_Recall):
        assert State_Recall['Latest-Obs'] is not None, ('make sure obs is ok')
        ENV_PAUSE = State_Recall['ENV-PAUSE']
        ENV_ACTIVE = ~ENV_PAUSE
        assert self.n_thread == len(ENV_ACTIVE), ('the number of thread is wrong?')

        n_active_thread = sum(ENV_ACTIVE)
        AirCarrierUID = 2

        # assert len(State_Recall['Latest-Obs']) == n_active_thread, ('make sure we have the right batch of obs')

        actions = np.zeros(shape=(self.n_thread, self.n_agent, ActDigitLen))
        self_agent_uid_range = GlobalConfig.ScenarioConfig.AGENT_ID_EACH_TEAM[self.team]

        for thread in range(self.n_thread):
            if ENV_PAUSE[thread]: 
                continue
            # sel_l = [] # 1
            # x_arr = np.array([d['agentLocationArr'][0] for d in np.array(State_Recall['Latest-Team-Info'][thread]['dataArr'])[self_agent_uid_range]]) # 1
            for a in range(self.n_agent):
                sel = sel_l[a] # 2
                # sel = (x_arr[a] + 35) // 70 # 1
                # sel_l.append(int(sel)) # 1
                actions[thread, a] = strActionToDigits(f'ActionSet1::ChangeColor;{pre_def_color[int(sel)%3]}')


        # set actions of in-active threads to NaN (will be done again in multi_team.py, this line is not necessary)
        actions[ENV_PAUSE] = np.nan
        return actions, {}