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

sel_l =  [
    [-8, -8, -4, -3, -5, -5, -4, -2, 0, 0, 1, 2, 3, 0, 4, 4, 3, 5, 6, 8, -7, -5, -6, -6, -3, -3, -2, -1, -1, 0, 1, 1, 0, 4, 5, 6, 5, 6, 6, 4, -7, -6, -5, -6, -3, -4, -3, -2, -1, 0, 0, -1, 0, 1, 4, 3, 5, 6, 5, 6, -7, -8, -5, -4, -4, -2, -1, 0, 0, -1, 0, 1, 2, 0, 3, 5, 2, 4, 4, 8, -7, -6, -5, -6, -3, -4, 0, -2, -1, 0, -1, -1, 1, 2, 1, 2, 6, 4, 3, 5, -7, -6, -5, -4, -4, -3, -3, -2, -3, 1, 1, 1, 3, 2, 2, 6, 5, 3, 5, 7, -5, -5, -2, -3, -4, -2, -4, -1, 0, -1, 0, 0, 2, 1, 5, 1, 2, 3, 6, 6, -7, -6, -7, -5, -4, -1, -2, -5, -2, -1, -1, 1, 1, 4, 3, 4, 4, 4, 5, 7, -8, -5, -4, -2, -3, -1, 0, -1, 1, -1, 2, -1, 3, 1, 0, 2, 5, 4, 4, 5, -6, -5, -3, -4, -3, -4, -2, 0, -1, -3, 0, 2, 2, -1, 2, 5, 5, 3, 5, 4],
    [-5, -6, -3, -4, -3, -4, -2, -2, -1, 1, -1, 2, 1, 2, 5, 4, 3, 2, 5, 8, -8, -8, -6, -2, -3, -2, -3, -1, 1, -2, 2, 1, 1, 3, 3, 3, 3, 4, 7, 6, -6, -5, -5, -7, -2, -2, -2, -4, -2, -1, 0, 1, 2, 2, 5, 3, 7, 5, 4, 7, -8, -8, -3, -4, -4, -4, -3, -3, -2, 1, -2, 1, 2, 1, 2, 4, 4, 5, 6, 7, -7, -6, -4, -3, -4, -3, -1, 1, -1, 0, 0, 0, 4, 2, 2, 3, 4, 5, 5, 5, -5, -5, -5, -3, -2, -2, -3, -2, -1, 0, 0, 2, 3, 3, 3, 2, 5, 5, 4, 6, -8, -6, -6, -3, -3, -2, 0, -2, -1, 2, 0, 2, 2, 2, 3, 2, 1, 4, 4, 7, -8, -6, -6, -3, -2, -3, -2, -1, 1, -1, 2, 3, 1, 2, 2, 3, 2, 3, 3, 8, -6, -6, -5, -2, -2, -2, -2, -1, 0, -1, -1, 2, 3, 2, 0, 3, 3, 5, 6, 8, -7, -5, -3, -5, -5, -4, -1, -2, 0, 0, 1, 1, 0, 1, 1, 3, 4, 3, 3, 5],
    [-8, -8, -5, -6, -1, -2, -2, 0, -2, -2, 0, 2, 2, 2, 5, 2, 3, 6, 7, 6, -8, -8, -4, -5, -4, -5, -2, -1, -1, -1, 1, 0, 3, 1, 3, 5, 5, 7, 5, 7, -7, -6, -5, -5, -7, -2, -1, 0, -1, -2, 1, 1, 0, 1, 3, 3, 6, 4, 5, 7, -8, -7, -6, -4, -3, -3, -2, -1, -1, -1, 0, 1, 0, 0, 3, 3, 4, 5, 5, 8, -6, -5, -6, -3, -4, -3, -3, -2, -1, 1, 0, 0, 1, 2, 2, 4, 5, 5, 4, 5, -8, -4, -7, -6, -3, -2, -3, -3, 1, 0, 0, 1, 1, 2, 2, 4, 4, 5, 6, 6, -5, -5, -3, -5, -4, -4, -1, -1, -1, -1, 0, 1, 4, 4, 6, 3, 4, 4, 5, 7, -6, -7, -5, -4, -3, -4, -1, -2, 0, -1, 1, 1, 1, 3, 2, 3, 4, 3, 3, 5, -7, -8, -5, -5, -3, -3, -3, -3, -2, 0, 0, 2, 1, 2, 3, 2, 3, 4, 7, 6, -8, -5, -4, -4, -4, -4, -1, -4, 0, -1, 1, 0, 0, 1, 4, 1, 3, 4, 6, 6],
    [-7, -6, -4, -6, -4, -4, -4, -2, -2, 1, -1, 1, 3, 3, 3, 4, 3, 6, 6, 8, -7, -5, -6, -7, -4, -3, -4, -2, 0, -1, 0, 2, 2, 0, 3, 4, 5, 5, 6, 7, -7, -6, -7, -3, -4, -3, -1, -5, -1, 0, -1, 1, 1, 2, 2, 3, 5, 5, 8, 6, -6, -6, -6, -4, -3, -2, -4, -2, -2, 2, 1, 1, 2, 0, 3, 4, 5, 5, 5, 7, -7, -5, -4, -3, -7, -2, -2, -2, -1, 0, 1, 1, 1, 4, 4, 4, 5, 4, 4, 6, -5, -5, -5, -4, -2, -3, -4, -1, 0, -1, -2, 1, 0, 2, 3, 3, 5, 6, 7, 6, -7, -5, -5, -2, -3, -3, -3, 1, 0, -2, 0, -1, 2, 2, 3, 4, 4, 4, 6, 7, -8, -6, -6, -4, -4, -2, -2, -2, -2, -1, 0, 0, 1, 2, 2, 3, 4, 2, 5, 5, -6, -4, -5, -4, -3, -3, -1, -1, -2, -2, 0, 1, 2, 2, 4, 5, 6, 5, 6, 5, -7, -5, -4, -2, -3, -4, -2, -2, -2, -1, 2, 1, 1, 2, 3, 3, 4, 4, 4, 6],
]
class TestReproduce(DummyAlgorithmBase):
    def __init__(self, n_agent, n_thread, space, mcv=None, team=None):
        super().__init__(n_agent, n_thread, space, mcv, team)
        self.episode = -1
    def interact_with_env(self, State_Recall):
        assert State_Recall['Latest-Obs'] is not None, ('make sure obs is ok')
        ENV_PAUSE = State_Recall['ENV-PAUSE']
        RST = State_Recall['Env-Suffered-Reset']
        ENV_ACTIVE = ~ENV_PAUSE
        assert self.n_thread == len(ENV_ACTIVE), ('the number of thread is wrong?')

        if all(RST):
            self.episode += 1
        # assert len(State_Recall['Latest-Obs']) == n_active_thread, ('make sure we have the right batch of obs')

        actions = np.zeros(shape=(self.n_thread, self.n_agent, ActDigitLen))
        self_agent_uid_range = GlobalConfig.ScenarioConfig.AGENT_ID_EACH_TEAM[self.team]

        for thread in range(self.n_thread):
            if ENV_PAUSE[thread]: 
                continue
            sel_l_ = [] # 1
            x_arr_ = np.array([d['agentLocationArr'][0] for d in np.array(State_Recall['Latest-Team-Info'][thread]['dataArr'])[self_agent_uid_range]]) # 1
            for a in range(self.n_agent):
                sel = sel_l[self.episode][a] # 2
                sel_ = (x_arr_[a] + 35) // 70 # 1
                sel_l_.append(int(sel_)) # 1
                actions[thread, a] = strActionToDigits(f'ActionSet1::ChangeColor;{pre_def_color[int(sel)%3]}')


        print(sel_l[self.episode][:10], sel_l_[:10])
        print(sel_l_)
        # set actions of in-active threads to NaN (will be done again in multi_team.py, this line is not necessary)
        actions[ENV_PAUSE] = np.nan
        return actions, {}

