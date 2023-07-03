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

class EscapeGreenPreprogramBaseline(DummyAlgorithmBase):
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
                # 如果,该线程停止，不做任何处理
                continue
            x_arr = np.array([d['agentLocationArr'][0] for d in np.array(State_Recall['Latest-Team-Info'][thread]['dataArr'])[self_agent_uid_range]])
            x_arr_valid = np.array([x for x in x_arr if np.isfinite(x)])
            x_avg = x_arr_valid.mean()
            for index, x in enumerate(x_arr):
                if not np.isfinite(x): pass

                actions[thread, index] = strActionToDigits(f'ActionSet2::SpecificAttacking;T1-0')

            # actions[thread, :] = strActionToDigits(f'ActionSet2::SpecificAttacking;T1-0')

        # set actions of in-active threads to NaN (will be done again in multi_team.py, this line is not necessary)
        actions[ENV_PAUSE] = np.nan
        return actions, {}
    
class EscapeRedPreprogramBaseline(DummyAlgorithmBase):
    def interact_with_env(self, State_Recall):
        assert State_Recall['Latest-Obs'] is not None, ('make sure obs is ok')
        ENV_PAUSE = State_Recall['ENV-PAUSE']
        ENV_ACTIVE = ~ENV_PAUSE
        assert self.n_thread == len(ENV_ACTIVE), ('the number of thread is wrong?')

        actions = np.zeros(shape=(self.n_thread, self.n_agent, ActDigitLen))
        self_agent_uid_range = GlobalConfig.ScenarioConfig.AGENT_ID_EACH_TEAM[self.team]

        for thread in range(self.n_thread):
            if ENV_PAUSE[thread]: 
                # 如果,该线程停止，不做任何处理
                continue
            x_arr = np.array([d['agentLocationArr'][0] for d in np.array(State_Recall['Latest-Team-Info'][thread]['dataArr'])[self_agent_uid_range]])
            x_arr_valid = np.array([x for x in x_arr if np.isfinite(x)])
            x_avg = x_arr_valid.mean()
            for index, x in enumerate(x_arr):
                if not np.isfinite(x): pass
                if index <2:
                    actions[thread, index] = strActionToDigits(f'ActionSet4::MoveToDirection;X=-1.0 Y=0.0 Z=0.0')
                else:
                    actions[thread, index] = strActionToDigits(f'ActionSet4::MoveToDirection;X=+1.0 Y=0.0 Z=0.0')

        # set actions of in-active threads to NaN (will be done again in multi_team.py, this line is not necessary)
        actions[ENV_PAUSE] = np.nan
        return actions, {}