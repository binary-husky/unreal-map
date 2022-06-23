import numpy as np
from UTILS.tensor_ops import copy_clone

class DummyAlgConfig():
    reserve = ""

class DummyAlgorithmBase():
    def __init__(self, n_agent, n_thread, space, mcv):
        from config import GlobalConfig
        self.n_agent = n_agent
        self.n_thread = n_thread
        self.scenario_config = GlobalConfig.scenario_config
        self.act2digit_dictionary = self.scenario_config.act2digit_dictionary

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

        actions = np.zeros(shape=(self.n_thread, self.n_agent))

        # set actions of in-active threads to NaN (will be done again in multi_team.py, this line is not necessary)
        actions[ENV_PAUSE] = np.nan
        
        # swap (self.n_thread, self.n_agent) -> (self.n_agent, self.n_thread) 
        actions = np.swapaxes(actions, 0, 1)
        return actions, {}


class DummyAlgorithmT1(DummyAlgorithmBase):
    def interact_with_env(self, State_Recall):
        assert State_Recall['Latest-Obs'] is not None, ('make sure obs is ok')
        
        ENV_PAUSE = State_Recall['ENV-PAUSE']
        ENV_ACTIVE = ~ENV_PAUSE

        assert self.n_thread == len(ENV_ACTIVE), ('the number of thread is wrong?')
        
        n_active_thread = sum(ENV_ACTIVE)

        # assert len(State_Recall['Latest-Obs']) == self.n_thread, ('make sure we have the right batch of obs')

        actions = np.zeros(shape=(self.n_thread, self.n_agent))

        env0_step = State_Recall['Current-Obs-Step']

        thread = 0

        # actions[thread, :] = self.act2digit_dictionary['ActionSet2::PatrolMoving;X=0.0 Y=0.0 Z=100.0']
        actions[:, :] = self.act2digit_dictionary['ActionSet2::PatrolMoving;Dir+Y']



        # set actions of in-active threads to NaN (will be done again in multi_team.py, this line is not necessary)
        actions[ENV_PAUSE] = np.nan
    
        # swap (self.n_thread, self.n_agent) -> (self.n_agent, self.n_thread) 
        actions = np.swapaxes(actions, 0, 1)
        return actions, {}

class DummyAlgorithmT2(DummyAlgorithmBase):
    def interact_with_env(self, State_Recall):
        assert State_Recall['Latest-Obs'] is not None, ('make sure obs is ok')
        
        ENV_PAUSE = State_Recall['ENV-PAUSE']
        ENV_ACTIVE = ~ENV_PAUSE

        assert self.n_thread == len(ENV_ACTIVE), ('the number of thread is wrong?')
        
        n_active_thread = sum(ENV_ACTIVE)

        # assert len(State_Recall['Latest-Obs']) == n_active_thread, ('make sure we have the right batch of obs')

        actions = np.zeros(shape=(self.n_thread, self.n_agent))

        env0_step = State_Recall['Current-Obs-Step']

        thread = 0

        actions[:, :] = self.act2digit_dictionary['ActionSet2::PatrolMoving;Dir-Y']

        # actions[thread, :] = self.act2digit_dictionary['ActionSet2::PatrolMoving;X=-6800.0 Y=6830.0 Z=397.215454']


        # actions[thread, 1] = self.act2digit_dictionary['ActionSet2::SpecificMoving;Dir-Y']
        # actions[thread, 2] = self.act2digit_dictionary['ActionSet2::SpecificMoving;Dir-Y']
        # actions[thread, 3] = self.act2digit_dictionary['ActionSet2::SpecificMoving;Dir-Y']

        # set actions of in-active threads to NaN (will be done again in multi_team.py, this line is not necessary)
        actions[ENV_PAUSE] = np.nan
    
        # swap (self.n_thread, self.n_agent) -> (self.n_agent, self.n_thread) 
        actions = np.swapaxes(actions, 0, 1)
        return actions, {}

'''
        if env0_step < 5:
            actions[thread, :] = self.act2digit_dictionary['ActionSet2::Idle;DynamicGuard']
        elif env0_step < 15:
            actions[thread, :] = self.act2digit_dictionary['ActionSet2::SpecificAttacking;UID-3']
        elif env0_step < 25:
            actions[thread, :] = self.act2digit_dictionary['ActionSet2::SpecificAttacking;UID-4']
        elif env0_step < 35:
            actions[thread, :] = self.act2digit_dictionary['ActionSet2::SpecificAttacking;UID-5']
        elif env0_step < 45:
            actions[thread, :] = self.act2digit_dictionary['ActionSet2::SpecificAttacking;UID-6']
        elif env0_step < 55:
            actions[thread, :] = self.act2digit_dictionary['ActionSet2::SpecificAttacking;UID-7']
'''

'''
        if env0_step < 5:
            actions[thread, 0] = self.act2digit_dictionary['ActionSet2::Idle;DynamicGuard']
        else:
            actions[thread, 0] = self.act2digit_dictionary['ActionSet2::SpecificAttacking;UID-1']
            

'''

'''
    if env0_step < 5:
        if env0_step%4 == 0:
            actions[thread, 0] = self.act2digit_dictionary['ActionSet2::PatrolMoving;Dir+X+Y']
        if env0_step%4 == 1:
            actions[thread, 0] = self.act2digit_dictionary['ActionSet2::PatrolMoving;Dir+X-Y']
        if env0_step%4 == 2:
            actions[thread, 0] = self.act2digit_dictionary['ActionSet2::PatrolMoving;Dir-X-Y']
        if env0_step%4 == 3:
            actions[thread, 0] = self.act2digit_dictionary['ActionSet2::PatrolMoving;Dir-X+Y']
    elif env0_step < 10:
        actions[thread, 0] = self.act2digit_dictionary['ActionSet2::Idle;DynamicGuard']
    elif env0_step < 15:
        if env0_step%4 == 0:
            actions[thread, 0] = self.act2digit_dictionary['ActionSet2::SpecificMoving;Dir+X']
        if env0_step%4 == 1:
            actions[thread, 0] = self.act2digit_dictionary['ActionSet2::SpecificMoving;Dir+Y']
        if env0_step%4 == 2:
            actions[thread, 0] = self.act2digit_dictionary['ActionSet2::SpecificMoving;Dir-X']
        if env0_step%4 == 3:
            actions[thread, 0] = self.act2digit_dictionary['ActionSet2::SpecificMoving;Dir-Y']
    elif env0_step < 20:
        actions[thread, 0] = self.act2digit_dictionary['ActionSet2::Idle;StaticAlert']
    elif env0_step < 30:
        if env0_step%4 == 0:
            actions[thread, 0] = self.act2digit_dictionary['ActionSet2::PatrolMoving;Dir+X+Y']
        if env0_step%4 == 1:
            actions[thread, 0] = self.act2digit_dictionary['ActionSet2::PatrolMoving;Dir+X-Y']
        if env0_step%4 == 2:
            actions[thread, 0] = self.act2digit_dictionary['ActionSet2::PatrolMoving;Dir-X-Y']
        if env0_step%4 == 3:
            actions[thread, 0] = self.act2digit_dictionary['ActionSet2::PatrolMoving;Dir-X+Y']
    else:
        actions[thread, 0] = self.act2digit_dictionary['ActionSet2::Idle;StaticAlert']

'''

"""
        thread = 0
        if env0_step%4 == 0:
            actions[thread, 0] = self.act2digit_dictionary['ActionSet2::PatrolMoving;Dir+X+Y']
        if env0_step%4 == 1:
            actions[thread, 0] = self.act2digit_dictionary['ActionSet2::PatrolMoving;Dir+X-Y']
        if env0_step%4 == 2:
            actions[thread, 0] = self.act2digit_dictionary['ActionSet2::PatrolMoving;Dir-X-Y']
        if env0_step%4 == 3:
            actions[thread, 0] = self.act2digit_dictionary['ActionSet2::PatrolMoving;Dir-X+Y']
"""

"""
        thread = 0
        if env0_step%4 == 0:
            actions[thread, 0] = self.act2digit_dictionary['ActionSet2::SpecificMoving;Dir+X+Y']
        if env0_step%4 == 1:
            actions[thread, 0] = self.act2digit_dictionary['ActionSet2::SpecificMoving;Dir+X-Y']
        if env0_step%4 == 2:
            actions[thread, 0] = self.act2digit_dictionary['ActionSet2::SpecificMoving;Dir-X-Y']
        if env0_step%4 == 3:
            actions[thread, 0] = self.act2digit_dictionary['ActionSet2::SpecificMoving;Dir-X+Y']
"""