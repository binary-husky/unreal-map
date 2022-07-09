import numpy as np
from UTIL.colorful import *
from UTIL.tensor_ops import my_view, __hash__, repeat_at
from .foundation import AlgorithmConfig
from .cython_func import roll_hisory
DEBUG = True

class ShellEnvWrapper(object):
    """
        A wrapper of env on ALG side to provide flexibility, e.g. achieving episode level resonance
    """
    def __init__(self, n_agent, n_thread, space, mcv, RL_functional, alg_config, scenario_config):
        self.n_agent = n_agent
        self.n_thread = n_thread
        self.space = space
        self.mcv = mcv
        self.RL_functional = RL_functional
        self.scenario_config = scenario_config
        if self.scenario_config.EntityOriented:
            self.core_dim = self.scenario_config.obs_vec_length
        else:
            self.core_dim = space['obs_space']['obs_shape']
        self.n_entity_placeholder = alg_config.n_entity_placeholder
        assert self.n_entity_placeholder >= 4

        # whether to use avail_act to block forbiden actions
        self.AvailActProvided = False
        if hasattr(self.scenario_config, 'AvailActProvided'):
            self.AvailActProvided = self.scenario_config.AvailActProvided 

        # whether to load previously saved checkpoint
        self.load_checkpoint = alg_config.load_checkpoint
        self.UseStepLevelResonance = alg_config.UseStepLevelResonance
        self.cold_start = True

    @staticmethod
    def get_binary_array(n, n_bits, dtype=np.float32):
        arr = np.zeros(n_bits, dtype=dtype)
        pointer = 0
        while True:
            arr[pointer] = int(n%2==1)
            n = n >> 1
            pointer += 1
            if n == 0: break
        return arr

    def interact_with_env(self, State_Recall):
        obs = State_Recall['Latest-Obs']
        P = State_Recall['ENV-PAUSE']
        RST = State_Recall['Env-Suffered-Reset']

        if RST.all():
            # just experienced full reset on all episode, this is the first step of all env threads
            if self.UseStepLevelResonance: 
                State_Recall['_FixMax_'] = None
            else:
                State_Recall['_FixMax_'] = ( np.random.rand(self.n_thread) < AlgorithmConfig.yita )
                
        act = np.zeros(shape=(self.n_thread, self.n_agent), dtype=np.int) - 1 # 初始化全部为 -1

        obs_feed = obs[~P]
        obs_feed_in = obs_feed.copy()

        I_State_Recall = {'obs':obs_feed_in, 
            'Test-Flag':State_Recall['Test-Flag'], 
            '_FixMax_':State_Recall['_FixMax_'][~P] if (not self.UseStepLevelResonance) else None, 
            'threads_active_flag':~P, 
            'Latest-Team-Info':State_Recall['Latest-Team-Info'][~P],
            }
        if self.AvailActProvided:
            avail_act = np.array([info['avail-act'] for info in np.array(State_Recall['Latest-Team-Info'][~P], dtype=object)])
            I_State_Recall.update({'avail_act':avail_act})

        act_active, internal_recall = self.RL_functional.interact_with_env_genuine(I_State_Recall)

        act[~P] = act_active
        actions_list = np.swapaxes(act, 0, 1) # swap thread(batch) axis and agent axis

        # return necessary handles to main platform
        if self.cold_start: self.cold_start = False

        
        State_Recall['_hook_'] = internal_recall['_hook_']
        assert State_Recall['_hook_'] is not None
        return actions_list, State_Recall 

