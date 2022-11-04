import os, torch
import numpy as np
try: 
    from numba import njit, jit
except: 
    from UTIL.tensor_ops import dummy_decorator as jit
    from UTIL.tensor_ops import dummy_decorator as njit
from UTIL.colorful import *
from UTIL.tensor_ops import copy_clone, my_view, add_onehot_id_at_last_dim, add_obs_container_subject
import pickle
from config import GlobalConfig
from .formation_execution import main_formation
# @njit

def stack_padding(l):
    import itertools
    return np.column_stack((itertools.zip_longest(*l, fillvalue=0)))
    
def distance_matrix(A):
    assert A.shape[-1] == 2 # assert 2D situation
    n_subject = A.shape[-2] # is 2
    A = np.repeat(np.expand_dims(A,-2), n_subject, axis=-2) # =>(64, 100, 100, 2)
    At = np.swapaxes(A,-2,-3) # =>(64, 100, 100, 2)
    dis = At-A # =>(64, 100, 100, 2)
    dis = np.linalg.norm(dis, axis=-1)
    return dis

def dir_to_rad_angle(delta_pos):
    result = np.empty(delta_pos.shape[:-1], dtype=complex)
    result.real = delta_pos[...,0]; result.imag = delta_pos[...,1]
    rad_angle = np.angle(result) 
    return rad_angle

def reg_angle_deg(deg):
    return (deg + 180)%360 -180

def reg_angle(rad):
    # it's OK to show "RuntimeWarning: invalid value encountered in remainder"
    return (rad + np.pi)%(2*np.pi) -np.pi

class ShellEnvWrapper(object):
    def __init__(self, n_agent, n_thread, space, mcv, RL_functional, 
                                          alg_config, ScenarioConfig):
        self.n_agent = n_agent
        self.n_thread = n_thread
        self.space = space
        self.mcv = mcv
        self.RL_functional = RL_functional
        self.n_basic_dim = ScenarioConfig.obs_vec_length
        self.n_entity = ScenarioConfig.num_entity
        self.agent_uid = ScenarioConfig.uid_dictionary['agent_uid']
        self.entity_uid = ScenarioConfig.uid_dictionary['entity_uid']
        self.dec = ScenarioConfig.dec_dictionary
        self.n_object = ScenarioConfig.num_object
        self.load_checkpoint = alg_config.load_checkpoint
        self.cold_start = True
        self._division_obsR_init = None
        self._division_obsL_init = None
        self.formation_ctl = main_formation()
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
        act = np.zeros(shape=(self.n_thread, self.n_agent), dtype=np.int) - 1 # 初始化全部为 -1
        # read internal coop graph info
        obs = State_Recall['Latest-Obs']
        n_thread = obs.shape[0]

        # 前8个entity是原始动作
        # action_c_padding = np.zeros(shape=(n_thread, self.n_agent, 8), dtype=np.int)

        ENV_PAUSE = State_Recall['ENV-PAUSE']
        I_State_Recall = {'obs':obs[~ENV_PAUSE], 
            'Test-Flag':State_Recall['Test-Flag'], 
            'threads_active_flag':~ENV_PAUSE, 
            'Latest-Team-Info':State_Recall['Latest-Team-Info'][~ENV_PAUSE]}
            
        # act_active, internal_recall = self.RL_functional.interact_with_env_genuine(I_State_Recall)

        act[~ENV_PAUSE] = self.formation_ctl.get_action(I_State_Recall)
        actions_list = []
        act = np.expand_dims(act, -1)
        for i in range(self.n_agent): actions_list.append(act[:,i,:])
        actions_list = np.array(actions_list)

        # return necessary handles to main platform
        if self.cold_start: self.cold_start = False

        # <2> call a empty frame to gather reward
        # State_Recall['_hook_'] = internal_recall['_hook_']
        # assert State_Recall['_hook_'] is not None
        return actions_list, State_Recall # state_recall dictionary will preserve states for next action making









    @staticmethod
    @jit(forceobj=True)
    def dir_to_action(vec, vel):
        def np_mat3d_normalize_each_line(mat):
            return mat / np.expand_dims(np.linalg.norm(mat, axis=2) + 1e-16, axis=-1)
        dis2target = np.linalg.norm(vec, axis=2)
        vec = np_mat3d_normalize_each_line(vec) #self.step

        e_u = np.array([0,1])
        e_d = np.array([0,-1])
        e_r = np.array([1,0])
        e_l = np.array([-1,0])

        vel_u = np_mat3d_normalize_each_line(vel + e_u * 0.1)
        vel_d = np_mat3d_normalize_each_line(vel + e_d * 0.1)
        vel_r = np_mat3d_normalize_each_line(vel + e_r * 0.1)
        vel_l = np_mat3d_normalize_each_line(vel + e_l * 0.1)

        proj_u = (vel_u * vec).sum(-1)
        proj_d = (vel_d * vec).sum(-1)
        proj_r = (vel_r * vec).sum(-1)
        proj_l = (vel_l * vec).sum(-1)

        _u = ((vec * e_u).sum(-1)>0).astype(np.int)
        _d = ((vec * e_d).sum(-1)>0).astype(np.int)
        _r = ((vec * e_r).sum(-1)>0).astype(np.int)
        _l = ((vec * e_l).sum(-1)>0).astype(np.int)

        proj_u = proj_u + _u*2
        proj_d = proj_d + _d*2
        proj_r = proj_r + _r*2
        proj_l = proj_l + _l*2

        dot_stack = np.stack([proj_u, proj_d, proj_r, proj_l])
        direct = np.argmax(dot_stack, 0)

        action = np.where(direct == 0, 3, 0)    # 3 up
        action += np.where(direct == 1, 4, 0)   # 4 down
        action += np.where(direct == 2, 1, 0)   # 1 right
        action += np.where(direct == 3, 2, 0)   # 2 left

        action = (dis2target>0.05).astype(np.int)*action
        # make sure that all nan vec become invalid act 0, 
        # be careful when a different numpy version is used
        assert (action[np.isnan(np.sum(dot_stack,0))] == 0).all()
        # action *= 0
        return np.expand_dims(action, axis=-1)
