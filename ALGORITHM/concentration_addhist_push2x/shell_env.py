import os, torch
import numpy as np
from numba import njit, jit
from UTILS.colorful import *
from UTILS.tensor_ops import my_view, __hash__
import pickle
from config import GlobalConfig
from .cython_func import roll_hisory
DEBUG = True

# @njit
def distance_matrix(A):
    assert A.shape[-1] == 2 # assert 2D situation
    n_subject = A.shape[-2] # is 2
    A = np.repeat(np.expand_dims(A,-2), n_subject, axis=-2) # =>(64, 100, 100, 2)
    At = np.swapaxes(A,-2,-3) # =>(64, 100, 100, 2)
    dis = At-A # =>(64, 100, 100, 2)
    dis = np.linalg.norm(dis, axis=-1)
    return dis

def stack_padding(l):
    import itertools
    return np.column_stack((itertools.zip_longest(*l, fillvalue=0)))

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
                                          alg_config, scenario_config):
        self.n_agent = n_agent
        self.n_thread = n_thread
        self.space = space
        self.mcv = mcv
        self.RL_functional = RL_functional
        self.n_basic_dim = scenario_config.obs_vec_length

        # whether to use avail_act to block forbiden actions
        self.block_invalid_action = False
        if hasattr(scenario_config, 'block_invalid_action'):
            self.block_invalid_action = scenario_config.block_invalid_action 

        # whether to load previously saved checkpoint
        self.load_checkpoint = alg_config.load_checkpoint
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
        act = np.zeros(shape=(self.n_thread, self.n_agent), dtype=np.int) - 1 # 初始化全部为 -1
        # read internal coop graph info
        obs = State_Recall['Latest-Obs']
        n_thread = obs.shape[0]

        # previous_obs = State_Recall['_Previous_Obs_'] if '_Previous_Obs_' in State_Recall else np.zeros_like(obs)
        his_pool_obs = State_Recall['_Histpool_Obs_'] if '_Histpool_Obs_' in State_Recall \
            else my_view(np.zeros_like(obs),[0, 0, -1, self.n_basic_dim])
            # else my_view(np.zeros_like(np.concatenate((obs,obs), -1)),[0, 0, -1, self.n_basic_dim])

        ENV_PAUSE = State_Recall['ENV-PAUSE']
        obs_feed = obs[~ENV_PAUSE]
        # prev_obs_feed = previous_obs[~ENV_PAUSE]
        his_pool_obs_feed = his_pool_obs[~ENV_PAUSE]
        # his_pool_obs_feed[ENV_PAUSE] = 0    # clear ended

        obs_feed_in, his_pool_next = self.solve_duplicate(obs_feed.copy(), his_pool_obs_feed.copy())
        his_pool_obs[~ENV_PAUSE] = his_pool_next
        his_pool_obs[ENV_PAUSE] = 0

        I_State_Recall = {'obs':obs_feed_in, 
            'Test-Flag':State_Recall['Test-Flag'], 
            'threads_active_flag':~ENV_PAUSE, 
            'Latest-Team-Info':State_Recall['Latest-Team-Info'][~ENV_PAUSE],
            }
        if self.block_invalid_action:
            avail_act = np.array([info['avail-act'] for info in np.array(State_Recall['Latest-Team-Info'][~ENV_PAUSE], dtype=object)])
            I_State_Recall.update({'avail_act':avail_act})

        act_active, internal_recall = self.RL_functional.interact_with_env_genuine(I_State_Recall)

        act[~ENV_PAUSE] = act_active
        actions_list = []
        act = np.expand_dims(act, -1)
        for i in range(self.n_agent): actions_list.append(act[:,i,:])
        actions_list = np.array(actions_list)

        # return necessary handles to main platform
        if self.cold_start: self.cold_start = False

        # <2> call a empty frame to gather reward
        # State_Recall['_Previous_Obs_'] = obs
        State_Recall['_Histpool_Obs_'] = his_pool_obs
        
        State_Recall['_hook_'] = internal_recall['_hook_']
        assert State_Recall['_hook_'] is not None
        return actions_list, State_Recall 

    def solve_duplicate(self, obs_feed_new, prev_his_pool):
        #  input might be (n_thread, n_agent, n_entity, basic_dim), or (n_thread, n_agent, n_entity*basic_dim)
        # both can be converted to (n_thread, n_agent, n_entity, basic_dim)
        obs_feed_new = my_view(obs_feed_new,[0, 0, -1, self.n_basic_dim])
        prev_obs_feed = my_view(prev_his_pool,[0, 0, -1, self.n_basic_dim])

        # turn history into more entities
        obs_feed = np.concatenate((obs_feed_new, prev_obs_feed), axis=-2)

        # turning all zero padding to NaN, which is excluded in normalization
        obs_feed[(obs_feed==0).all(-1)] = np.nan
        obs_feed_new[(obs_feed_new==0).all(-1)] = np.nan
        valid_mask = ~np.isnan(obs_feed_new).any(-1)    #

        next_his_pool = np.zeros_like(prev_obs_feed) # twice size  (64 threads, 50 agents, 12 subjects)

        # set self as not valid to avoid buffering self obs! valid_mask
        valid_mask[:,:,0] = False
        N_valid = valid_mask.sum(-1)
        next_his_pool = roll_hisory(obs_feed_new, prev_obs_feed, valid_mask, N_valid, next_his_pool)

        # a very important assumption: if an agent observe itself as NaN *When and Only When* it is dead
        alive_mask = ~np.isnan(obs_feed_new[:,:,0]).any(-1) 
        if (~alive_mask).any():
            obs_feed[~alive_mask] = np.nan
        return obs_feed, next_his_pool

    # @staticmethod
    # @jit(forceobj=True)
    # def roll_hisory(obs_feed_new, prev_obs_feed, valid_mask, N_valid, next_his_pool):
    #     for th in range(N_valid.shape[0]):
    #         for a in range(N_valid.shape[1]):
    #             n_v = N_valid[th,a]
    #             next_his_pool[th,a,:n_v] = obs_feed_new[th,a,valid_mask[th,a]]
    #             next_his_pool[th,a,n_v:] = prev_obs_feed[th,a,:(24-n_v)]
    #     return next_his_pool

    def get_mask_id(self, obs_feed):
        mask_and_id = np.zeros_like(obs_feed)[:,:,:, 0] # thread,agent,agent_obs
        binary = obs_feed[...,-8:]
        alive = obs_feed[..., 0]
        for i in range(8):
            mask_and_id += binary[..., i]* 2**i
        # print(mask_and_id)
        mask_and_id = np.where(alive==1, mask_and_id, np.nan)
        return mask_and_id




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
