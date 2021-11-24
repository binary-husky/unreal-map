# cython: language_level=3
from config import GlobalConfig
import numpy as np
from numpy.core.numeric import indices
from .foundation import AlgorithmConfig
from ..commom.traj import TRAJ_BASE
import copy
from UTILS.colorful import *
from UTILS.tensor_ops import __hash__, my_view, np_one_hot, np_repeat_at, np_softmax, scatter_with_nan
def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class trajectory(TRAJ_BASE):

    def __init__(self, traj_limit, env_id):
        super().__init__(traj_limit, env_id)
        self.reference_track_name = 'value'

    def early_finalize(self):
        assert not self.readonly_lock   # unfinished traj
        self.need_reward_bootstrap = True

    def set_terminal_obs(self, tobs):
        self.tobs = copy.deepcopy(tobs)

    def cut_tail(self):
        # 删去多余的预留空间
        super().cut_tail()
        TJ = lambda key: getattr(self, key)
        # 进一步地， 根据这个轨迹上的NaN，删除所有无效时间点
        reference_track = getattr(self, self.reference_track_name)
        if self.need_reward_bootstrap:
            assert False, ('it should not go here if everything goes as expected')
            # print('need_reward_bootstrap') 找到最后一个不是nan的位置
            T = np.where(~np.isnan(reference_track.squeeze()))[0][-1]
            self.boot_strap_value = {
                'bootstrap_value':TJ('value').squeeze()[T].copy(), 
            }
            assert not hasattr(self,'tobs')
            self.set_terminal_obs(TJ('g_obs')[T].copy())
            reference_track[T] = np.nan
        # deprecated if nothing in it
        p_invalid = np.isnan(my_view(reference_track, [0, -1])).any(axis=-1)
        p_valid = ~p_invalid
        if p_invalid.all(): #invalid traj
            self.deprecated_flag = True
            return
        # adjust reward position
        reward = TJ('reward')
        for i in reversed(range(self.time_pointer)):
            if p_invalid[i] and i != 0: # invalid, push reward forward
                reward[i-1] += reward[i]; reward[i] = np.nan
        setattr(self, 'reward', reward)
        # clip NaN
        for key in self.key_dict: setattr(self, key, TJ(key)[p_valid])
        # all done
        return

    def reward_push_forward(self, dead_mask):
        for i in reversed(range(self.time_pointer)):
            if i==0: continue
            self.reward[i-1] += self.reward[i]* dead_mask[i].astype(np.int)
            self.reward[i] = self.reward[i]* (~dead_mask[i]).astype(np.int)

    # new finalize
    def finalize(self):
        self.readonly_lock = True
        assert not self.deprecated_flag
        TJ = lambda key: getattr(self, key) 
        assert not np.isnan(TJ('reward')).any()
        # deadmask
        tmp = np.isnan(my_view(self.obs, [0,0,-1]))
        dead_mask = tmp.all(-1)
        if (True): # check if the mask is correct
            dead_mask_self = np.isnan(my_view(self.obs, [0,0,-1])[:,:,0])
            assert (dead_mask==dead_mask_self).all()
        # dead_mask2 = tmp.any(-1)
        # assert (dead_mask==dead_mask2).all()
        self.reward_push_forward(dead_mask) # push terminal reward forward 38 42 54
        threat = np.zeros(shape=dead_mask.shape) - 1
        assert dead_mask.shape[0] == self.time_pointer
        for i in reversed(range(self.time_pointer)):
            # threat[:(i+1)] 不包含threat[(i+1)]
            if i+1 < self.time_pointer:
                threat[:(i+1)] += (~(dead_mask[i+1]&dead_mask[i])).astype(np.int)
            elif i+1 == self.time_pointer:
                threat[:] += (~dead_mask[i]).astype(np.int)
        
        SAFE_LIMIT = 11
        threat = np.clip(threat, -1, SAFE_LIMIT)
        setattr(self, 'threat', np.expand_dims(threat, -1))

        # ! Use GAE to calculate return
        self.gae_finalize_return(reward_key='reward', value_key='value', new_return_name='return')
        return

    def gae_finalize_return(self, reward_key, value_key, new_return_name):
        # ------- gae parameters -------
        gamma = AlgorithmConfig.gamma 
        tau = AlgorithmConfig.tau
        # ------- -------------- -------
        rewards = getattr(self, reward_key)
        value = getattr(self, value_key)
        length = rewards.shape[0]
        assert rewards.shape[0]==value.shape[0]
        # if dimension not aligned
        if rewards.ndim == value.ndim-1: rewards = np.expand_dims(rewards, -1)
        # initalize two more tracks
        setattr(self, new_return_name, np.zeros_like(value))
        self.key_dict.append(new_return_name)

        returns = getattr(self, new_return_name)
        boot_strap = 0 if not self.need_reward_bootstrap else self.boot_strap_value['bootstrap_'+value_key]

        for step in reversed(range(length)):
            if step==(length-1): # 最后一帧
                value_preds_delta = rewards[step] + gamma * boot_strap      - value[step]
                gae = value_preds_delta
            else:
                value_preds_delta = rewards[step] + gamma * value[step + 1] - value[step]
                gae = value_preds_delta + gamma * tau * gae
            returns[step] = gae + value[step]














class TrajPoolManager(object):
    def __init__(self, n_pool):
        self.n_pool =  n_pool
        self.cnt = 0

    def absorb_finalize_pool(self, pool):
        for traj_handle in pool:
            traj_handle.cut_tail()
        pool = list(filter(lambda traj: not traj.deprecated_flag, pool))
        for traj_handle in pool: traj_handle.finalize()
        self.cnt += 1
        task = ['train']
        return task, pool













'''
    轨迹池管理
'''

class TrajManagerBase(object):
    def __init__(self, n_env, traj_limit):
        self.n_env = n_env
        self.traj_limit = traj_limit
        self.update_cnt = 0
        self.traj_pool = []
        self.registered_keys = []
        self.live_trajs = [trajectory(self.traj_limit, env_id=i) for i in range(self.n_env)]
        self.live_traj_frame = [0 for _ in range(self.n_env)]
        self._traj_lock_buf = None
        self.patience = 1000
        pass
    
    def __check_integraty(self, traj_frag):
        if self.patience < 0: 
            return # stop wasting time checking this
        self.patience -= 1
        for key in traj_frag:
            if key not in self.registered_keys and (not key.startswith('_')):
                self.registered_keys.append(key)
        for key in self.registered_keys:
            assert key in traj_frag, ('this key sometimes disappears from the traj_frag:', key)

    def batch_update(self, traj_frag):
        self.__check_integraty(traj_frag)
        done = traj_frag['_DONE_']; traj_frag.pop('_DONE_') # done flag
        skip = traj_frag['_SKIP_']; traj_frag.pop('_SKIP_') # skip/frozen flag
        tobs = traj_frag['_TOBS_']; traj_frag.pop('_TOBS_') # terminal obs
        # single bool to list bool
        if isinstance(done, bool): done = [done for i in range(self.n_env)]
        if isinstance(skip, bool): skip = [skip for i in range(self.n_env)]
        n_active = sum(~skip)
        # feed
        cnt = 0
        for env_i in range(self.n_env):
            if skip[env_i]: continue
            # otherwise
            frag_index = cnt; cnt += 1
            env_index = env_i
            traj_handle = self.live_trajs[env_index]
            for key in traj_frag:
                self.traj_remember(traj_handle, key=key, content=traj_frag[key],frag_index=frag_index, n_active=n_active)
            self.live_traj_frame[env_index] += 1
            traj_handle.time_shift()
            if done[env_i]:
                assert tobs[env_i] is not None # get the final obs
                traj_handle.set_terminal_obs(tobs[env_i])
                self.traj_pool.append(traj_handle)
                self.live_trajs[env_index] = trajectory(self.traj_limit, env_id=env_index)
                self.live_traj_frame[env_index] = 0

    def traj_remember(self, traj, key, content, frag_index, n_active):
        if content is None: traj.remember(key, None)
        elif isinstance(content, dict):
            for sub_key in content: 
                self.traj_remember(traj, "".join((key , ">" , sub_key)), content=content[sub_key], frag_index=frag_index, n_active=n_active)
        else:
            assert n_active == len(content), ('length error')
            traj.remember(key, content[frag_index]) # *


class BatchTrajManager(TrajManagerBase):
    def __init__(self, n_env, traj_limit, trainer_hook):
        super().__init__(n_env, traj_limit)
        self.trainer_hook = trainer_hook
        self.traj_limit = traj_limit
        self.train_traj_needed = AlgorithmConfig.train_traj_needed
        self.upper_training_epoch = AlgorithmConfig.upper_training_epoch
        self.pool_manager = TrajPoolManager(n_pool=self.upper_training_epoch)

    def update(self, traj_frag, index):
        assert traj_frag is not None
        for j, env_i in enumerate(index):
            traj_handle = self.live_trajs[env_i]
            for key in traj_frag:
                if traj_frag[key] is None:
                    assert False, key
                if isinstance(traj_frag[key], dict):  # 如果是二重字典，特殊处理
                    for sub_key in traj_frag[key]:
                        content = traj_frag[key][sub_key][j]
                        traj_handle.remember(key + ">" + sub_key, content)
                else:
                    content = traj_frag[key][j]
                    traj_handle.remember(key, content)
            self.live_traj_frame[env_i] += 1
            traj_handle.time_shift()
        return

    # 函数入口
    def feed_traj(self, traj_frag, require_hook=False):
        # an unlock hook must be executed before new trajectory feed in
        assert self._traj_lock_buf is None
        if require_hook: 
            # the traj_frag is not intact, lock up traj_frag, wait for more
            assert '_SKIP_' in traj_frag
            assert '_DONE_' not in traj_frag
            assert 'reward' not in traj_frag
            self._traj_lock_buf = traj_frag
            return self.unlock_fn
        else:
            assert '_DONE_' in traj_frag
            assert '_SKIP_' in traj_frag
            self.batch_update(traj_frag=traj_frag)
            return

        
    def train_and_clear_traj_pool(self):
        print('do update %d'%self.update_cnt)

        current_task_l, self.traj_pool = self.pool_manager.absorb_finalize_pool(pool=self.traj_pool)
        for current_task in current_task_l:
            ppo_update_cnt = self.trainer_hook(self.traj_pool, current_task)

        self.traj_pool = []
        self.update_cnt += 1
        # assert ppo_update_cnt == self.update_cnt
        return self.update_cnt

    def can_exec_training(self):
        if len(self.traj_pool) >= self.train_traj_needed:  return True
        else:  return False
 
    def unlock_fn(self, traj_frag):
        assert self._traj_lock_buf is not None
        traj_frag.update(self._traj_lock_buf)
        self._traj_lock_buf = None
        assert '_DONE_' in traj_frag
        assert '_SKIP_' in traj_frag
        self.batch_update(traj_frag=traj_frag)


