# cython: language_level=3
import numpy as np
import copy
from UTIL.colorful import *
def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])

'''
    轨迹
'''
class trajectory():

    def __init__(self, traj_limit, env_id):
        from .reinforce_foundation import CoopAlgConfig
        self.h_reward_on_R = CoopAlgConfig.h_reward_on_R
        self.gamma = CoopAlgConfig.gamma
        self.tau = CoopAlgConfig.tau
        
        self.readonly_lock = False
        self.traj_limit = traj_limit
        self.time_pointer = 0
        self.n_frame_clip = -1
        self.key_dict = []
        self.env_id = env_id
        self.done_cut_tail = False
        self.lock = False

    def remember(self, key, content):
        assert not self.readonly_lock
        if not (key in self.key_dict) and (content is not None):
            assert isinstance(content, np.ndarray) or isinstance(content, float), (key, content.__class__)
            assert self.time_pointer == 0, key
            tensor_size = ((self.traj_limit,) + tuple(content.shape))
            set_item = np.zeros(shape=tensor_size, dtype=content.dtype)
            set_item[:] = np.nan  if np.issubdtype(content.dtype, np.floating) else 0
            setattr(self, key, set_item)
            self.key_dict.append(key)
            getattr(self, key)[self.time_pointer] = content
        elif (key in self.key_dict) and (content is not None):
            getattr(self, key)[self.time_pointer] = content
        else:
            pass # do nothing

    def time_shift(self):
        assert self.time_pointer < self.traj_limit
        self.time_pointer += 1
        
    def get_hyper_reward(self):
        self.readonly_lock = True
        n_frame = self.time_pointer
        if not self.done_cut_tail:
            self.done_cut_tail = True
            # clip tail
            for key in self.key_dict:
                set_item = getattr(self, key)[:n_frame]
                setattr(self, key, set_item)
            # 根据这个轨迹上的NaN，删除所有无效时间点
            # before clip NaN, push reward forward
            reference_track = getattr(self, 'value_R')  
            reward = getattr(self, 'reward')
            p_invalid = np.isnan(reference_track).squeeze()
            p_valid = ~p_invalid
            assert ~p_invalid[0]
            for i in reversed(range(n_frame)):
                if p_invalid[i] and i != 0 : # invalid, push reward forward
                    reward[i-1] += reward[i]
                    reward[i] = np.nan
            # clip NaN
            for key in self.key_dict:  
                set_item = getattr(self, key)
                setattr(self, key, set_item[p_valid])

        reward_key = 'reward'
        reward = getattr(self, reward_key)
        assert not np.isnan(reward).any()
        return np.sum(reward) # sum of reward! not average!!

    # new finalize
    def finalize(self, hyper_reward=None):
        if hyper_reward is not None: 
            assert self.finalize
        self.readonly_lock = True
        n_frame = self.time_pointer

        assert self.done_cut_tail

        if hyper_reward is not None: 
            self.copy_track(origin_key='reward', new_key='h_reward')
            h_rewards = getattr(self, 'h_reward')
            assert not np.isnan(h_rewards[-1])
            # h_rewards[-1] = hyper_reward*(1-self.h_reward_percent) + h_rewards[-1]*self.h_reward_percent # reward fusion

        if self.h_reward_on_R:
            if hyper_reward is not None:  
                self.gae_finalize_return(reward_key='h_reward', value_key='value_R', new_return_name='return_R')
            self.gae_finalize_return(reward_key='reward', value_key='value_L', new_return_name='return_L')
        else:
            if hyper_reward is not None:  
                self.gae_finalize_return(reward_key='h_reward', value_key='value_L', new_return_name='return_L')
            self.gae_finalize_return(reward_key='reward', value_key='value_R', new_return_name='return_R')
        pass



    def clip_reward_track(self, reward_key, n_frame_clip):
        reward = getattr(self, reward_key)
        reward_tail = reward[n_frame_clip:].sum()
        reward[n_frame_clip-1] += reward_tail
        set_item = reward[:n_frame_clip]
        setattr(self, reward_key, set_item)
        #return getattr(self, reward_key)

    def copy_track(self, origin_key, new_key):
        if hasattr(self, origin_key):
            origin_handle = getattr(self, origin_key)
            setattr(self, new_key, origin_handle.copy())
            new_handle = getattr(self, new_key)
            self.key_dict.append(new_key)
            #return origin_handle, new_handle
        else:
            real_key_list = [real_key for real_key in self.__dict__ if (origin_key+'>' in real_key)]
            assert len(real_key_list)>0
            for real_key in real_key_list:
                mainkey, subkey = real_key.split('>')
                self.copy_track(real_key, (new_key+'>'+subkey))
            #return

    def gae_finalize_return(self, reward_key, value_key, new_return_name):

        gamma = self.gamma # ------- gae parameters -------
        tau = self.tau
        # ------- -------------- -------
        rewards = getattr(self, reward_key)
        value = getattr(self, value_key)
        length = rewards.shape[0]
        assert rewards.shape[0]==value.shape[0]
        gae = 0
        # initalize two more tracks
        setattr(self, new_return_name, np.zeros_like(value))
        self.key_dict.append(new_return_name)

        returns = getattr(self, new_return_name)
        for step in reversed(range(length)):
            if step==(length-1): # 最后一帧
                value_preds_delta = rewards[step] - value[step]
                gae = value_preds_delta
            else:
                value_preds_delta = rewards[step] + gamma * value[step + 1] -  value[step]
                gae = value_preds_delta + gamma * tau * gae
            returns[step] = gae + value[step]

def calculate_sample_entropy(samples):
    key = []
    freq = []
    n_sample = len(samples)
    for s in samples:
        if s not in key:
            key.append(s)
            freq.append(1)
        else:
            i = key.index(s)
            freq[i] += 1

    entropy = 0.0
    for j,f in enumerate(freq):
        freq[j] /= n_sample
        entropy += -freq[j] * np.log(freq[j])
    # print亮红(key)
    # print亮红(freq)
    return entropy

class TrajPoolManager(object):

    def __init__(self, n_pool):
        from .reinforce_foundation import CoopAlgConfig
        self.h_reward_on_R = CoopAlgConfig.h_reward_on_R
        self.n_pool =  n_pool
        self.traj_pool_history = []
        self.hyper_reward = []
        self.hyper_reward_discount = []
        self.traj_pool_index = []
        self.cnt = 0

    def absorb_finalize_pool(self, pool):
        self.traj_pool_history.append(pool)
        h_reward = np.array([traj_handle.get_hyper_reward() for traj_handle in pool]).mean()
        self.hyper_reward.append(h_reward)  # used to accumulate hyper reward
        self.traj_pool_index.append(self.cnt)

        if len(self.traj_pool_history) >= self.n_pool: # now there is n_pool+1 trajpool inside
            self.hyper_reward_discount.append(0)    # take place
        else:
            self.hyper_reward_discount.append(None) 

        if len(self.traj_pool_history) > self.n_pool:
            self.traj_pool_history.pop(0)
            self.hyper_reward.pop(0) # used to accumulate hyper reward
            self.traj_pool_index.pop(0)
            self.hyper_reward_discount.pop(0)

        if self.hyper_reward_discount[-1] is not None: 
            # self.hyper_reward: used to accumulate hyper reward
            assert len(self.hyper_reward) == self.n_pool
            self.hyper_reward_discount[-1] = sum(self.hyper_reward)/len(self.hyper_reward)

        # calculate the return of trajectories, return = GAE(r,v)
        for traj_handle in pool: 
            traj_handle.finalize(hyper_reward=self.hyper_reward_discount[-1])
        
        self.cnt += 1

        if self.h_reward_on_R:
            task = ['train_L']
            if self.hyper_reward_discount[-1] is not None:
                print(self.hyper_reward_discount)
                task.append('train_R')
                return task
            return task
        else:
            task = ['train_R']
            if self.hyper_reward_discount[-1] is not None:
                print(self.hyper_reward_discount)
                task.append('train_L')
                return task
            return task

'''
    轨迹池管理
'''
class BatchTrajManager():

    def __init__(self, n_env, traj_limit, trainer_hook):
        from .reinforce_foundation import CoopAlgConfig
        self.trainer_hook = trainer_hook
        self.n_env = n_env
        self.traj_limit = traj_limit
        self.train_traj_needed = CoopAlgConfig.train_traj_needed
        self.upper_training_epoch = CoopAlgConfig.upper_training_epoch
        self.live_trajs = [trajectory(self.traj_limit, env_id=i) for i in range(n_env)]
        self.live_traj_frame = [0 for _ in range(self.n_env)]
        self.traj_pool = []
        self.registered_keys = []
        self._traj_lock_buf = None
        self.pool_manager = TrajPoolManager(n_pool=self.upper_training_epoch)
        self.patience = 1e3
        self.update_cnt = 0

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
        assert self._traj_lock_buf is None
        # an unlock hook must be exected before new trajectory feed in
        if require_hook: # the traj_frag is not intact, lock up traj_frag, wait for more
            assert 'done' not in traj_frag
            assert 'reward' not in traj_frag
            self._traj_lock_buf = traj_frag
            return self._unlock_hook
        else:
            assert 'done' in traj_frag
            assert 'skip' in traj_frag
            self.__batch_update(traj_frag=traj_frag)


    def _unlock_hook(self, traj_frag):
        assert self._traj_lock_buf is not None
        traj_frag.update(self._traj_lock_buf)
        self._traj_lock_buf = None
        assert 'done' in traj_frag
        assert 'skip' in traj_frag
        traj_frag.pop('info')
        traj_frag.pop('Latest-Obs')
        traj_frag.pop('Terminal-Obs-Echo')
        self.__batch_update(traj_frag=traj_frag)

    def ___check_integraty(self, traj_frag):
        # can not alway waste time checking this
        if self.patience < 0: return
        self.patience -= 1

        for key in traj_frag:
            if key not in self.registered_keys:
                self.registered_keys.append(key)

        for key in self.registered_keys:
            assert key in traj_frag, ('this key sometimes disappears from the traj_frag:', key)

    def __batch_update(self, traj_frag):
        self.___check_integraty(traj_frag)
        done = traj_frag['done']; traj_frag.pop('done')
        skip = traj_frag['skip']; traj_frag.pop('skip')

        assert len(traj_frag['reward'])==len(skip)
        traj_frag['reward'] = traj_frag['reward'][~skip]

        # print('done', done)
        # print('skip', skip)
        # print('reward', traj_frag['reward'])
        # single bool to list bool
        if isinstance(done, bool): done = [done for i in range(self.n_env)]
        if isinstance(skip, bool): skip = [skip for i in range(self.n_env)]

        # feed
        cnt = 0
        for env_i, env_done, skip_this in zip(range(self.n_env), done, skip):
            if skip_this: continue
            # otherwise
            frag_index = cnt; cnt += 1
            env_index = env_i

            traj_handle = self.live_trajs[env_index]
            assert not traj_handle.lock
            for key in traj_frag:
                if traj_frag[key] is None:
                    traj_handle.remember(key, None)
                elif isinstance(traj_frag[key], dict):  # 如果是二重字典，特殊处理
                    for sub_key in traj_frag[key]:
                        content = traj_frag[key][sub_key][frag_index]
                        traj_handle.remember( "".join((key , ">" , sub_key)), content )
                else:
                    content = traj_frag[key][frag_index]
                    traj_handle.remember(key, content)

            self.live_traj_frame[env_index] += 1
            traj_handle.time_shift()
            if env_done:
                traj_handle.lock = True
                self.traj_pool.append(traj_handle)
                self.live_trajs[env_index] = trajectory(self.traj_limit, env_id=env_index)
                self.live_traj_frame[env_index] = 0

    def get_traj_frame(self):
        return self.live_traj_frame

    def train_and_clear_traj_pool(self):
        print亮紫('do update %d'%self.update_cnt)
        current_task_l = self.pool_manager.absorb_finalize_pool(pool=self.traj_pool)
        for current_task in current_task_l:
            ppo_update_cnt = self.trainer_hook(self.traj_pool, current_task)
            
        self.traj_pool = []
        self.update_cnt += 1
        # assert ppo_update_cnt == self.update_cnt
        return self.update_cnt

    def can_exec_training(self):
        if len(self.traj_pool) >= self.train_traj_needed:
            return True
        else:
            return False