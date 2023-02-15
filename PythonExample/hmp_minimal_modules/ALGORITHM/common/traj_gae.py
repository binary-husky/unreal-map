# cython: language_level=3
import numpy as np
from ALGORITHM.common.traj import TRAJ_BASE
import copy
from UTIL.colorful import *
from UTIL.tensor_ops import my_view, repeat_at, gather_righthand

class trajectory(TRAJ_BASE):
    dead_mask_check = True  # confirm mask ok
    def __init__(self, traj_limit, env_id, alg_cfg):
        super().__init__(traj_limit, env_id)
        self.agent_alive_reference = 'alive'
        self.alg_cfg = alg_cfg

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
        agent_alive = getattr(self, self.agent_alive_reference)
        assert len(agent_alive.shape) == 2, "shoud be 2D (time, agent)/dead_or_alive"
        if self.need_reward_bootstrap:
            assert False, ('it should not go here if everything goes as expected')
        # deprecated if nothing in it
        p_valid = agent_alive.any(axis=-1)
        p_invalid = ~p_valid
        is_fully_valid_traj = (p_valid[-1] == True)
        # assert p_valid[-1] == True, 如果有三只队伍，很有可能出现一只队伍全体阵亡，但游戏仍未结束的情况
        if p_invalid.all(): #invalid traj
            self.deprecated_flag = True
            return
        if not is_fully_valid_traj:
            # adjust reward position if not fully valid
            reward = TJ('reward')
            for i in reversed(range(self.time_pointer)):
                if p_invalid[i] and i != 0: # invalid, push reward forward
                    reward[i-1] += reward[i]; reward[i] = np.nan
            setattr(self, 'reward', reward)
        # clip NaN
        for key in self.key_dict: setattr(self, key, TJ(key)[p_valid])
        if not is_fully_valid_traj:
            # reset time pointer
            self.time_pointer = p_valid.sum()
        # all done
        return

    def reward_push_forward(self, dead_mask):
        # self.new_reward = self.reward.copy()
        if self.alg_cfg.gamma_in_reward_forwarding:
            gamma = self.alg_cfg.gamma_in_reward_forwarding_value 
            for i in reversed(range(self.time_pointer)):
                if i==0: continue
                self.reward[i-1] += np.where(dead_mask[i], self.reward[i]*gamma, 0)  # if dead_mask[i]==True, this frame is invalid, move reward forward, set self.reward[i] to 0
                self.reward[i]    = np.where(dead_mask[i], 0, self.reward[i])        # if dead_mask[i]==True, this frame is invalid, move reward forward, set self.reward[i] to 0

        else:
            for i in reversed(range(self.time_pointer)):
                if i==0: continue
                self.reward[i-1] += np.where(dead_mask[i], self.reward[i], 0)        # if dead_mask[i]==True, this frame is invalid, move reward forward, set self.reward[i] to 0
                self.reward[i]    = np.where(dead_mask[i], 0, self.reward[i])        # if dead_mask[i]==True, this frame is invalid, move reward forward, set self.reward[i] to 0
        return


    # new finalize
    def finalize(self):
        self.readonly_lock = True
        assert not self.deprecated_flag
        TJ = lambda key: getattr(self, key) 
        assert not np.isnan(TJ('reward')).any()
        # deadmask
        agent_alive = getattr(self, self.agent_alive_reference)
        dead_mask = ~agent_alive

        if trajectory.dead_mask_check:
            trajectory.dead_mask_check = False
            if not dead_mask.any(): 
                assert False, "Are you sure agents cannot die? If so, delete this check."

        self.reward_push_forward(dead_mask) # push terminal reward forward 38 42 54
        threat = np.zeros(shape=dead_mask.shape) - 1
        assert dead_mask.shape[0] == self.time_pointer
        for i in reversed(range(self.time_pointer)):
            # threat[:(i+1)] 不包含threat[(i+1)]
            if i+1 < self.time_pointer:
                threat[:(i+1)] += (~(dead_mask[i+1]&dead_mask[i])).astype(np.int)
            elif i+1 == self.time_pointer:
                threat[:] += (~dead_mask[i]).astype(np.int)

        SAFE_LIMIT = 8
        threat = np.clip(threat, -1, SAFE_LIMIT)
        setattr(self, 'threat', np.expand_dims(threat, -1))

        # ! Use GAE to calculate return
        if self.alg_cfg.use_policy_resonance:
            self.gae_finalize_return_pr(reward_key='reward', value_key='BAL_value_all_level', new_return_name='BAL_return_all_level')
        else:
            self.gae_finalize_return(reward_key='reward', value_key='value', new_return_name='return')
        return

    def gae_finalize_return(self, reward_key, value_key, new_return_name):
        # ------- gae parameters -------
        gamma = self.alg_cfg.gamma 
        tau = self.alg_cfg.tau
        # ------- -------------- -------
        rewards = getattr(self, reward_key)
        value = getattr(self, value_key)
        # ------- -------------- -------
        length = rewards.shape[0]
        assert rewards.shape[0]==value.shape[0]
        # if dimension not aligned
        if rewards.ndim == value.ndim-1: rewards = np.expand_dims(rewards, -1)
        # initalize two more tracks
        setattr(self, new_return_name, np.zeros_like(value))
        self.key_dict.append(new_return_name)
        # ------- -------------- -------
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

    def gae_finalize_return_pr(self, reward_key, value_key, new_return_name):
        # ------- gae parameters -------
        gamma = self.alg_cfg.gamma 
        tau = self.alg_cfg.tau
        # ------- -------------- -------
        BAL_value_all_level = copy.deepcopy(getattr(self, value_key))
        # reshape to (batch, agent*distribution_precision, 1)
        value = my_view(BAL_value_all_level, [0, -1, 1])
        # ------- ------- reshape reward ------- -------
        rewards_cp = copy.deepcopy(getattr(self, reward_key))
        # if dimension not aligned
        if rewards_cp.ndim == value.ndim-1: rewards_cp = np.expand_dims(rewards_cp, -1)
        assert rewards_cp.shape[-1] == 1
        n_agent = rewards_cp.shape[-2]
        assert BAL_value_all_level.shape[-2] == n_agent
        assert BAL_value_all_level.shape[-1] == self.alg_cfg.distribution_precision
        rewards_cp = repeat_at(rewards_cp.squeeze(-1), -1, self.alg_cfg.distribution_precision)
        rewards_cp = my_view(rewards_cp, [0, -1, 1])
        # ------- -------------- -------
        length = rewards_cp.shape[0]
        assert rewards_cp.shape[0]==value.shape[0]
        # ------- -------------- -------
        returns = np.zeros_like(value)
        boot_strap = 0 if not self.need_reward_bootstrap else self.boot_strap_value['bootstrap_'+value_key]
        for step in reversed(range(length)):
            if step==(length-1): # 最后一帧
                value_preds_delta = rewards_cp[step] + gamma * boot_strap      - value[step]
                gae = value_preds_delta
            else:
                value_preds_delta = rewards_cp[step] + gamma * value[step + 1] - value[step]
                gae = value_preds_delta + gamma * tau * gae
            returns[step] = gae + value[step]
        # ------- -------------- -------
        returns = my_view(returns, [0, n_agent, self.alg_cfg.distribution_precision])   # BAL_return_all_level
        setattr(self, new_return_name, returns)
        self.key_dict.append(new_return_name)

        
        def select_value_level(BAL_all_level, randl):
            n_agent = BAL_all_level.shape[1]
            tmp_index = np.expand_dims(repeat_at(randl, -1, n_agent), -1)

            return gather_righthand(src=BAL_all_level, index=tmp_index, check=False)

        self.value_selected = select_value_level(BAL_all_level=self.BAL_value_all_level, randl=self.randl)
        self.return_selected = select_value_level(BAL_all_level=self.BAL_return_all_level, randl=self.randl)


'''
    轨迹池管理
'''

class TrajManagerBase(object):
    def __init__(self, n_env, traj_limit, alg_cfg):
        self.alg_cfg = alg_cfg
        self.n_env = n_env
        self.traj_limit = traj_limit
        self.update_cnt = 0
        self.traj_pool = []
        self.registered_keys = []
        self.live_trajs = [trajectory(self.traj_limit, env_id=i, alg_cfg=self.alg_cfg) for i in range(self.n_env)]
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
        if isinstance(done, bool): done = [done for _ in range(self.n_env)]
        if isinstance(skip, bool): skip = [skip for _ in range(self.n_env)]
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
                self.live_trajs[env_index] = trajectory(self.traj_limit, env_id=env_index, alg_cfg=self.alg_cfg)
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
    def __init__(self, n_env, traj_limit, trainer_hook, alg_cfg):
        super().__init__(n_env, traj_limit, alg_cfg)
        self.trainer_hook = trainer_hook
        self.traj_limit = traj_limit

    # 函数入口
    def feed_traj(self, traj_frag, require_hook=False):
        if require_hook:  raise ModuleNotFoundError("not supported anymore")
        assert self._traj_lock_buf is None
        assert '_DONE_' in traj_frag
        assert '_SKIP_' in traj_frag
        self.batch_update(traj_frag=traj_frag)  # call parent's batch_update()
        return

    def train_and_clear_traj_pool(self):
        print('do update %d'%self.update_cnt)
        for traj_handle in self.traj_pool:
            traj_handle.cut_tail()
        self.traj_pool = list(filter(lambda traj: not traj.deprecated_flag, self.traj_pool))
        for traj_handle in self.traj_pool: traj_handle.finalize()
        self.trainer_hook(self.traj_pool, 'train')
        self.traj_pool = []
        self.update_cnt += 1
        return self.update_cnt

    def can_exec_training(self):
        num_traj_needed = self.alg_cfg.train_traj_needed
        if len(self.traj_pool) >= num_traj_needed:  
            return True
        else:  
            return False
 