import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from UTIL.colorful import *
import numpy as np
from UTIL.tensor_ops import _2tensor


class PPO():
    def __init__(self, policy_and_critic, mcv=None):
        from .reinforce_foundation import CoopAlgConfig
        self.policy_and_critic = policy_and_critic
        self.clip_param = CoopAlgConfig.clip_param
        self.ppo_epoch = CoopAlgConfig.ppo_epoch
        self.n_pieces_batch_division = CoopAlgConfig.n_pieces_batch_division
        self.value_loss_coef = CoopAlgConfig.value_loss_coef
        self.entropy_coef = CoopAlgConfig.entropy_coef
        self.max_grad_norm = CoopAlgConfig.max_grad_norm
        self.lr = CoopAlgConfig.lr
        self.g_optimizer = optim.Adam(policy_and_critic.parameters(), lr=self.lr)
        self.g_update_delayer = 0
        self.g_initial_value_loss = 0
        self.invalid_penalty = CoopAlgConfig.invalid_penalty
        # 轮流训练式
        self.train_switch = True
        self.mcv = mcv
        self.ppo_update_cnt = 0
        self.loss_bias =  CoopAlgConfig.balance

    def train_on_traj(self, traj_pool, task):

        # print(traj_pool) 从轨迹中采样
        g_value_loss_epoch = 0
        g_action_loss_epoch = 0
        g_dist_entropy_epoch = 0
        error_act_loss_epoch = 0
        self.train_switch = not self.train_switch
        num_updates = self.ppo_epoch * self.n_pieces_batch_division

        if task == 'train_R':
            flag='train_R'
            print('train_R')
        elif task == 'train_L':
            flag='train_L'
            print('train_L')

        for e in range(self.ppo_epoch):
            data_generator = self.轨迹采样(traj_pool, flag=flag)
            n_batch = next(data_generator)
            for small_batch in range(n_batch):
                sample = next(data_generator)
                self.g_optimizer.zero_grad()
                loss_final = 0
                policy_loss, entropy_loss, gx_value_loss, error_act_loss, loss_final_t = self.get_loss(flag, sample)
                g_value_loss_epoch += gx_value_loss.item() / num_updates
                g_action_loss_epoch += policy_loss.item() / num_updates
                g_dist_entropy_epoch += entropy_loss.item() / num_updates
                error_act_loss_epoch += error_act_loss.item() / num_updates
                if flag == 'train_R':
                    loss_final += loss_final_t * self.loss_bias
                if flag == 'train_L':
                    loss_final += loss_final_t * (1 - self.loss_bias)

                loss_final.backward()
                nn.utils.clip_grad_norm_(self.policy_and_critic.parameters(), self.max_grad_norm)
                self.g_optimizer.step()
            pass # finish small batch update
        pass # finish all epoch update
        self.ppo_update_cnt += 1
        print亮靛('value loss', g_value_loss_epoch, 'policy loss',g_action_loss_epoch, 'entropy loss',g_dist_entropy_epoch,'invalid-action loss', error_act_loss_epoch)
        return self.ppo_update_cnt


    def 轨迹采样(self, traj_pool, flag):
        container = {}

        if flag=='train_R':
            req_dict =        ['g_obs', 'g_actions', 'g_actionLogProbs_R', 'return_R', 'value_R',     'ctr_mask_R' ]
            req_dict_rename = ['g_obs', 'g_actions', 'g_actionLogProbs'  , 'return',   'state_value', 'ctr_mask']
        elif flag=='train_L':
            req_dict =        ['g_obs', 'g_actions', 'g_actionLogProbs_L', 'return_L', 'value_L',     'ctr_mask_L']
            req_dict_rename = ['g_obs', 'g_actions', 'g_actionLogProbs'  , 'return',   'state_value', 'ctr_mask']

        return_rename = "return"
        value_rename =  "state_value"
        advantage_rename = "advantage"
        # 将 g_obs 替换为 g_obs>xxxx
        for key_index, key in enumerate(req_dict):
            key_name =  req_dict[key_index]
            key_rename = req_dict_rename[key_index]
            if not hasattr(traj_pool[0], key_name):
                real_key_list = [real_key for real_key in traj_pool[0].__dict__ if (key_name+'>' in real_key)]
                assert len(real_key_list) > 0, ('检查提供的变量', key,key_index)
                for real_key in real_key_list:
                    mainkey, subkey = real_key.split('>')
                    req_dict.append(real_key)
                    req_dict_rename.append(key_rename+'>'+subkey)
        big_batch_size = -1  # 检查是不是长度统一
        # 加载轨迹进container数组
        for key_index, key in enumerate(req_dict):
            key_name =  req_dict[key_index]
            key_rename = req_dict_rename[key_index]
            if not hasattr(traj_pool[0], key_name): continue
            set_item = np.concatenate([getattr(traj, key_name) for traj in traj_pool], axis=0)
            if not (big_batch_size==set_item.shape[0] or (big_batch_size<0)):
                print('error')
            assert big_batch_size==set_item.shape[0] or (big_batch_size<0), (key,key_index)
            big_batch_size = set_item.shape[0]
            container[key_rename] = set_item    # 指针赋值

        container[advantage_rename] = container[return_rename] - container[value_rename]
        container[advantage_rename] = ( container[advantage_rename] - container[advantage_rename].mean() ) / (container[advantage_rename].std() + 1e-5)
        mini_batch_size = math.ceil(big_batch_size / self.n_pieces_batch_division)  # size of minibatch for each agent
        sampler = BatchSampler(SubsetRandomSampler(range(big_batch_size)), mini_batch_size, drop_last=False)
        yield len(sampler)
        for indices in sampler:
            selected = {}
            for key in container:
                selected[key] = container[key][indices]
            for key in [key for key in selected if '>' in key]:
                # 重新把子母键值组合成二重字典
                mainkey, subkey = key.split('>')
                if not mainkey in selected: selected[mainkey] = {}
                selected[mainkey][subkey] = selected[key]
                del selected[key]
            yield selected




    def get_loss(self, flag, sample):
        obs =       _2tensor(sample['g_obs'])
        # obs:          $batch.$n_agent.$core_dim                   used in [action eval],
        advantage = _2tensor(sample['advantage'])
        # advantage A(s_t):    $batch.$1.(advantage)                       used in [policy reinforce],
        action =    _2tensor(sample['g_actions'])
        # action:      $batch.$2.(two actions)                     not used yet
        oldPi_actionLogProb = _2tensor(sample['g_actionLogProbs'])
        # oldPi_actionLogProb:  $batch.$1.(the output from act)        used in [clipped version of value loss],
        real_value =   _2tensor(sample['return'])
        # real_value:       $batch.$1.(r_t0+r_t1+r_t2+...)
        ctr_mask = _2tensor(sample['ctr_mask'])


        if flag == 'train_R':
            newPi_value, newPi_actionLogProb, entropy_loss, probs_, _,_,_,_ = self.policy_and_critic.evaluate_actions(obs, action)
        elif flag == 'train_L':
            _,_,_,_, newPi_value, newPi_actionLogProb, entropy_loss, probs_ = self.policy_and_critic.evaluate_actions(obs, action)
        else:
            assert False


        error_act_loss = (ctr_mask*probs_).mean()

        ratio = torch.exp(newPi_actionLogProb - oldPi_actionLogProb)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = 0.5 * F.mse_loss(real_value, newPi_value)
        loss_final = policy_loss   +value_loss*self.value_loss_coef   -entropy_loss*self.entropy_coef   +error_act_loss * self.invalid_penalty
        return policy_loss, entropy_loss, value_loss, error_act_loss, loss_final


