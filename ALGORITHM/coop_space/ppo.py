import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from colorful import *
from z_config import GlobalConfig
import numpy as np
from my_utils import _2tensor


class PPO():
    def __init__(self, policy_and_critic, mcv=None):
        self.policy_and_critic = policy_and_critic
        self.clip_param = GlobalConfig.clip_param
        self.ppo_epoch = GlobalConfig.ppo_epoch
        self.n_pieces_batch_division = GlobalConfig.n_pieces_batch_division
        self.value_loss_coef = GlobalConfig.value_loss_coef
        self.entropy_coef = GlobalConfig.entropy_coef
        self.max_grad_norm = GlobalConfig.max_grad_norm
        self.lr = GlobalConfig.lr
        self.g_optimizer = optim.Adam(policy_and_critic.parameters(), lr=self.lr)
        self.g_update_delayer = 0
        self.g_initial_value_loss = 0
        # 轮流训练式
        self.train_switch = True
        self.mcv = mcv
        self.ppo_update_cnt = 0
        self.loss_bias =  GlobalConfig.balance
        self.batch_size_reminder = True

    def train_on_traj(self, traj_pool, task):

        # print(traj_pool) 从轨迹中采样
        g_value_loss_epoch = 0
        g_action_loss_epoch = 0
        g_dist_entropy_epoch = 0
        error_loss_epoch = 0
        probs_loss_epoch = 0
        self.train_switch = not self.train_switch
        num_updates = self.ppo_epoch * self.n_pieces_batch_division

        if task == 'train_R':
            flag='train_R'
            print('train_R')
        elif task == 'train_L':
            flag='train_L'
            print('train_L')
        ppo_valid_percent_list = []
        for e in range(self.ppo_epoch):
            data_generator = self.轨迹采样(traj_pool, flag=flag)
            n_batch = next(data_generator)
            for small_batch in range(n_batch):
                sample = next(data_generator)
                self.g_optimizer.zero_grad()
                loss_final = 0
                policy_loss, entropy_loss, gx_value_loss, loss_final_t, percent, newPi_actionLogProb, ppo_valid_percent, probs_loss = self.get_loss(flag, sample)
                ppo_valid_percent_list.append(ppo_valid_percent)
                g_value_loss_epoch += gx_value_loss.item() / num_updates
                g_action_loss_epoch += policy_loss.item() / num_updates
                g_dist_entropy_epoch += entropy_loss.item() / num_updates
                error_loss_epoch += percent / num_updates
                probs_loss_epoch += probs_loss.item() / num_updates
                if flag == 'train_R':
                    loss_final += loss_final_t * self.loss_bias
                if flag == 'train_L':
                    loss_final += loss_final_t * (1 - self.loss_bias)
                loss_final.backward()
                nn.utils.clip_grad_norm_(self.policy_and_critic.parameters(), self.max_grad_norm)
                # grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in self.policy_and_critic.parameters() if p.grad is not None]))
                # print('grad_norm', grad_norm)
                self.g_optimizer.step()

            pass # finish small batch update
        pass # finish all epoch update
        print亮黄(ppo_valid_percent_list)
        self.mcv.rec(g_value_loss_epoch, 'value loss')
        self.mcv.rec(g_action_loss_epoch, 'policy loss')
        self.mcv.rec(g_dist_entropy_epoch, 'entropy loss')
        self.mcv.rec_show()
        self.ppo_update_cnt += 1
        print亮靛('value loss', g_value_loss_epoch, 'policy loss',g_action_loss_epoch, 'entropy loss',g_dist_entropy_epoch, 'error percent', error_loss_epoch,'probs loss',probs_loss_epoch)
        return self.ppo_update_cnt

# np.concatenate([getattr(traj, 'return_L') for traj in traj_pool], axis=0)
# np.concatenate([getattr(traj, 'value_L') for traj in traj_pool], axis=0)
    def 轨迹采样(self, traj_pool, flag):
        container = {}

        if flag=='train_R':
            req_dict =        ['g_obs', 'g_actions', 'g_actionLogProbs_R', 'return_R', 'value_R',   ]
            req_dict_rename = ['g_obs', 'g_actions', 'g_actionLogProbs'  , 'return'  , 'state_value', ]
        elif flag=='train_L':
            req_dict =        ['g_obs', 'g_actions', 'g_actionLogProbs_L', 'return_L', 'value_L',    ]
            req_dict_rename = ['g_obs', 'g_actions', 'g_actionLogProbs'  , 'return'  , 'state_value',]

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


        if flag == 'train_R':
            newPi_value, newPi_actionLogProb, entropy_loss, probs, _,_,_,_, others = self.policy_and_critic.evaluate_actions(obs, action)
        elif flag == 'train_L':
            _,_,_,_, newPi_value, newPi_actionLogProb, entropy_loss, probs, others = self.policy_and_critic.evaluate_actions(obs, action)
        else:
            assert False

        batchsize = newPi_actionLogProb.shape[0]
        if self.batch_size_reminder:
            print亮红('the batch size in each ppo update is %d'%batchsize); self.batch_size_reminder = False
            
        n_actions = probs.shape[-1]
        assert n_actions <= 15  # 
        penalty_prob_line = 0.02
        if GlobalConfig.add_prob_loss:
            probs_loss = (penalty_prob_line - torch.clamp(probs, min=0, max=penalty_prob_line)).mean()
        else:
            probs_loss = 0
        # ratio = torch.exp(newPi_actionLogProb - oldPi_actionLogProb)
        # surr1 = ratio * advantage
        # surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage
        # policy_loss = -torch.min(surr1, surr2).mean()

        E = newPi_actionLogProb - oldPi_actionLogProb
        E_clip = torch.zeros_like(E)
        E_clip = torch.where(advantage > 0, torch.clamp(E, max=np.log(1.0+self.clip_param)), E_clip) #  ratio_final = torch.where(advantage > 0, torch.clamp(ratio, min=0, max=(1.0+self.clip_param)), ratio_final) 
        E_clip = torch.where(advantage < 0, torch.clamp(E, min=np.log(1.0-self.clip_param), max=np.log(5) ), E_clip)  # ratio_final = torch.where(advantage < 0, torch.clamp(ratio, min=(1.0-self.clip_param),  max=3), ratio_final)
        ratio = torch.exp(E_clip)
        # # print('unclipped ratio: %.2f%%'%(((E_clip == (newPi_actionLogProb - oldPi_actionLogProb)).int().sum()/batchsize).item()*100) )
        policy_loss = -(ratio*advantage).mean()

        value_loss = 0.5 * F.mse_loss(real_value, newPi_value)

        loss_final =  policy_loss   +value_loss*self.value_loss_coef   -entropy_loss*self.entropy_coef + probs_loss*20

        percent = 0
        ppo_valid_percent = ((E_clip == E).int().sum()/batchsize).item()
        return policy_loss, entropy_loss, value_loss, loss_final, percent, newPi_actionLogProb, ppo_valid_percent, probs_loss



