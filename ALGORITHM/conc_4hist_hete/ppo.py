import torch, math, traceback
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from random import randint, sample
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from UTIL.colorful import *
from UTIL.tensor_ops import _2tensor, __hash__, __hashn__
from config import GlobalConfig as cfg
from UTIL.gpu_share import GpuShareUnit
from .ppo_sampler import TrajPoolSampler

class PPO():
    def __init__(self, policy_and_critic, ppo_config, mcv=None):
        self.policy_and_critic = policy_and_critic
        self.clip_param = ppo_config.clip_param
        self.ppo_epoch = ppo_config.ppo_epoch
        self.use_avail_act = ppo_config.ppo_epoch
        self.n_pieces_batch_division = ppo_config.n_pieces_batch_division
        self.value_loss_coef = ppo_config.value_loss_coef
        self.entropy_coef = ppo_config.entropy_coef
        self.max_grad_norm = ppo_config.max_grad_norm
        self.add_prob_loss = ppo_config.add_prob_loss
        self.prevent_batchsize_oom = ppo_config.prevent_batchsize_oom
        # self.freeze_body = ppo_config.freeze_body
        self.lr = ppo_config.lr
        self.all_parameter = list(policy_and_critic.named_parameters())
        self.at_parameter = [(p_name, p) for p_name, p in self.all_parameter if 'AT_' in p_name]
        self.ct_parameter = [(p_name, p) for p_name, p in self.all_parameter if 'CT_' in p_name]
        # self.ae_parameter = [(p_name, p) for p_name, p in self.all_parameter if 'AE_' in p_name]
        # 检查剩下是是否全都是不需要训练的参数
        remove_exists = lambda LA,LB: list(set(LA).difference(set(LB)))
        res = self.all_parameter
        res = remove_exists(res, self.at_parameter)
        res = remove_exists(res, self.ct_parameter)
        # res = remove_exists(res, self.ae_parameter)
        for p_name, p in res:   
            assert not p.requires_grad, ('a parameter must belong to either CriTic or AcTor, unclassified parameter:',p_name)

        self.cross_parameter = [(p_name, p) for p_name, p in self.all_parameter if ('CT_' in p_name) and ('AT_' in p_name)]
        assert len(self.cross_parameter)==0,('a parameter must belong to either CriTic or AcTor, not both')


        # if not self.freeze_body:
        self.at_parameter = [p for p_name, p in self.all_parameter if 'AT_' in p_name]
        self.at_optimizer = optim.Adam(self.at_parameter, lr=self.lr)

        self.ct_parameter = [p for p_name, p in self.all_parameter if 'CT_' in p_name]
        self.ct_optimizer = optim.Adam(self.ct_parameter, lr=self.lr*10.0) #(self.lr)
        # else:
        #     self.at_parameter = [p for p_name, p in self.all_parameter if 'AT_div_tree' in p_name]
        #     self.at_optimizer = optim.Adam(self.at_parameter, lr=self.lr)
        #     self.ct_parameter = [p for p_name, p in self.all_parameter if 'CT_' in p_name]
        #     self.ct_optimizer = optim.Adam(self.ct_parameter, lr=self.lr*10.0) #(self.lr)

        self.g_update_delayer = 0
        self.g_initial_value_loss = 0
        # 轮流训练式
        self.mcv = mcv
        self.ppo_update_cnt = 0
        self.batch_size_reminder = True
        self.trivial_dict = {}

        assert self.n_pieces_batch_division == 1

        self.gpu_share_unit = GpuShareUnit(cfg.device, gpu_party=cfg.gpu_party)

    def freeze_body(self):
        self.freeze_body = True
        self.at_parameter = [p for p_name, p in self.all_parameter if 'AT_policy_head' in p_name]
        self.at_optimizer = optim.Adam(self.at_parameter, lr=self.lr)
        self.ct_parameter = [p for p_name, p in self.all_parameter if 'CT_' in p_name]
        self.ct_optimizer = optim.Adam(self.ct_parameter, lr=self.lr*10.0) #(self.lr)
        print('change train object')

    def train_on_traj(self, traj_pool, task):
        while True:
            try:
                with self.gpu_share_unit:
                    self.train_on_traj_(traj_pool, task) 
                break # 运行到这说明显存充足
            except RuntimeError as err:
                print(traceback.format_exc())
                if self.prevent_batchsize_oom:
                    # in some cases, reversing MaxSampleNum a single time is not enough
                    if TrajPoolSampler.MaxSampleNum[-1] < 0: TrajPoolSampler.MaxSampleNum.pop(-1)
                    assert TrajPoolSampler.MaxSampleNum[-1] > 0
                    TrajPoolSampler.MaxSampleNum[-1] = -1
                    print亮红('显存不足！ 回溯上次的样本量')
                else:
                    assert False
            torch.cuda.empty_cache()

    def train_on_traj_(self, traj_pool, task):

        ppo_valid_percent_list = []
        sampler = TrajPoolSampler(n_div=1, traj_pool=traj_pool, flag=task, prevent_batchsize_oom=self.prevent_batchsize_oom)
        # before_training_hash = [__hashn__(t.parameters()) for t in (self.policy_and_critic._nets_flat_placeholder_)]
        for e in range(self.ppo_epoch):
            sample_iter = sampler.reset_and_get_iter()
            self.at_optimizer.zero_grad()
            self.ct_optimizer.zero_grad()
            # ! get traj fragment
            sample = next(sample_iter)
            # ! build graph, then update network
            loss_final, others = self.establish_pytorch_graph(task, sample, e)
            loss_final = loss_final*0.5
            if e==0: print('[PPO.py] Memory Allocated %.2f GB'%(torch.cuda.memory_allocated()/1073741824))
            loss_final.backward()
            # log
            ppo_valid_percent_list.append(others.pop('PPO valid percent').item())
            self.log_trivial(dictionary=others); others = None
            nn.utils.clip_grad_norm_(self.at_parameter, self.max_grad_norm)
            self.at_optimizer.step()
            self.ct_optimizer.step()
            
            if ppo_valid_percent_list[-1] < 0.70: 
                print亮黄('policy change too much, epoch terminate early'); break
        pass # finish all epoch update

        print亮黄(np.array(ppo_valid_percent_list))
        self.log_trivial_finalize()
        
        net_updated = [any([p.grad is not None for p in t.parameters()]) for t in (self.policy_and_critic._nets_flat_placeholder_)]

                
        self.at_optimizer.zero_grad(set_to_none=True)
        self.ct_optimizer.zero_grad(set_to_none=True)
        # after_training_hash = [__hashn__(t.parameters()) for t in (self.policy_and_critic._nets_flat_placeholder_)]
        # for h1,h2 in zip(before_training_hash, after_training_hash):
        #     if h1==h2:
        #         print亮绿(h1,'-->',h2)
        #     else:
        #         print亮蓝(h1,'-->',h2)
                
        self.ppo_update_cnt += 1
        for updated, net in zip(net_updated, self.policy_and_critic._nets_flat_placeholder_):
            if updated:
                net.update_cnt.data[0] = self.ppo_update_cnt
                
        self.policy_and_critic.on_update(self.ppo_update_cnt)
        
        return self.ppo_update_cnt

    def log_trivial(self, dictionary):
        for key in dictionary:
            if key not in self.trivial_dict: self.trivial_dict[key] = []
            item = dictionary[key].item() if hasattr(dictionary[key], 'item') else dictionary[key]
            self.trivial_dict[key].append(item)

    def log_trivial_finalize(self, print=True):
        for key in self.trivial_dict:
            self.trivial_dict[key] = np.array(self.trivial_dict[key])
        
        print_buf = ['[ppo.py] ']
        for key in self.trivial_dict:
            self.trivial_dict[key] = self.trivial_dict[key].mean()
            print_buf.append(' %s:%.3f, '%(key, self.trivial_dict[key]))
            if self.mcv is not None:  self.mcv.rec(self.trivial_dict[key], key)
        if print: print紫(''.join(print_buf))
        if self.mcv is not None:
            self.mcv.rec_show()
        self.trivial_dict = {}


    def establish_pytorch_graph(self, flag, sample, n):
        obs = _2tensor(sample['obs'])
        advantage = _2tensor(sample['advantage'])
        action = _2tensor(sample['action'])
        oldPi_actionLogProb = _2tensor(sample['actionLogProb'])
        real_value = _2tensor(sample['return'])
        real_threat = _2tensor(sample['threat'])
        hete_pick = _2tensor(sample['hete_pick'])
        gp_sel_summary = _2tensor(sample['gp_sel_summary'])
        avail_act = _2tensor(sample['avail_act']) if 'avail_act' in sample else None

        # batchsize = advantage.shape[0]#; print亮紫(batchsize)
        batch_agent_size = advantage.shape[0]*advantage.shape[1]

        assert flag == 'train'
        newPi_value, newPi_actionLogProb, entropy, probs, others = \
            self.policy_and_critic.evaluate_actions(
                obs=obs, 
                eval_actions=action, 
                test_mode=False, 
                avail_act=avail_act, 
                hete_pick=hete_pick,
                gp_sel_summary=gp_sel_summary)
        entropy_loss = entropy.mean()


        # [motivational] threat approximation
        SAFE_LIMIT = 11
        filter = (real_threat<SAFE_LIMIT) & (real_threat>=0)
        threat_loss = F.mse_loss(others['threat'][filter], real_threat[filter])


        # [motivational] state value
        inv_value_loss = 0.5 * F.mse_loss(real_value, others['value'])

        n_actions = probs.shape[-1]
        if self.add_prob_loss: assert n_actions <= 15  # 
        penalty_prob_line = (1/n_actions)*0.12
        probs_loss = (penalty_prob_line - torch.clamp(probs, min=0, max=penalty_prob_line)).mean()
        if not self.add_prob_loss:
            probs_loss = torch.zeros_like(probs_loss)

        # dual clip ppo core
        E = newPi_actionLogProb - oldPi_actionLogProb
        E_clip = torch.zeros_like(E)
        E_clip = torch.where(advantage > 0, torch.clamp(E, max=np.log(1.0+self.clip_param)), E_clip)
        E_clip = torch.where(advantage < 0, torch.clamp(E, min=np.log(1.0-self.clip_param), max=np.log(5) ), E_clip)
        ratio = torch.exp(E_clip)
        policy_loss = -(ratio*advantage).mean()

        # add all loses
        value_loss = 0.5 * F.mse_loss(real_value, newPi_value)


        AT_net_loss = policy_loss - entropy_loss*self.entropy_coef # + probs_loss*20
        CT_net_loss = value_loss * 1.0 + threat_loss * 0.1 + inv_value_loss * 0.1 # + friend_threat_loss*0.01
        # AE_new_loss = ae_loss * 1.0

        loss_final =  AT_net_loss + CT_net_loss  # + AE_new_loss

        ppo_valid_percent = ((E_clip == E).int().sum()/batch_agent_size)

        nz_mask = real_value!=0
        value_loss_abs = (real_value[nz_mask] - newPi_value[nz_mask]).abs().mean()
        others = {
            # 'Policy loss':              policy_loss,
            # 'Entropy loss':             entropy_loss,
            'Value loss Abs':           value_loss_abs,
            # 'friend_threat_loss':       friend_threat_loss,
            'PPO valid percent':        ppo_valid_percent,
            'threat loss':              threat_loss,
            # 'Auto encoder loss':        ae_loss,
            'CT_net_loss':              CT_net_loss,
            'AT_net_loss':              AT_net_loss,
        }


        return loss_final, others


