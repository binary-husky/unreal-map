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
from VISUALIZE.mcom import mcom

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
        self.parameter = [p for p_name, p in self.all_parameter]    # 535

        # set learning rate differently?
        if ppo_config.type_agent_diff_lr:
            others_parameters = [v for k,v in self.all_parameter if '_nets_flat' not in k]
            adam_lr_list = [
                {'params': list(n.parameters()), 'lr':self.lr*n.lr_div} for n in policy_and_critic._nets_flat_placeholder_
            ] + [{'params': list(policy_and_critic._critic_central.parameters()), 'lr':self.lr}]   # 33*3*5 + 40 = 535
            assert sum([len(list(d['params'])) for d in adam_lr_list]) == len(self.all_parameter)
            self.optimizer = optim.Adam(adam_lr_list, lr=self.lr)
        else:
            self.optimizer = optim.Adam(self.parameter, lr=self.lr)

        self.g_update_delayer = 0
        self.g_initial_value_loss = 0
        
        # 轮流训练式
        self.mcv = mcv
        self.ppo_update_cnt = 0
        self.batch_size_reminder = True
        self.trivial_dict = {}

        assert self.n_pieces_batch_division == 1
        self.gpu_share_unit = GpuShareUnit(cfg.device, gpu_party=cfg.gpu_party)

        self.mcv2 = mcom( path='%s/logger/ppo/'%cfg.logdir,
                          image_path='%s/detail_reward.jpg'%cfg.logdir,
                          rapid_flush=True,
                          draw_mode=cfg.draw_mode,
                          tag='[ppo.py]' )
        self.mcv2.rec_init(color='g')

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
                    print亮红('Insufficient gpu memory, using previous sample size !')
                else:
                    assert False
            torch.cuda.empty_cache()

    def log_reward_rich(self, traj_pool, mcv2):
        tags = {}
        for traj in traj_pool:
            traj.reward_sum = sum(traj.reward[:,0])
            gp_list = traj.gp_sel_summary[0] 
            if (gp_list==0).all():
                tag = 'frontend'
                if tag not in tags: tags[tag] = []
                tags[tag].append(traj.reward_sum)
            else:
                gp = max(gp_list)
                wr = self.policy_and_critic.ckpg_info[gp-1]['win_rate']
                tp = np.argmax(gp_list)
                tag = 'tp:%d wr:%.2f'%(tp, wr)
                if tag not in tags: tags[tag] = []
                tags[tag].append(traj.reward_sum)
        tags = dict(sorted(tags.items()))
        for k in tags:
            mcv2.rec(np.array(tags[k]).mean(), k)
        mcv2.rec_show()

    def train_on_traj_(self, traj_pool, task):
        self.log_reward_rich(traj_pool, self.mcv2)
        ppo_valid_percent_list = []
        sampler = TrajPoolSampler(n_div=1, traj_pool=traj_pool, flag=task, prevent_batchsize_oom=self.prevent_batchsize_oom, mcv=self.mcv)
        # before_training_hash = [__hashn__(t.parameters()) for t in (self.policy_and_critic._nets_flat_placeholder_)]
        for e in range(self.ppo_epoch):
            sample_iter = sampler.reset_and_get_iter()
            self.optimizer.zero_grad()
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
            nn.utils.clip_grad_norm_(self.parameter, self.max_grad_norm)
            self.optimizer.step()
            
            if ppo_valid_percent_list[-1] < 0.70: 
                print亮黄('policy change too much, epoch terminate early'); break
        pass # finish all epoch update

        print亮黄(np.array(ppo_valid_percent_list))
        self.log_trivial_finalize()
        
        net_updated = [any([p.grad is not None for p in t.parameters()]) for t in (self.policy_and_critic._nets_flat_placeholder_)]
        self.optimizer.zero_grad(set_to_none=True)
        self.ppo_update_cnt += 1
        for updated, net in zip(net_updated, self.policy_and_critic._nets_flat_placeholder_):
            if updated:
                net.update_cnt.data[0] = self.ppo_update_cnt
        self.policy_and_critic.on_update(self.ppo_update_cnt)
        
        torch.cuda.empty_cache()
        return self.ppo_update_cnt

    def freeze_body(self):
        assert False, "function forbidden"
        self.freeze_body = True
        self.parameter_pv = [p_name for p_name, p in self.all_parameter if not any(p_name.startswith(kw)  for kw in ('obs_encoder', 'attention_layer'))]
        self.parameter = [p for p_name, p in self.all_parameter if not any(p_name.startswith(kw)  for kw in ('obs_encoder', 'attention_layer'))]
        self.optimizer = optim.Adam(self.parameter, lr=self.lr)
        print('change train object')

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
        hete_pick = _2tensor(sample['hete_pick'])
        hete_type = _2tensor(sample['hete_type'])
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
                hete_type=hete_type,
                gp_sel_summary=gp_sel_summary)
        entropy_loss = entropy.mean()


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
        ratio  = torch.exp(E_clip)
        policy_loss = -(ratio*advantage).mean()

        # add all loses
        value_loss = 0.5 * F.mse_loss(real_value, newPi_value)


        AT_net_loss = policy_loss - entropy_loss*self.entropy_coef # + probs_loss*20
        CT_net_loss = value_loss * 1.0
        # AE_new_loss = ae_loss * 1.0

        loss_final =  AT_net_loss + CT_net_loss  # + AE_new_loss

        ppo_valid_percent = ((E_clip == E).int().sum()/batch_agent_size)

        nz_mask = real_value!=0
        value_loss_abs = (real_value[nz_mask] - newPi_value[nz_mask]).abs().mean()
        others = {
            'Value loss Abs':           value_loss_abs,
            'PPO valid percent':        ppo_valid_percent,
            'CT_net_loss':              CT_net_loss,
            'AT_net_loss':              AT_net_loss,
        }


        return loss_final, others


