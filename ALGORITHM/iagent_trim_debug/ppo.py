import torch, math  # v
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from random import randint, sample
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from UTILS.colorful import *
from UTILS.tensor_ops import _2tensor, _2cpu2numpy, repeat_at
from UTILS.tensor_ops import my_view, scatter_with_nan, sample_balance
from config import GlobalConfig as cfg
from UTILS.gpu_share import gpu_share_unit

class TrajPoolSampler():
    def __init__(self, n_div, traj_pool, flag, ratio):
        self.n_pieces_batch_division = n_div
        self.num_batch = None
        self.container = {}
        self.warned = False
        self.ratio = ratio
        assert flag=='train'
        req_dict =        ['obs', 'action', 'actionLogProb', 'return', 'reward', 'threat', 'value']
        req_dict_rename = ['obs', 'action', 'actionLogProb', 'return', 'reward', 'threat', 'state_value']

        return_rename = "return"
        value_rename =  "state_value"
        advantage_rename = "advantage"
        # replace 'obs' to 'obs > xxxx'
        for key_index, key in enumerate(req_dict):
            key_name =  req_dict[key_index]
            key_rename = req_dict_rename[key_index]
            if not hasattr(traj_pool[0], key_name):
                real_key_list = [real_key for real_key in traj_pool[0].__dict__ if (key_name+'>' in real_key)]
                assert len(real_key_list) > 0, ('check variable provided!', key,key_index)
                for real_key in real_key_list:
                    mainkey, subkey = real_key.split('>')
                    req_dict.append(real_key)
                    req_dict_rename.append(key_rename+'>'+subkey)
        self.big_batch_size = -1  # vector should have same length, check it!
        
        # load traj into a 'container'
        for key_index, key in enumerate(req_dict):
            key_name =  req_dict[key_index]
            key_rename = req_dict_rename[key_index]
            if not hasattr(traj_pool[0], key_name): continue
            set_item = np.concatenate([getattr(traj, key_name) for traj in traj_pool], axis=0)
            if not (self.big_batch_size==set_item.shape[0] or (self.big_batch_size<0)):
                print('error')
            assert self.big_batch_size==set_item.shape[0] or (self.big_batch_size<0), (key,key_index)
            self.big_batch_size = set_item.shape[0]
            self.container[key_rename] = set_item    # 指针赋值

        # normalize advantage inside the batch
        self.container[advantage_rename] = self.container[return_rename] - self.container[value_rename]
        self.container[advantage_rename] = ( self.container[advantage_rename] - self.container[advantage_rename].mean() ) / (self.container[advantage_rename].std() + 1e-5)
        # size of minibatch for each agent
        self.mini_batch_size = math.floor(self.big_batch_size / self.n_pieces_batch_division)  

        # def filter_dead(container):
        #     obs = container['obs']
        #     dead_mask = ~np.isnan(my_view(obs, [0,0, -1])).all(-1)
        #     print('警告！ 反向调试！');dead_mask = ~dead_mask
        #     for key in container:
        #         container[key] = container[key][dead_mask]
        #     # for key in container
        #     return container
        
        # self.container = filter_dead(self.container)
        # self.big_batch_size = len(self.container[value_rename])
        # self.mini_batch_size = self.big_batch_size
        # self.sampler = BatchSampler(SubsetRandomSampler(range(self.big_batch_size)), self.mini_batch_size, drop_last=True)
        # if not self.warned: print('警告：设置了死信息过滤'); self.warned = True
        
        # set_max_sample = True
        # max_sample = 9200
        # if set_max_sample:
        #     self.sampler = BatchSampler(SubsetRandomSampler(range(self.big_batch_size)), min(max_sample, self.big_batch_size), drop_last=True)
        #     if not self.warned: print('警告：设置了最大样本数'); self.warned = True

    def __len__(self):
        return self.n_pieces_batch_division

    def reset_and_get_iter(self):
        self.sampler = BatchSampler(SubsetRandomSampler(range(self.big_batch_size)), self.mini_batch_size, drop_last=True)
        for indices in self.sampler:
            if self.ratio != 1:
                # print亮红('警告，仅抽取一部分样本，原样本数%d，现样本数%d'%(len(indices), int(len(indices)*self.ratio)))
                indices = sample(indices, int(len(indices)*self.ratio))
            selected = {}
            for key in self.container:
                selected[key] = self.container[key][indices]
            for key in [key for key in selected if '>' in key]:
                # 重新把子母键值组合成二重字典
                mainkey, subkey = key.split('>')
                if not mainkey in selected: selected[mainkey] = {}
                selected[mainkey][subkey] = selected[key]
                del selected[key]
            yield selected

def est_check(x, y):
    import random
    y = y.flatten()
    x = x.flatten()
    size = y.shape[0]
    index = list(range(size))
    random.shuffle(index)
    index = index[:500]
    xsub = y[index]
    new_index = ~torch.isnan(xsub)
    
    final = []
    t = xsub.max()
    for i in range(15):
        if new_index.any():
            t = xsub[torch.where(new_index)[0][0]]
            final.append(index[torch.where(new_index)[0][0]])
            new_index = new_index & (xsub!=t)
        else:
            break
    
    x_ = x[final]
    y_ = y[final]
    s = torch.argsort(y_)
    y_ = y_[s]
    x_ = x_[s]
    print(x_.detach().cpu().numpy())
    print(y_.detach().cpu().numpy())
    return

class PPO():
    def __init__(self, policy_and_critic, ppo_config, mcv=None):
        self.policy_and_critic = policy_and_critic
        self.clip_param = ppo_config.clip_param
        self.ppo_epoch = ppo_config.ppo_epoch
        self.n_pieces_batch_division = ppo_config.n_pieces_batch_division
        self.value_loss_coef = ppo_config.value_loss_coef
        self.entropy_coef = ppo_config.entropy_coef
        self.max_grad_norm = ppo_config.max_grad_norm
        self.lr = ppo_config.lr
        self.extral_train_loop = ppo_config.extral_train_loop
        self.turn_off_threat_est = ppo_config.turn_off_threat_est
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
            assert not p.requires_grad, ('unclassified parameter:',p_name)
        # 不再需要参数名
        self.at_parameter = [p for p_name, p in self.all_parameter if 'AT_' in p_name]
        self.at_optimizer = optim.Adam(self.at_parameter, lr=self.lr)

        self.ct_parameter = [p for p_name, p in self.all_parameter if 'CT_' in p_name]
        self.ct_optimizer = optim.Adam(self.ct_parameter, lr=self.lr*10.0) #(self.lr)
        # self.ae_parameter = [p for p_name, p in self.all_parameter if 'AE_' in p_name]
        # self.ae_optimizer = optim.Adam(self.ae_parameter, lr=self.lr*100.0) #(self.lr)
        self.g_update_delayer = 0
        self.g_initial_value_loss = 0
        # 轮流训练式
        self.mcv = mcv
        self.ppo_update_cnt = 0
        self.loss_bias =  ppo_config.balance
        self.batch_size_reminder = True
        self.trivial_dict = {}

    def train_on_traj(self, traj_pool, task):
        ratio = 1.0
        while True:
            try:
                with gpu_share_unit(cfg.device, gpu_party=cfg.gpu_party):
                    self.train_on_traj_(traj_pool, task, ratio=ratio) 
                    # 运行到这说明显存充足
                break
            except RuntimeError:
                print亮红('显存不足！ 切换小batch, ratio: %.1f'%ratio)
                if ratio>=0.6: ratio -= 0.1
                else: assert False, ('显存严重不足！?')
            torch.cuda.empty_cache()

                
    def train_on_traj_(self, traj_pool, task, ratio=1.0):

        num_updates = self.ppo_epoch * self.n_pieces_batch_division
        ppo_valid_percent_list = []
        sampler = TrajPoolSampler(n_div=self.n_pieces_batch_division, traj_pool=traj_pool, flag=task, ratio=ratio)
        n_batch = len(sampler)
        for e in range(self.ppo_epoch):
            # print亮紫('pulse')
            sample_iter = sampler.reset_and_get_iter()
            for i in range(n_batch):
                # ! get traj fragment
                sample = next(sample_iter)
                if (e+i)==0 and self.extral_train_loop:
                    assert False
                    self.debug_pytorch_graph(task, sample, e)
                # ! build graph, then update network!
                self.at_optimizer.zero_grad(); self.ct_optimizer.zero_grad()
                # self.ae_optimizer.zero_grad()
                loss_final, others = self.establish_pytorch_graph(task, sample, e)
                loss_final = loss_final*0.5
                loss_final.backward()
                # log
                ppo_valid_percent_list.append(others.pop('PPO valid percent').item())
                self.log_trivial(dictionary=others); others = None
                nn.utils.clip_grad_norm_(self.at_parameter, self.max_grad_norm)
                self.at_optimizer.step(); self.ct_optimizer.step()
                # ! overwatch other details
            pass # finish small batch update
            if ppo_valid_percent_list[-1] < 0.70: 
                print亮黄('policy change too much, epoch terminate early'); break
        pass # finish all epoch update

        print亮黄(np.array(ppo_valid_percent_list))
        self.log_trivial_finalize()

        print亮红('Peak Memory Allocated %.2f GB'%(torch.cuda.max_memory_allocated()/1073741824))
        print亮红('Leaky Memory Allocated %.2f GB'%(torch.cuda.memory_allocated()/1073741824))

        self.ppo_update_cnt += 1
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

        batchsize = advantage.shape[0]
        batch_agent_size = advantage.shape[0]*advantage.shape[1]

        assert flag == 'train'
        newPi_value, newPi_actionLogProb, entropy, probs, others = self.policy_and_critic.evaluate_actions(obs, action=action, test_mode=False)
        entropy_loss = entropy.mean()

        # friend threat estimation
        # def gather_with_nan(src, index):
        #     assert src.shape[-1]==1; src_ = src.squeeze(-1)
        #     n_subject = src_.shape[-1]
        #     src__ = torch.zeros(size=(*src_.shape[:-1],1), device=src_.device, dtype=src_.dtype) + float('nan')
        #     src__ = torch.cat((src_, src__), -1)
        #     gather = torch.gather(src__, dim=-1, index=torch.nan_to_num(index,nan=n_subject).long())
        #     return gather
        # friend_threat_hat = others['friend_threat']
        # friend_threat_id = others['friend_threat_id_with_nan']
        # friend_threat = gather_with_nan(real_threat, friend_threat_id)
        # nan_filter2 = ~torch.isnan(friend_threat)
        # nan_filter = ~torch.isnan(friend_threat_id)
        # assert (nan_filter==nan_filter2).all()
        # friend_threat_loss = F.mse_loss(friend_threat_hat[nan_filter], friend_threat[nan_filter])

        # threat approximation
        SAFE_LIMIT = 11
        filter = (real_threat<SAFE_LIMIT) & (real_threat>=0)
        threat_loss = F.mse_loss(others['threat'][filter], real_threat[filter])
        if self.turn_off_threat_est: 
            # print('清空threat_loss')
            threat_loss = 0
        # if n==14: est_check(x=others['threat'][filter], y=real_threat[filter])
        
        # probs_loss (should be turn off)
        # n_actions = probs.shape[-1]
        # penalty_prob_line = (1/n_actions)*0.12
        # probs_loss = (penalty_prob_line - torch.clamp(probs, min=0, max=penalty_prob_line)).mean()
        # if not self.add_prob_loss:
        #     probs_loss = torch.zeros_like(probs_loss)

        # dual clip ppo core
        E = newPi_actionLogProb - oldPi_actionLogProb
        E_clip = torch.zeros_like(E)
        E_clip = torch.where(advantage > 0, torch.clamp(E, max=np.log(1.0+self.clip_param)), E_clip)
        E_clip = torch.where(advantage < 0, torch.clamp(E, min=np.log(1.0-self.clip_param), max=np.log(5) ), E_clip)
        ratio = torch.exp(E_clip)
        policy_loss = -(ratio*advantage).mean()

        # # auto encoder loss
        # ae_io = others['ae_io']
        # ae_loss = F.mse_loss(ae_io['obs'], ae_io['obs_hat'])
        

        # add all loses
        value_loss = 0.5 * F.mse_loss(real_value, newPi_value)
        AT_net_loss = policy_loss -entropy_loss*self.entropy_coef # + probs_loss*20
        CT_net_loss = value_loss * 1.0 + threat_loss * 0.1 # + friend_threat_loss*0.01
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
            # 'AE_new_loss':              AE_new_loss,
        }
        # print('ae_loss',ae_loss)


        return loss_final, others



    def debug_pytorch_graph(self, flag, sample, n):


        def mybuild_loss(flag, sample, n):
            obs = _2tensor(sample['obs'])
            advantage = _2tensor(sample['advantage'])
            action = _2tensor(sample['action'])
            oldPi_actionLogProb = _2tensor(sample['actionLogProb'])
            real_value = _2tensor(sample['return'])
            real_threat = _2tensor(sample['threat'])
            batchsize = advantage.shape[0]
            batch_agent_size = advantage.shape[0]*advantage.shape[1]
            assert flag == 'train'
            newPi_value, newPi_actionLogProb, entropy, probs, others = self.policy_and_critic.evaluate_actions(obs, action=action, test_mode=False)
            entropy_loss = entropy.mean()
            # threat approximation
            SAFE_LIMIT = 11
            filter = (real_threat<SAFE_LIMIT) & (real_threat>=0)
            threat_loss = F.mse_loss(others['threat'][filter], real_threat[filter])
            if n%20 == 0: est_check(x=others['threat'][filter], y=real_threat[filter])
            value_loss = 0.5 * F.mse_loss(real_value, newPi_value)
            nz_mask = real_value!=0
            value_loss_abs = (real_value[nz_mask] - newPi_value[nz_mask]).abs().mean()

            CT_net_loss = threat_loss * 0.1 + value_loss * 1.0
            loss_final = CT_net_loss # + AT_net_loss + AE_new_loss # + 
            others = {
                'Value loss Abs':           value_loss_abs,
                'threat loss':              threat_loss,
                'CT_net_loss':              CT_net_loss,
            }
            return loss_final, others

        def step(loss_final, step):
            self.at_optimizer.zero_grad()
            self.ct_optimizer.zero_grad()
            # self.ae_optimizer.zero_grad()
            loss_final.backward()
            self.at_optimizer.step()
            self.ct_optimizer.step()
            # self.ae_optimizer.step()

        for t in range(16):
            self.trivial_dict = {}
            loss_final, others = mybuild_loss(flag, sample, t)
            self.log_trivial(dictionary=others)
            others = None
            
            step(loss_final, t)
            self.log_trivial_finalize(print=False)




