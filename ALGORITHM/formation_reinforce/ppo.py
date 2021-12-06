import torch, math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from UTILS.colorful import *
from UTILS.tensor_ops import _2tensor, _2cpu2numpy, repeat_at
from UTILS.tensor_ops import my_view, scatter_with_nan, sample_balance
from config import GlobalConfig

class TrajPoolSampler():
    def __init__(self, n_div, traj_pool, flag):
        self.n_pieces_batch_division = n_div
        self.num_batch = None
        self.container = {}

        assert flag=='train'
        req_dict =        ['obs', 'action', 'actionLogProb', 'return', 'reward', 'value']
        req_dict_rename = ['obs', 'action', 'actionLogProb', 'return', 'reward', 'state_value']

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
        big_batch_size = -1  # vector should have same length, check it!
        
        # load traj into a 'container'
        for key_index, key in enumerate(req_dict):
            key_name =  req_dict[key_index]
            key_rename = req_dict_rename[key_index]
            if not hasattr(traj_pool[0], key_name): continue
            set_item = np.concatenate([getattr(traj, key_name) for traj in traj_pool], axis=0)
            if not (big_batch_size==set_item.shape[0] or (big_batch_size<0)):
                print('error')
            assert big_batch_size==set_item.shape[0] or (big_batch_size<0), (key,key_index)
            big_batch_size = set_item.shape[0]
            self.container[key_rename] = set_item    # 指针赋值

        # normalize advantage inside the batch
        self.container[advantage_rename] = self.container[return_rename] - self.container[value_rename]
        self.container[advantage_rename] = ( self.container[advantage_rename] - self.container[advantage_rename].mean() ) / (self.container[advantage_rename].std() + 1e-5)
        # size of minibatch for each agent
        mini_batch_size = math.ceil(big_batch_size / self.n_pieces_batch_division)  
        self.sampler = BatchSampler(SubsetRandomSampler(range(big_batch_size)), mini_batch_size, drop_last=False)

    def __len__(self):
        return len(self.sampler)

    def get_iter(self):
        for indices in self.sampler:
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



class PPO():
    def __init__(self, policy_and_critic, ppo_config, mcv=None):
        self.policy_and_critic = policy_and_critic
        self.clip_param = ppo_config.clip_param
        self.ppo_epoch = ppo_config.ppo_epoch
        self.n_pieces_batch_division = ppo_config.n_pieces_batch_division
        self.value_loss_coef = ppo_config.value_loss_coef
        self.entropy_coef = ppo_config.entropy_coef
        self.max_grad_norm = ppo_config.max_grad_norm
        self.add_prob_loss = ppo_config.add_prob_loss
        self.lr = ppo_config.lr
        self.g_optimizer = optim.Adam(policy_and_critic.parameters(), lr=self.lr)
        
        self.all_parameter = list(policy_and_critic.named_parameters())
        self.at_parameter = [(p_name, p) for p_name, p in self.all_parameter if 'AT_' in p_name]
        self.ct_parameter = [(p_name, p) for p_name, p in self.all_parameter if 'CT_' in p_name]
        # self.re_parameter = [(p_name, p) for p_name, p in self.all_parameter if 'RE_' in p_name]
        # 检查剩下是是否全都是不需要训练的参数
        remove_exists = lambda LA,LB: list(set(LA).difference(set(LB)))
        res = self.all_parameter
        res = remove_exists(res, self.at_parameter)
        res = remove_exists(res, self.ct_parameter)
        # res = remove_exists(res, self.re_parameter)
        for p_name, p in res:   
            assert not p.requires_grad, ('unclassified parameter:',p_name)
        # 不再需要参数名
        self.at_parameter = [p for p_name, p in self.all_parameter if 'AT_' in p_name]
        self.ct_parameter = [p for p_name, p in self.all_parameter if 'CT_' in p_name]
        # self.re_parameter = [p for p_name, p in self.all_parameter if 'RE_' in p_name]




        self.g_update_delayer = 0
        self.g_initial_value_loss = 0
        # 轮流训练式
        self.mcv = mcv
        self.ppo_update_cnt = 0
        self.loss_bias =  ppo_config.balance
        self.batch_size_reminder = True
        
        from .focal import FocalLoss
        self.focal_loss = FocalLoss(alpha=torch.Tensor([1, 1, 1, 1]).to(GlobalConfig.device), gamma=2)    # 2class default alpha: 0.25   gamma: 2.0

    def train_on_traj(self, traj_pool, task):
        # print(traj_pool) 从轨迹中采样
        g_value_loss_epoch = 0
        g_action_loss_epoch = 0
        g_dist_entropy_epoch = 0
        probs_loss_epoch = 0
        reward_est_loss_epoch = 0
        num_updates = self.ppo_epoch * self.n_pieces_batch_division


        flag = task
        print('train')

        ppo_valid_percent_list = []
        self.trivial_dict = {}
        for e in range(self.ppo_epoch):
            sampler = TrajPoolSampler(n_div=self.n_pieces_batch_division, traj_pool=traj_pool, flag=task)
            n_batch = len(sampler)
            sample_iter = sampler.get_iter()
            for _ in range(n_batch):
                # ! get traj fragment
                sample = next(sample_iter)
                self.g_optimizer.zero_grad()
                loss_final, others = self.establish_pytorch_graph(task, sample)
                loss_final = loss_final*0.5
                loss_final.backward()
                # log
                ppo_valid_percent_list.append(others.pop('PPO valid percent'))
                self.log_trivial(dictionary=others); others = None
                nn.utils.clip_grad_norm_(self.at_parameter, self.max_grad_norm)
                self.g_optimizer.step()
            pass # finish small batch update
            if ppo_valid_percent_list[-1] < 0.70:
                print亮黄('policy change too much, epoch terminate early')
                break
        pass # finish all epoch update

        print亮黄(np.array(ppo_valid_percent_list))
        self.log_trivial_finalize()

        self.ppo_update_cnt += 1
        return self.ppo_update_cnt

    def log_trivial(self, dictionary):
        for key in dictionary:
            if key not in self.trivial_dict: self.trivial_dict[key] = []
            item = dictionary[key].item() if hasattr(dictionary[key], 'item') else dictionary[key]
            self.trivial_dict[key].append(item)

    def log_trivial_finalize(self):
        for key in self.trivial_dict:
            self.trivial_dict[key] = np.array(self.trivial_dict[key])
        
        print_buf = ['[ppo.py] ']
        for key in self.trivial_dict:
            self.trivial_dict[key] = self.trivial_dict[key].mean()
            print_buf.append(' %s:%.3f, '%(key, self.trivial_dict[key]))
            if self.mcv is not None:  self.mcv.rec(self.trivial_dict[key], key)
        print紫(''.join(print_buf))
        if self.mcv is not None:
            self.mcv.rec_show()
        self.trivial_dict = None


    def establish_pytorch_graph(self, flag, sample):
        obs = _2tensor(sample['obs'])
        advantage = _2tensor(sample['advantage'])
        action = _2tensor(sample['action'])
        oldPi_actionLogProb = _2tensor(sample['actionLogProb'])
        real_value = _2tensor(sample['return'])

        batchsize = advantage.shape[0]
        batch_agent_size = advantage.shape[0]*advantage.shape[1]

        assert flag == 'train'
        newPi_value, newPi_actionLogProb, entropy, probs = self.policy_and_critic.evaluate_actions(obs, action=action, test_mode=False)

        entropy_loss = entropy.mean()
        n_actions = probs.shape[-1]
        if self.add_prob_loss: assert n_actions <= 15  # 
        penalty_prob_line = (1/n_actions)*0.12
        probs_loss = (penalty_prob_line - torch.clamp(probs, min=0, max=penalty_prob_line)).mean()
        if not self.add_prob_loss:
            probs_loss = torch.zeros_like(probs_loss)

        #dual clip ppo core
        E = newPi_actionLogProb - oldPi_actionLogProb
        E_clip = torch.zeros_like(E)
        E_clip = torch.where(advantage > 0, torch.clamp(E, max=np.log(1.0+self.clip_param)), E_clip)
        E_clip = torch.where(advantage < 0, torch.clamp(E, min=np.log(1.0-self.clip_param), max=np.log(5) ), E_clip)
        ratio = torch.exp(E_clip)
        policy_loss = -(ratio*advantage).mean()

        # add all loses
        value_loss = 0.5 * F.mse_loss(real_value, newPi_value)
        loss_final =  (policy_loss -entropy_loss*self.entropy_coef + probs_loss*20)    +value_loss*self.value_loss_coef #  + reward_est_loss*0.5
        ppo_valid_percent = ((E_clip == E).int().sum()/batch_agent_size).item()

        others = {
            'Policy loss':              policy_loss,
            'Entropy loss':             entropy_loss,
            'Value loss':               value_loss,
            'NewPi actionLogProb':      newPi_actionLogProb.detach().mean(),
            'PPO valid percent':        ppo_valid_percent,
            'Probs loss':               probs_loss,
        }


        if self.batch_size_reminder:
            print亮红('the batch size in each ppo update is %d, memory allocated %.2f GB'%(batchsize, torch.cuda.memory_allocated()/1073741824)); self.batch_size_reminder = False

        return loss_final, others


# 加个训练缓冲池会不会好些？
# 合并奖励？
# 减小学习率
# 减少网络层数

# self.g_optimizer.zero_grad()
# loss_final.backward()
# [p.grad for p in self.at_parameter]
# [p.grad for p in self.ct_parameter]
# [p.grad for p in self.re_parameter]











# 4468 , 827012 , 620
# torch.gather(reward_est, dim=2, index=vis_f_masked_nan2zero)
# reward_filtered = reward_filtered.unsqueeze(-2).expand(r_batch_size,50,50)
# reward_est_loss = 0.5 * F.mse_loss(reward_est, reward_filtered)

# reward_est_loss2 = 0.5 * F.mse_loss(reward_est, real_value.squeeze(-1))
# print('reward_est_loss_verify: %.4f'%reward_est_loss.item())

'''
reward_class_onehot_ = reward_class_onehot_.flatten()
reward_est_ = my_view(reward_est, [-1, 0])
reward_est_loss = F.cross_entropy(input=reward_est_, target=reward_class_onehot_) # nn.CrossEntropyLoss(, )





rand = torch.randn(2, 2, 3) + 1

reward_class = rand.clone()
v,indices = reward_class.max(-1, keepdim=True)
reward_class_onehot_ = torch.where(reward_class==v,1,0)
reward_class_onehot_ = indices

reward_est_ = rand.clone()
reward_est_loss = F.cross_entropy(input=reward_est_, target=reward_class_onehot_) # nn.CrossEntropyLoss(, )


'''

'''
interested_rewardest_masked = scatter_with_nan(vis_f_masked, n_agent, 'bool').flatten()
reward_est__ = reward_est.flatten()[interested_rewardest_masked]
reward_filtered__ = repeat_at(reward_filtered, -2, 50).flatten()[interested_rewardest_masked]
reward_est_loss = 0.5 * F.mse_loss(reward_est__, reward_filtered__)
print('reward_est_loss_verify: %.4f'%reward_est_loss.item())

# 过滤掉全0奖励的batch， 过滤掉不可观测的友方， +再过滤掉零奖励，打印奖励误差，用于调试
NZ_mask = (reward_filtered__ != 0)
reward_est__ = reward_est__[NZ_mask]
reward_filtered__ = reward_filtered__[NZ_mask]
reward_est_loss_verify = 0.5 * F.mse_loss(reward_est__, reward_filtered__)
print('reward_est_loss_verify no zero: %.4f'%reward_est_loss_verify.item())
'''

# reward_est_NZ = reward_est.flatten()
# reward_NZ = reward_filtered.flatten()
# NZ_mask = (reward_NZ != 0)
# reward_est_NZ = reward_est_NZ[NZ_mask]
# reward_NZ = reward_NZ[NZ_mask]
# reward_est_loss2 = 0.5 * F.mse_loss(reward_est_NZ, reward_NZ)
# print(reward_est_loss2)


# print(reward_est_loss2)