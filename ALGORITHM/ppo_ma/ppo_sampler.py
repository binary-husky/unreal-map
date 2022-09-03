import torch, math, traceback
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from random import randint, sample
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from UTIL.colorful import *
from UTIL.tensor_ops import _2tensor, __hash__, repeat_at
from config import GlobalConfig as cfg
from UTIL.gpu_share import GpuShareUnit
class TrajPoolSampler():
    def __init__(self, n_div, traj_pool, flag, prevent_batchsize_oom=False):
        self.n_pieces_batch_division = n_div
        self.prevent_batchsize_oom = prevent_batchsize_oom    

        if self.prevent_batchsize_oom:
            assert self.n_pieces_batch_division==1, ('?')

        self.num_batch = None
        self.container = {}
        self.warned = False
        assert flag=='train'
        req_dict =        ['avail_act', 'obs', 'action', 'actionLogProb', 'return', 'reward', 'value']
        req_dict_rename = ['avail_act', 'obs', 'action', 'actionLogProb', 'return', 'reward', 'state_value']
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
        self.mini_batch_size = math.ceil(self.big_batch_size / self.n_pieces_batch_division)  

    def __len__(self):
        return self.n_pieces_batch_division

    def determine_max_n_sample(self):
        assert self.prevent_batchsize_oom
        if not hasattr(TrajPoolSampler,'MaxSampleNum'):
            # initialization
            TrajPoolSampler.MaxSampleNum =  [int(self.big_batch_size*(i+1)/50) for i in range(50)]
            max_n_sample = self.big_batch_size
        elif TrajPoolSampler.MaxSampleNum[-1] > 0:  
            # meaning that oom never happen, at least not yet
            # only update when the batch size increases
            if self.big_batch_size > TrajPoolSampler.MaxSampleNum[-1]: TrajPoolSampler.MaxSampleNum.append(self.big_batch_size)
            max_n_sample = self.big_batch_size
        else:
            # meaning that oom already happened, choose TrajPoolSampler.MaxSampleNum[-2] to be the limit
            assert TrajPoolSampler.MaxSampleNum[-2] > 0
            max_n_sample = TrajPoolSampler.MaxSampleNum[-2]
        return max_n_sample

    def reset_and_get_iter(self):
        if not self.prevent_batchsize_oom:
            self.sampler = BatchSampler(SubsetRandomSampler(range(self.big_batch_size)), self.mini_batch_size, drop_last=False)
        else:
            max_n_sample = self.determine_max_n_sample()
            n_sample = min(self.big_batch_size, max_n_sample)
            if not hasattr(self,'reminded'):
                self.reminded = True
                print('droping %.1f percent samples..'%((self.big_batch_size-n_sample)/self.big_batch_size*100))
            self.sampler = BatchSampler(SubsetRandomSampler(range(n_sample)), n_sample, drop_last=False)

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
