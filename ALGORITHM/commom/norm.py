"""
    ****************************
    live vector normalization using pytorch, 
    therefore the parameter of normalization (mean and var) 
    can be save together with network parameters
    light up exclude_one_hot=True to prevent onehot component being normalized
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal

class DynamicNorm(nn.Module):
    # ! warning! this module will mess with multi-gpu setting!!
    def __init__(self, input_size, only_for_last_dim, exclude_one_hot=True, exclude_nan=False):
        super().__init__()
        assert only_for_last_dim
        self.exclude_one_hot = exclude_one_hot
        self.mean = nn.Parameter(torch.zeros(input_size, requires_grad=False), requires_grad=False)
        self.var = nn.Parameter(torch.ones(input_size, requires_grad=False), requires_grad=False)
        self.n_sample = nn.Parameter(torch.zeros(1, requires_grad=False, dtype=torch.long), requires_grad=False)
        if self.exclude_one_hot: 
            self.one_hot_filter = nn.Parameter(torch.ones(input_size, requires_grad=False, dtype=torch.bool), requires_grad=False)
        self.input_size = input_size
        self.exclude_nan = exclude_nan

    def forward(self, x, get_mu_var=False):
        assert self.input_size == x.shape[-1], ('self.input_size',self.input_size,'x.shape[-1]',x.shape[-1])
        _2dx = x.detach().reshape(-1, self.input_size)
        if self.exclude_nan: _2dx = _2dx[~torch.isnan(_2dx).any(axis=-1)]
        this_batch_size = _2dx.shape[0]
        # assert this_batch_size>=1
        if this_batch_size<=0:  
            print('Warning! An empty batch just being normalized')
            x = torch.clip_((x - self.mean) / torch.sqrt_(self.var + 1e-8), -10, 10)
            return x
            
        if self.training:
            with torch.no_grad():
                this_batch_mean = torch.mean(_2dx, dim=0)
                this_batch_var = torch.var(_2dx, dim=0, unbiased=False)
                if torch.isnan(this_batch_var).any():
                    assert False, ('nan value detected in normalization! but you can turn on exclude_nan')
                assert _2dx.dim() == 2
                delta = this_batch_mean - self.mean
                tot_count = self.n_sample + this_batch_size
                new_mean = self.mean + delta * this_batch_size / tot_count
                m_a = self.var * (self.n_sample)
                m_b = this_batch_var * (this_batch_size)
                M2 = m_a + m_b + torch.square_(delta) * self.n_sample * this_batch_size / (self.n_sample + this_batch_size)
                new_var = M2 / (self.n_sample + this_batch_size)
                if self.exclude_one_hot:    # 滤除-1,0和1的点位
                    self.one_hot_filter.data &= ~(((_2dx != 0) & (_2dx != 1) & (_2dx != -1)).any(dim=0))
                self.mean.data = torch.where(self.one_hot_filter, self.mean, new_mean) if self.exclude_one_hot else new_mean # new_mean
                new_var_clip = torch.clamp(new_var, min=0.01, max=1000)
                self.var.data = torch.where(self.one_hot_filter, self.var, new_var_clip) if self.exclude_one_hot else new_var_clip
                self.n_sample.data = tot_count
        if get_mu_var:
            return self.mean, self.var
        
        x = torch.clip_((x - self.mean) / torch.sqrt_(self.var + 1e-8), -10, 10)
        return x



