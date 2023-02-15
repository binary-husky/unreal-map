"""
    CASIA, fuqingxu
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
from UTIL.tensor_ops import my_view
from UTIL.tensor_ops import Args2tensor_Return2numpy

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
        self.patience = 1000

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


    # @Args2tensor_Return2numpy
    # def get_mean_var(self, x):
        # return self.forward(x, get_mu_var=True)


class DynamicNormFix(nn.Module):
    # ! warning! this module will mess with multi-gpu setting!!
    def __init__(self, input_size, only_for_last_dim, exclude_one_hot=True, exclude_nan=False):
        super().__init__()
        assert only_for_last_dim
        self.exclude_one_hot = exclude_one_hot
        self.mean = nn.Parameter(torch.zeros(input_size, requires_grad=False), requires_grad=False)
        self.var = nn.Parameter(torch.ones(input_size, requires_grad=False), requires_grad=False)
        self.var_fix = nn.Parameter(torch.ones(input_size, requires_grad=False), requires_grad=False)
        self.min = nn.Parameter(torch.ones(input_size, requires_grad=False)+float('inf'), requires_grad=False)
        self.max = nn.Parameter(torch.ones(input_size, requires_grad=False)-float('inf'), requires_grad=False)
        self.n_sample = nn.Parameter(torch.zeros(1, requires_grad=False, dtype=torch.long), requires_grad=False)
        if self.exclude_one_hot: 
            self.one_hot_filter = nn.Parameter(torch.ones(input_size, requires_grad=False, dtype=torch.bool), requires_grad=False)
        self.input_size = input_size
        self.exclude_nan = exclude_nan
        self.patience = 1000
        self.var_fix_wait = 1000
        
        # var fixing, T2 is maximum x abs value after normalization
        self.T1 = 5
        self.T2 = 10
        self.TD = (self.T2**2 - self.T1**2)/self.T2**2
        self.first_run = True
        self.debug = True

    # 兼容np
    @Args2tensor_Return2numpy
    def np_forward(self, x, freeze=False, get_mu_var=False):
        return self.forward(x, freeze, get_mu_var)
    
    def forward(self, x, freeze=False, get_mu_var=False):
        assert self.input_size == x.shape[-1], ('self.input_size',self.input_size,'x.shape[-1]',x.shape[-1])
        _2dx = x.detach().reshape(-1, self.input_size)
        if self.exclude_nan: _2dx = _2dx[~torch.isnan(_2dx).any(axis=-1)]
        _2dx_view = my_view(_2dx, [-1, 0])
        this_batch_size = _2dx.shape[0]
        # assert this_batch_size>=1
        if this_batch_size<=0:  
            print('Warning! An empty batch just being normalized')
            x = torch.clip_((x - self.mean) / torch.sqrt_(self.var_fix + 1e-8), -10, 10)
            return x
            
        if self.training and (not freeze):
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
                # if self.patience > 0: self.check_errors(_2dx, new_var)
                self.var.data = torch.where(self.one_hot_filter, self.var, new_var) if self.exclude_one_hot else new_var

                # begin fix variance
                max_tmp, _ = _2dx_view.max(0)
                min_tmp, _ = _2dx_view.min(0)

                # if self.first_run:
                if self.patience > 0:
                    self.patience -= 1
                    self.first_run = False
                    self.max.data = torch.maximum(max_tmp, self.max)
                    self.min.data = torch.minimum(min_tmp, self.min)

                else:
                    # self.max.data = torch.maximum(max_tmp, self.max)
                    # self.min.data = torch.minimum(min_tmp, self.min)
                    self.max.data = self.max + (torch.maximum(max_tmp, self.max)-self.max) * this_batch_size / tot_count
                    self.min.data = self.min + (torch.minimum(min_tmp, self.min)-self.min) * this_batch_size / tot_count
                # # if self.debug: self.mcv.rec(max_tmp.squeeze().item(), 'batch max')
                # # if self.debug: self.mcv.rec(min_tmp.squeeze().item(), 'batch min')
                # # if self.debug: self.mcv.rec(torch.maximum(max_tmp, self.max).squeeze().item(), 'hist max')
                # # if self.debug: self.mcv.rec(torch.minimum(min_tmp, self.min).squeeze().item(), 'hist min')
                # if self.debug: self.mcv.rec(self.max.data, 'fixed max')
                # if self.debug: self.mcv.rec(self.min.data, 'fixed min')
                # if self.debug: self.mcv.rec_show()
                
                dm = torch.maximum((self.max - self.mean), (self.mean - self.min))
                # std_th_1 = dm / self.T1
                std_threshold_2 = dm / self.T2
                # var1 = std_th_1**2
                var2 = std_threshold_2**2
                leak = self.TD * self.var + var2    # leak = (var1 - var2)/(var1) *self.var + var2
                new_var_fix = torch.maximum(self.var, leak)
                self.var_fix.data = torch.where(self.one_hot_filter, self.var_fix, new_var_fix) if self.exclude_one_hot else new_var_fix
                
                # if self.debug: self.mcv.rec(self.var.data, 'var')
                # if self.debug: self.mcv.rec(self.var_fix.data, 'var fix')
                # if self.debug: self.mcv.rec(self.var_fix.data-self.var.data, 'delta var')
                # if self.debug: self.mcv.rec((1 - self.mean) / torch.sqrt_(self.var_fix + 1e-8), 'base line +1')
                # if self.debug: self.mcv.rec((-1 - self.mean) / torch.sqrt_(self.var_fix + 1e-8), 'base line -1')
                # if self.debug: self.mcv.rec((10 - self.mean) / torch.sqrt_(self.var_fix + 1e-8), 'base line +10')
                # if self.debug: self.mcv.rec((-10 - self.mean) / torch.sqrt_(self.var_fix + 1e-8), 'base line -10')
                
                
                
                # !!! qq = self.var_fix.data-self.var.data
                # !!! if self.patience > 0 and self.patience < 800 and (not (qq==0).all()):
                # !!!     print('[norm.py] Input issue: cannot be well expressed by normal distribution', torch.where(qq!=0))
                    
                    
                self.n_sample.data = tot_count
                
                
        # t = (_2dx_view - self.mean) / torch.sqrt_(self.var_fix + 1e-8)
        if get_mu_var:
            return self.mean, self.var_fix
        return (x - self.mean) / torch.sqrt_(self.var_fix + 1e-8)

    # def check_errors(self, _2dx, new_var):
    #     self.patience -= 1



'''
test script

import torch, time
from ALGORITHM.common.norm import DynamicNormFix

input_size = 1
only_for_last_dim = True
dynamic_norm = DynamicNormFix(input_size, only_for_last_dim, exclude_one_hot=True, exclude_nan=False)

for _ in range(101100):
    
    # mask = (torch.randn(60, 1, out=None) > 0)
    # x = torch.where(mask,
    #                 torch.randn(60, 1, out=None)*10,
    #                 torch.randn(60, 1, out=None)*5,
    #                 )
    # 左边
    std = 0.01; offset = -0.01;  num = 5
    x3 = torch.randn(num, 1, out=None) * std + offset
  
    # 中间
    std = 0.01; offset = 0;    num = 500
    x2 = torch.randn(num, 1, out=None) * std + offset

    # 右边
    std = 0.01; offset = 1;   num = 5
    x1 = torch.randn(num, 1, out=None) * std + offset
        
        
    # # 左边
    # std = 1; offset = -10;  num = 5
    # x3 = torch.randn(num, 1, out=None) * std + offset
  
    # # 中间
    # std = 1; offset = 5;    num = 500
    # x2 = torch.randn(num, 1, out=None) * std + offset

    # # 右边
    # std = 1; offset = 5;   num = 5
    # x1 = torch.randn(num, 1, out=None) * std + offset
    
    x = torch.cat((x1,x2,x3), 0)
    y = dynamic_norm(x)
    
print(y)
time.sleep(60)

'''