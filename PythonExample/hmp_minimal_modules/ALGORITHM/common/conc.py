import math
import torch,time,random
import torch.nn as nn
import torch.nn.functional as F
from UTIL.tensor_ops import my_view, __hash__, __hashn__, pad_at_dim, gather_righthand

class Concentration(nn.Module):
    def __init__(self, n_focus_on, h_dim, skip_connect=False, skip_connect_dim=0, adopt_selfattn=False):
        super().__init__()
        self.n_focus_on = n_focus_on
        self.skip_connect = skip_connect
        self.skip_dim = h_dim+skip_connect_dim
        self.CT_W_query = nn.Parameter(torch.Tensor(h_dim, h_dim))
        self.CT_W_key = nn.Parameter(torch.Tensor(h_dim, h_dim))
        self.CT_W_val = nn.Parameter(torch.Tensor(h_dim, h_dim))
        self.CT_motivate_mlp = nn.Sequential(nn.Linear(h_dim * 2, h_dim), nn.ReLU(inplace=True))
        self.AT_forward_mlp = nn.Sequential(nn.Linear((n_focus_on+1)*self.skip_dim, h_dim), nn.ReLU(inplace=True))
        self.adopt_selfattn = adopt_selfattn
        if self.adopt_selfattn:
            assert False, ('no longer support')
        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, vs, ve, ve_dead, skip_connect_ze=None, skip_connect_zs=None):
        mask = ve_dead
        Q = torch.matmul(vs, self.CT_W_query) 
        K = torch.matmul(ve, self.CT_W_key) 

        norm_factor = 1 / math.sqrt(Q.shape[-1])
        compat = norm_factor * torch.matmul(Q, K.transpose(2, 3)) 
        assert compat.shape[-2] == 1
        compat = compat.squeeze(-2)
        compat[mask.bool()] = -math.inf
        score = F.softmax(compat, dim=-1)
        # nodes with no neighbours were softmax into nan, fix them to 0
        score = torch.nan_to_num(score, 0)
        # ----------- motivational brach -------------
        Va = torch.matmul(score.unsqueeze(-2), torch.matmul(ve, self.CT_W_val)) 
        v_M = torch.cat((vs, Va), -1).squeeze(-2) 
        v_M_final = self.CT_motivate_mlp(v_M)
        # -----------   forward branch   -------------
        score_sort_index = torch.argsort(score, dim=-1, descending=True)
        score_sort_drop_index = score_sort_index[..., :self.n_focus_on]
        if self.skip_connect:
            ve = torch.cat((ve, skip_connect_ze), -1)
            vs = torch.cat((vs, skip_connect_zs), -1)
        ve_C = gather_righthand(src=ve,  index=score_sort_drop_index, check=False)
        need_padding = (score_sort_drop_index.shape[-1] != self.n_focus_on)
        if need_padding:
            print('the n_focus param is large than input, advise: pad observation instead of pad here')
            ve_C = pad_at_dim(ve_C, dim=-2, n=self.n_focus_on)
        v_C_stack = torch.cat((vs, ve_C), dim=-2)
        if self.adopt_selfattn:
            v_C_stack = self.AT_Attention(v_C_stack, mask=None)

        v_C_flat = my_view(v_C_stack, [0, 0, -1]); assert v_C_stack.dim()==4
        v_C_final = self.AT_forward_mlp(v_C_flat)
        return v_C_final, v_M_final



class ConcentrationHete(nn.Module):
    def __init__(self, n_focus_on, h_dim, skip_connect=False, skip_connect_dim=0, adopt_selfattn=False):
        super().__init__()
        self.n_focus_on = n_focus_on
        self.skip_connect = skip_connect
        self.skip_dim = h_dim+skip_connect_dim
        self.AT_W_query = nn.Parameter(torch.Tensor(h_dim, h_dim))
        self.AT_W_key = nn.Parameter(torch.Tensor(h_dim, h_dim))
        self.AT_W_val = nn.Parameter(torch.Tensor(h_dim, h_dim))
        self.AT_motivate_mlp = nn.Sequential(nn.Linear(h_dim * 2, h_dim), nn.ReLU(inplace=True))
        self.AT_forward_mlp = nn.Sequential(nn.Linear((n_focus_on+1)*self.skip_dim, h_dim), nn.ReLU(inplace=True))
        self.adopt_selfattn = adopt_selfattn
        if self.adopt_selfattn:
            assert False, ('no longer support')
        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, vs, ve, ve_dead, skip_connect_ze=None, skip_connect_zs=None):
        mask = ve_dead
        Q = torch.matmul(vs, self.AT_W_query) 
        K = torch.matmul(ve, self.AT_W_key) 

        norm_factor = 1 / math.sqrt(Q.shape[-1])
        compat = norm_factor * torch.matmul(Q, K.transpose(2, 3)) 
        assert compat.shape[-2] == 1
        compat = compat.squeeze(-2)
        compat[mask.bool()] = -math.inf
        score = F.softmax(compat, dim=-1)
        # nodes with no neighbours were softmax into nan, fix them to 0
        score = torch.nan_to_num(score, 0)
        # ----------- motivational brach -------------
        Va = torch.matmul(score.unsqueeze(-2), torch.matmul(ve, self.AT_W_val)) 
        v_M = torch.cat((vs, Va), -1).squeeze(-2) 
        v_M_final = self.AT_motivate_mlp(v_M)
        # -----------   forward branch   -------------
        score_sort_index = torch.argsort(score, dim=-1, descending=True)
        score_sort_drop_index = score_sort_index[..., :self.n_focus_on]
        if self.skip_connect:
            ve = torch.cat((ve, skip_connect_ze), -1)
            vs = torch.cat((vs, skip_connect_zs), -1)
        ve_C = gather_righthand(src=ve,  index=score_sort_drop_index, check=False)
        need_padding = (score_sort_drop_index.shape[-1] != self.n_focus_on)
        if need_padding:
            print('the n_focus param is large than input, advise: pad observation instead of pad here')
            ve_C = pad_at_dim(ve_C, dim=-2, n=self.n_focus_on)
        v_C_stack = torch.cat((vs, ve_C), dim=-2)
        if self.adopt_selfattn:
            v_C_stack = self.AT_Attention(v_C_stack, mask=None)

        v_C_flat = my_view(v_C_stack, [0, 0, -1]); assert v_C_stack.dim()==4
        v_C_final = self.AT_forward_mlp(v_C_flat)
        return v_C_final, v_M_final

