import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
from UTIL.tensor_ops import my_view

class MultiHeadAttention(nn.Module):
    # taken from https://github.com/wouterkool/attention-tsp/blob/master/graph_encoder.py
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        if embed_dim is not None:
            self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, k=None, v=None, mask=None, return_attn=False, return_attn_weight=False):
        if q.dim()<=3: 
            out = self.forward_(q, k, v, mask, return_attn, return_attn_weight)
            if return_attn:
                out, attn = out
                assert attn.shape[0]==1
                attn = attn.squeeze(0)
                return out, attn
            return out


        hyper_dim = q.shape[:-2]
        q = my_view(q, [-1, *q.shape[-2:]])
        if k is not None: 
            k = my_view(k, [-1, *k.shape[-2:]])
        if v is not None: 
            v = my_view(v, [-1, *v.shape[-2:]])
        if mask is not None: mask = my_view(mask, [-1, *mask.shape[-2:]])
        out = self.forward_(q, k, v, mask, return_attn, return_attn_weight)
        if return_attn:
            out, attn = out
            if hyper_dim is not None:
                out = out.view(*hyper_dim, *out.shape[-2:])
                attn = attn.view(*hyper_dim, *attn.shape[-2:]) #??
            return out, attn
        else:
            if hyper_dim is not None:
                out = out.view(*hyper_dim, *q.shape[-2:])
            return out

    def forward_(self, q, k=None, v=None, mask=None, return_attn=False, return_attn_weight=False):
        """
        :param q: queries (batch_size, n_query, input_dim)
        :param k: data (batch_size, n_key/graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        if k is None:
            k = q  # compute self-attention
        if v is None:
            v = k
        # k should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = k.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        kflat = k.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)
        vflat = v.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(kflat, self.W_key).view(shp)
        V = torch.matmul(vflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))
        if return_attn_weight:
            assert self.n_heads == 1
            if mask is not None:
                mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
                compatibility[mask.bool()] = -math.inf
            return compatibility.squeeze(0)

        # Optionally apply mask to prevent attention
        if mask is not None:    # expand to n_heads
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask.bool()] = -math.inf

        attn = F.softmax(compatibility, dim=-1)

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask.bool()] = 0
            attn = attnc

        # 为了在这里解决 0*nan = nan 的问题，输入必须将V中的nan转化为0
        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        if return_attn:
            return out, attn
        return out




class SimpleAttention(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.W_query = nn.Parameter(torch.Tensor(h_dim, h_dim))
        self.W_key = nn.Parameter(torch.Tensor(h_dim, h_dim))
        self.W_val = nn.Parameter(torch.Tensor(h_dim, h_dim))
        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, k, q, v, mask=None):
        Q = torch.matmul(q, self.W_query) 
        K = torch.matmul(k, self.W_key) 
        V = torch.matmul(v, self.W_val)

        norm_factor = 1 / math.sqrt(Q.shape[-1])
        compat = norm_factor * torch.matmul(Q, K.transpose(-1, -2)) 
        if mask is not None: compat[mask.bool()] = -math.inf
        # 为了在这里解决 0*nan = nan 的问题，输入必须将V中的nan转化为0
        score = torch.nan_to_num(F.softmax(compat, dim=-1), 0)
        return torch.matmul(score, V) 
