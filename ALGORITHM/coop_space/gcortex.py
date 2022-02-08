import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
from UTILS.tensor_ops import my_view, Args2tensor_Return2numpy, Args2tensor



def weights_init(m):
    classname = m.__class__.__name__
    if classname ==  'Conv':
        assert False
    elif classname == 'GNet':
        return
    elif classname == 'Sequential':
        return
    elif classname == 'ModuleDict':
        return
    elif classname ==  'MultiHeadAttention':
        return
    elif classname ==  'ReLU':
        m.inplace=True
    elif classname ==  'Py_Dynamic_Norm':
        return
    elif classname ==  'Linear':
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None: m.bias.data.fill_(0)
    elif classname == 'LinearFinal':
        nn.init.orthogonal_(m.weight.data, gain=0.01)
        m.bias.data.fill_(0)
    else:
        assert False, ('how to handle the initialization of this class? ', classname)






class GNet(nn.Module):

    def __init__(self, num_agents, num_entities, basic_vec_len, hidden_dim=128):
        from .reinforce_foundation import CoopAlgConfig
        super().__init__()
        n_cluster = CoopAlgConfig.g_num
        n_agent = num_agents

        h_dim = hidden_dim
        n_entity = num_entities
        activation_func = nn.ReLU

        if not CoopAlgConfig.one_more_container:
            _n_cluster = n_cluster
        else:
            _n_cluster = n_cluster + 1

        TopNetDim = {
            'I_I': h_dim,
            'I_O': _n_cluster,
            'O_I': h_dim + _n_cluster,
            'O_O': _n_cluster,
        }

        if not CoopAlgConfig.reverse_container:
            BottomNetDim = {
                'I_I': h_dim,
                'I_O': n_entity,
                'O_I': h_dim + n_entity,
                'O_O': n_entity,
            }
        else:
            BottomNetDim = {
                'I_I': h_dim,
                'I_O': _n_cluster,
                'O_I': h_dim + _n_cluster,
                'O_O': _n_cluster,
            }






        agent_emb_dim = n_agent + basic_vec_len    +  _n_cluster
        entity_emb_dim = n_entity + basic_vec_len                     +  _n_cluster
        cluster_emb_dim = _n_cluster              +  n_agent           +  n_entity

        self.cluster_enter_encoder = nn.Sequential(
            # Py_Dynamic_Norm(cluster_emb_dim, only_for_last_dim=True),
            nn.Linear(cluster_emb_dim, h_dim),
            activation_func(),
            nn.Linear(h_dim, h_dim)
        )

        self.entity_enter_encoder = nn.Sequential(
            # Py_Dynamic_Norm(entity_emb_dim, only_for_last_dim=True),
            nn.Linear(entity_emb_dim, h_dim),
            activation_func(),
            nn.Linear(h_dim, h_dim)
        )

        self.agent_enter_encoder = nn.Sequential(
            # Py_Dynamic_Norm(agent_emb_dim, only_for_last_dim=True),
            nn.Linear(agent_emb_dim, h_dim),
            activation_func(),
            nn.Linear(h_dim, h_dim)
        )

        self.attns = nn.ModuleDict({
            'right': MultiHeadAttention(n_heads=1, input_dim=h_dim, embed_dim=h_dim),
            'left': MultiHeadAttention(n_heads=1, input_dim=h_dim, embed_dim=h_dim),
            'top': MultiHeadAttention(n_heads=1, input_dim=h_dim, embed_dim=h_dim),
            'bottom': MultiHeadAttention(n_heads=1, input_dim=h_dim, embed_dim=h_dim),
            'right_vtop': MultiHeadAttention(n_heads=1, input_dim=h_dim, embed_dim=h_dim),
            'left_vtop': MultiHeadAttention(n_heads=1, input_dim=h_dim, embed_dim=h_dim),
            'right_vbottom': MultiHeadAttention(n_heads=1, input_dim=h_dim, embed_dim=h_dim),
            'left_vbottom': MultiHeadAttention(n_heads=1, input_dim=h_dim, embed_dim=h_dim),
        })

        self.central_zip = nn.Sequential(
            nn.Linear(h_dim * 4, h_dim),
            activation_func(),
            nn.Linear(h_dim, h_dim)
        )


        self.downsample = nn.ModuleDict({
            'top_downsample': nn.Linear(h_dim * 2, h_dim // 2),
            'top_nonlin':   nn.Sequential(
                                nn.Linear(_n_cluster * h_dim // 2, h_dim),
                                activation_func(),
                                nn.Linear(h_dim, h_dim)
                            ),
            'bottom_downsample': nn.Linear(h_dim * 2, h_dim // 2),
            'bottom_nonlin':   nn.Sequential(
                                nn.Linear(_n_cluster * h_dim // 2, h_dim),
                                activation_func(),
                                nn.Linear(h_dim, h_dim)
                            ),
        })



        self.value_head_top = nn.ModuleDict({
            'centralize':nn.Sequential(
                            nn.Linear(h_dim * 4, h_dim),
                            activation_func(),
                            nn.Linear(h_dim, h_dim)
                        ),
            'downsample':nn.Sequential(
                            nn.Linear(h_dim, h_dim // 4),
                        ),
            'nonlin'    :nn.Sequential(
                            nn.Linear(_n_cluster * h_dim // 4, h_dim),
                            activation_func(),
                            nn.Linear(h_dim, 1)
                        ),
        })

        self.value_head_bottom = nn.ModuleDict({
            'centralize':nn.Sequential(
                            nn.Linear(h_dim * 4, h_dim),
                            activation_func(),
                            nn.Linear(h_dim, h_dim)
                        ),
            'downsample':nn.Sequential(
                            nn.Linear(h_dim, h_dim // 4),
                        ),
            'nonlin'    :nn.Sequential(
                            nn.Linear(_n_cluster * h_dim // 4, h_dim),
                            activation_func(),
                            nn.Linear(h_dim, 1)
                        ),
        })

        self.fifo_net_top = nn.ModuleDict({
            'INet': nn.Sequential(
                nn.Linear(TopNetDim['I_I'], h_dim // 2),
                activation_func(),
                LinearFinal(h_dim // 2, TopNetDim['I_O'])
             ),
            'ONet': nn.Sequential(
                nn.Linear(TopNetDim['O_I'], h_dim // 2),
                activation_func(),
                LinearFinal(h_dim // 2, TopNetDim['O_O'])
            )
        })

        self.fifo_net_bottom = nn.ModuleDict({
            'INet': nn.Sequential(
                nn.Linear(BottomNetDim['I_I'], h_dim // 2),
                activation_func(),
                LinearFinal(h_dim // 2, BottomNetDim['I_O'])
             ),
            'ONet': nn.Sequential(
                nn.Linear(BottomNetDim['O_I'], h_dim // 2),
                activation_func(),
                LinearFinal(h_dim // 2, BottomNetDim['O_O'])
            )
        })

        self.is_recurrent = False
        self.apply(weights_init)
        return

    @Args2tensor_Return2numpy
    def act(self, all_emb, test_mode=False):
        return self._act(all_emb, test_mode=test_mode)

    @Args2tensor
    def evaluate_actions(self, embedding, action):
        return self._act(embedding, eval_mode=True, eval_actions=action)

    def _act(self, all_emb, eval_mode=False, eval_actions=None, test_mode=False):
        eval_top_act, eval_bottom_act = (eval_actions[:, (0, 1)], eval_actions[:,(2,3)]) if eval_mode \
                                   else (None,                    None                 )

        top_feature, bottom_feature, value_top, value_bottom = self.get_feature(all_emb)

        act_top, actLogProbs_top, distEntropy_top, probs_top = self.get_fifo_act(net=self.fifo_net_top, feature=top_feature,
                                                                                eval_mode=eval_mode, eval_actions=eval_top_act, test_mode=test_mode)

        act_bottom, actLogProbs_bottom, distEntropy_bottom, probs_bottom = self.get_fifo_act(net=self.fifo_net_bottom, feature=bottom_feature,
                                                                                eval_mode=eval_mode, eval_actions=eval_bottom_act, test_mode=test_mode)

        act = torch.cat(tensors=[act_top, act_bottom], dim=1)

        if not eval_mode:
            return act, value_top, value_bottom, actLogProbs_top, actLogProbs_bottom
        else:
            return value_top, actLogProbs_top, distEntropy_top, probs_top, \
                value_bottom, actLogProbs_bottom, distEntropy_bottom, probs_bottom

    def chain_acts(self):
        return

    def get_fifo_act(self, net, feature, eval_mode, eval_actions=None, test_mode=False):
        i_act_logits = net['INet'](feature)
        i_act_dist = Categorical(logits=i_act_logits)
        if not test_mode:  
            i_act = i_act_dist.sample() if not eval_mode else eval_actions[:, 0]
        else: 
            i_act = torch.argmax(i_act_dist.probs, axis=-1)
        i_act_oh = torch.nn.functional.one_hot(i_act, i_act_logits.shape[-1])
        o_act_logits = net['ONet'](torch.cat([feature, i_act_oh], dim=-1))
        o_act_dist = Categorical(logits=o_act_logits)
        if not test_mode:  
            o_act = o_act_dist.sample() if not eval_mode else eval_actions[:, 1]
        else: 
            o_act = torch.argmax(o_act_dist.probs, axis=-1)
        act = torch.cat(tensors=[i_act.unsqueeze(-1), o_act.unsqueeze(-1)], dim=1)
        actLogProbs = self._get_act_log_probs(i_act_dist, i_act) + self._get_act_log_probs(o_act_dist, o_act)
        distEntropy = i_act_dist.entropy().mean() + o_act_dist.entropy().mean() if eval_mode else None
        return act, actLogProbs, distEntropy, o_act_dist.probs



    def get_feature(self, all_emb):
        agent_final_emb = all_emb['agent_final_emb']
        entity_final_emb = all_emb['entity_final_emb']
        cluster_final_emb = all_emb['cluster_final_emb']
        agent_enc   = self.agent_enter_encoder(agent_final_emb)
        entity_enc  = self.entity_enter_encoder(entity_final_emb)
        cluster_enc = self.cluster_enter_encoder(cluster_final_emb)
        right_attn_enc  = self._attention(src=cluster_enc, passive=agent_enc, attn_key='right')
        left_attn_enc   = self._attention(src=cluster_enc, passive=entity_enc, attn_key='left')
        central_enc = torch.cat(tensors=(right_attn_enc, left_attn_enc), dim=2)
        central_enc = self.central_zip(central_enc)

        right_attn_enc_vtop  = self._attention(src=cluster_enc, passive=agent_enc, attn_key='right_vtop')
        left_attn_enc_vtop   = self._attention(src=cluster_enc, passive=entity_enc, attn_key='left_vtop')
        central_enc_vtop = torch.cat(tensors=(right_attn_enc_vtop, left_attn_enc_vtop), dim=2)
        value_top = self._get_value_way2top(central_enc_vtop)
        right_attn_enc_vbottom  = self._attention(src=cluster_enc, passive=agent_enc, attn_key='right_vbottom')
        left_attn_enc_vbottom   = self._attention(src=cluster_enc, passive=entity_enc, attn_key='left_vbottom')
        central_enc_vbottom = torch.cat(tensors=(right_attn_enc_vbottom, left_attn_enc_vbottom), dim=2)
        value_bottom = self._get_value_way2bottom(central_enc_vbottom)



        top_attn_enc    = self._attention(src=central_enc, passive=agent_enc, attn_key='top')
        bottom_attn_enc = self._attention(src=central_enc, passive=entity_enc, attn_key='bottom')
        top_feature     = self._merge_top_feature(top_attn_enc)
        bottom_feature  = self._merge_bottom_feature(bottom_attn_enc)
        return top_feature, bottom_feature, value_top, value_bottom

    def _get_act_log_probs(self, distribution, action):
        return distribution.log_prob(action.squeeze(-1)).unsqueeze(-1)

    def _attention(self, src, passive, attn_key):
        src_q = src
        passive_k = passive
        passive_v = passive
        src_attn_res = self.attns[attn_key](q=src_q, k=passive_k, v=passive_v)
        src_attn_enc = torch.cat(tensors=(src, src_attn_res), dim=2)
        return src_attn_enc

    def _get_value_way1(self, enc):
        value_each_group = self.value_head_way1(enc)
        value = torch.sum(value_each_group, dim=1)
        return value

    def _get_value_way2top(self, enc):
        enc = self.value_head_top['centralize'](enc)
        enc = self.value_head_top['downsample'](enc)
        enc = my_view(x=enc, shape=[0, -1])
        value = self.value_head_top['nonlin'](enc)
        return value
    def _get_value_way2bottom(self, enc):
        enc = self.value_head_bottom['centralize'](enc)
        enc = self.value_head_bottom['downsample'](enc)
        enc = my_view(x=enc, shape=[0, -1])
        value = self.value_head_bottom['nonlin'](enc)
        return value

    def _merge_top_feature(self, top_attn_enc):
        enc = self.downsample['top_downsample'](top_attn_enc)
        enc = my_view(x=enc, shape=[0, -1]) # 0 for dim unchanged, -1 for auto calculation
        enc = self.downsample['top_nonlin'](enc)
        return enc

    def _merge_bottom_feature(self, bottom_attn_enc):
        enc = self.downsample['bottom_downsample'](bottom_attn_enc)
        enc = my_view(x=enc, shape=[0, -1]) # 0 for dim unchanged, -1 for auto calculation
        enc = self.downsample['bottom_nonlin'](enc)
        return enc



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

    def forward(self, q, k=None, v=None, mask=None, return_attn=False):
        """
        :param q: queries (batch_size, n_query, input_dim)
        :param k: data (batch_size, n_key, input_dim)
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
        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask.bool()] = -math.inf

        attn = F.softmax(compatibility, dim=-1)

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask.bool()] = 0
            attn = attnc

        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        if return_attn:
            return out, attn
        return out


class LinearFinal(nn.Module):

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(LinearFinal, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )



class Py_Dynamic_Norm(nn.Module):
    def __init__(self, input_size, only_for_last_dim):
        super().__init__()
        assert only_for_last_dim
        self.mean = nn.Parameter(torch.zeros(input_size, requires_grad=False), requires_grad=False)
        self.var = nn.Parameter(torch.ones(input_size, requires_grad=False), requires_grad=False)
        self.n_sample = nn.Parameter(torch.ones(1, requires_grad=False)*(1e-4), requires_grad=False)
        self.input_size = input_size

    # 这里加入no_grad
    def forward(self, x):
        assert self.input_size == x.shape[-1], ('self.input_size',self.input_size,'x.shape[-1]',x.shape[-1])
        _2dx = x.detach().view(-1, self.input_size)
        this_batch_size = _2dx.shape[0]
        assert this_batch_size>=1

        if self.training:
            with torch.no_grad():
                this_batch_mean = torch.mean(_2dx, dim=0)
                this_batch_var = torch.var(_2dx, dim=0, unbiased=False)
                if torch.isnan(this_batch_var).any():
                    print(x.shape, _2dx.shape, _2dx)
                assert _2dx.dim() == 2
                delta = this_batch_mean - self.mean
                tot_count = self.n_sample + this_batch_size
                new_mean = self.mean + delta * this_batch_size / tot_count
                m_a = self.var * (self.n_sample)
                m_b = this_batch_var * (this_batch_size)
                M2 = m_a + m_b + torch.square(delta) * self.n_sample * this_batch_size / (self.n_sample + this_batch_size)
                new_var = M2 / (self.n_sample + this_batch_size)
                self.mean.data = new_mean
                self.var.data = torch.clamp(new_var, min=1e-6)
                self.n_sample.data = tot_count

        x = torch.clip((x - self.mean) / torch.sqrt(self.var + 1e-8), -20, 20)
        return x
'''
python main.py --note norm-no-limit-1
python main.py --note norm-no-limit-2
python main.py --note norm-no-limit-3
'''