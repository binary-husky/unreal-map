import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
from UTIL.tensor_ops import my_view, Args2tensor_Return2numpy, Args2tensor



def weights_init(m):
    classname = m.__class__.__name__
    if classname ==  'Conv':
        assert False
    elif classname == 'GNet':
        return
    elif classname == 'Sequential':
        return
    elif classname == 'ModuleList':
        return
    elif classname == 'ModuleDict':
        return
    elif classname ==  'MultiHeadAttention':
        return
    elif classname == 'SimpleMLP':
        return
    elif 'Extraction_Module' in classname :
        return
    elif classname ==  'SelfAttention_Module':
        return
    elif classname ==  'ReLU':
        m.inplace=True
    elif classname ==  'Py_Dynamic_Norm':
        return
    elif classname ==  'RandomNet':
        return
    elif classname ==  'BatchNorm1d':
        return
    elif classname ==  'Linear':
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None: m.bias.data.fill_(0)
    elif classname == 'LinearFinal':
        nn.init.orthogonal_(m.weight.data, gain=0.01)
        if m.bias is not None: m.bias.data.fill_(0)
    elif classname == 'LinearBiasInit':
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None: m.bias.data.fill_(0)
    else:
        assert False, ('how to handle the initialization of this class? ', classname)


class SimpleMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=128, use_normalization=False):
        super().__init__()
        activation_func = nn.ReLU
        h_dim = hidden_dim

        if use_normalization:
            self.mlp = nn.Sequential(
                Py_Dynamic_Norm(in_dim, only_for_last_dim=True, exclude_one_hot=True),
                nn.Linear(in_dim, h_dim),
                activation_func(),
                nn.Linear(h_dim, out_dim)
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, h_dim),
                activation_func(),
                nn.Linear(h_dim, out_dim)
            )

    def forward(self,x):
        return self.mlp(x)

# class Extraction_Module(nn.Module):
#     def __init__(self, hidden_dim=128, activate_output=False):
#         super().__init__()
#         self.activation_func = nn.ReLU
#         h_dim = hidden_dim
#         self.attn_right = MultiHeadAttention(n_heads=1, input_dim=h_dim, embed_dim=h_dim)
#         self.attn_left = MultiHeadAttention(n_heads=1, input_dim=h_dim, embed_dim=h_dim)
#         if activate_output:
#             self.MLP = nn.Sequential(nn.Linear(h_dim * 2, h_dim), self.activation_func(), nn.Linear(h_dim, h_dim), self.activation_func())
#         else:
#             self.MLP = nn.Sequential(nn.Linear(h_dim * 2, h_dim), self.activation_func(), nn.Linear(h_dim, h_dim))

#     def forward(self, cluster_enc, right_enc, left_enc):
#         attn_out_right = self.attn_right(q=cluster_enc, k=right_enc, v=right_enc)
#         attn_out_left = self.attn_left(q=cluster_enc, k=left_enc, v=left_enc)
#         concated_attn_result = torch.cat(tensors=(attn_out_right, attn_out_left), dim=2)
#         return self.MLP(concated_attn_result)/2 + cluster_enc/2   # dim = 1*H

class SelfAttention_Module(nn.Module):
    def __init__(self, hidden_dim=128, activate_output=False):
        super().__init__()
        self.activation_func = nn.ReLU
        h_dim = hidden_dim
        self.attn = MultiHeadAttention(n_heads=1, input_dim=h_dim, embed_dim=h_dim)
        if activate_output:
            self.MLP = nn.Sequential(nn.Linear(h_dim * 2, h_dim), self.activation_func(), nn.Linear(h_dim, h_dim), self.activation_func())
            print("activate_output")
        else:
            self.MLP = nn.Sequential(nn.Linear(h_dim * 2, h_dim), self.activation_func(), nn.Linear(h_dim, h_dim))
            print("no activate_output")

    def forward(self, cluster_enc):
        new_cluster_enc = self.attn(q=cluster_enc, k=cluster_enc, v=cluster_enc)
        concated_attn_result = torch.cat(tensors=(new_cluster_enc, cluster_enc), dim=2)
        return self.MLP(concated_attn_result)
        
class Extraction_Module(nn.Module): # merge by MLP version
    def __init__(self, hidden_dim=128, activate_output=False):
        super().__init__()
        self.activation_func = nn.ReLU
        h_dim = hidden_dim
        self.attn_right = MultiHeadAttention(n_heads=1, input_dim=h_dim, embed_dim=h_dim)
        self.attn_left = MultiHeadAttention(n_heads=1, input_dim=h_dim, embed_dim=h_dim)
        if activate_output:
            self.MLP = nn.Sequential(nn.Linear(h_dim * 3, h_dim), self.activation_func(), nn.Linear(h_dim, h_dim), self.activation_func())
            print("activate_output")
        else:
            self.MLP = nn.Sequential(nn.Linear(h_dim * 3, h_dim), self.activation_func(), nn.Linear(h_dim, h_dim))
            print("no activate_output")

    def forward(self, cluster_enc, right_enc, left_enc):
        attn_out_right = self.attn_right(q=cluster_enc, k=right_enc, v=right_enc)
        attn_out_left = self.attn_left(q=cluster_enc, k=left_enc, v=left_enc)
        concated_attn_result = torch.cat(tensors=(cluster_enc, attn_out_right, attn_out_left), dim=2)   # dim = 3*H
        return self.MLP(concated_attn_result)   # dim = 1*H

'''
    比Extraction_Module，MLP分别处理left和right，返回提取到的特征（H维）
'''
class Action_Extraction_Module(nn.Module):
    def __init__(self, hidden_dim=128):
        from .reinforce_foundation import CoopAlgConfig
        super().__init__()
        self.reverse_container = CoopAlgConfig.reverse_container
        activation_func = nn.ReLU
        h_dim = hidden_dim
        self.attn_right = MultiHeadAttention(n_heads=1, input_dim=h_dim, embed_dim=h_dim)
        self.MLP_right = nn.Sequential(nn.Linear(h_dim * 2, h_dim), activation_func())
        self.attn_left = MultiHeadAttention(n_heads=1, input_dim=h_dim, embed_dim=h_dim)
        self.MLP_left = nn.Sequential(nn.Linear(h_dim * 2, h_dim), activation_func())

    def forward(self, cluster_enc, right_enc, left_enc):
        attn_out_right = self.attn_right(q=cluster_enc, k=right_enc, v=right_enc)
        cat_right = torch.cat(tensors=(cluster_enc, attn_out_right), dim=2)   # dim = 2*H
        if not self.reverse_container:
            attn_out_left = self.attn_left(q=left_enc, k=cluster_enc, v=cluster_enc)
            cat_left = torch.cat(tensors=(left_enc, attn_out_left), dim=2)   # dim = 2*H
        else:
            attn_out_left = self.attn_left(q=cluster_enc, k=left_enc, v=left_enc)
            cat_left = torch.cat(tensors=(cluster_enc, attn_out_left), dim=2)   # dim = 2*H

        feature_top = self.MLP_right(cat_right)
        feature_bottom = self.MLP_left(cat_left)

        return feature_top, feature_bottom   # dim = 1*H
'''
    比Extraction_Module多了降到单个值的MLP
'''
class Value_Extraction_Module(nn.Module):  
    def __init__(self, n_cluster, hidden_dim=128):
        super().__init__()
        activation_func = nn.ReLU
        h_dim = hidden_dim
        self.MLP1 = nn.Sequential(nn.Linear(h_dim, h_dim), 
                                 activation_func(), 
                                 nn.Linear(h_dim, h_dim//4), 
                                 activation_func())
        self.MLP2 = nn.Sequential(nn.Linear(n_cluster * (h_dim//4), h_dim), 
                                  activation_func(), 
                                  nn.Linear(h_dim, 1))

    def forward(self, cluster_enc):
        v = self.MLP1(cluster_enc)
        v = my_view(x=v, shape=[0, -1])
        v = self.MLP2(v)
        return v   # dim = 1*H


























class GNet(nn.Module):
    def __init__(self, num_agents, num_entities, basic_vec_len, K_EMod=2 ,hidden_dim=128):
        super().__init__()
        from .reinforce_foundation import CoopAlgConfig
        n_cluster = CoopAlgConfig.g_num
        n_agent = num_agents
        self.n_agent = n_agent
        h_dim = hidden_dim
        n_entity = num_entities
        basic_vec_len = basic_vec_len
        activation_func = nn.ReLU

        if not CoopAlgConfig.one_more_container:
            self._n_cluster = _n_cluster = n_cluster
        else:
            self._n_cluster = _n_cluster = n_cluster + 1

        agent_emb_dim = n_agent + basic_vec_len    +  _n_cluster
        entity_emb_dim = n_entity + basic_vec_len                     +  _n_cluster
        cluster_emb_dim = _n_cluster               +  n_agent         +  n_entity

        self.cluster_enter_encoder = SimpleMLP(in_dim=cluster_emb_dim, out_dim=h_dim, use_normalization=True)
        self.entity_enter_encoder = SimpleMLP(in_dim=entity_emb_dim, out_dim=h_dim, use_normalization=True)
        self.agent_enter_encoder = SimpleMLP(in_dim=agent_emb_dim, out_dim=h_dim, use_normalization=True)

        self.value_head_top_ExtMod = Value_Extraction_Module(n_cluster=_n_cluster)
        self.value_head_bottom_ExtMod = Value_Extraction_Module(n_cluster=_n_cluster)

        self.K_EMod = K_EMod
        self.extraction_modules          = nn.ModuleList([ Extraction_Module(activate_output=False)    for i in range(self.K_EMod)])
        self.clusters_internal_attention = nn.ModuleList([ SelfAttention_Module(activate_output=False) for i in range(self.K_EMod)])

        self.agent_cluster_networking = MultiHeadAttention(n_heads=1, input_dim=h_dim, embed_dim=h_dim)
        self.feature_extraction_4_action = Action_Extraction_Module()

        self.fifo_net_top = nn.ModuleDict({
            'INet': nn.Sequential(
                nn.Linear(h_dim, h_dim // 4),
                activation_func(),
                LinearFinal(h_dim // 4, 1)
             ),
            'ONet': nn.Sequential(
                nn.Linear(h_dim+1, h_dim // 4),
                activation_func(),
                LinearFinal(h_dim // 4, 1)
            )
        })

        self.norm_输入 = Py_Dynamic_Norm(input_size=entity_emb_dim*n_entity+agent_emb_dim*n_agent, only_for_last_dim=True)
        self.解码网络1 = SimpleMLP(in_dim=h_dim*_n_cluster, out_dim=entity_emb_dim*n_entity+agent_emb_dim*n_agent)

        self.test_mode = CoopAlgConfig.eval_mode
        self.is_recurrent = False
        self.apply(weights_init)
        return

    def _get_act_log_probs(self, distribution, action):
        return distribution.log_prob(action.squeeze(-1)).unsqueeze(-1)

    def _get_state_value(self, cluster_enc):
        value_top = self.value_head_top_ExtMod(cluster_enc=cluster_enc)
        value_bottom = self.value_head_bottom_ExtMod(cluster_enc=cluster_enc)
        return value_top, value_bottom

    def get_feature(self, all_emb, eval_mode):
        agent_final_emb = all_emb['agent_final_emb']    # shape = $n_batch.$n_agent.$core_vec
        entity_final_emb = all_emb['entity_final_emb']
        cluster_final_emb = all_emb['cluster_final_emb']
        agent_enc   = self.agent_enter_encoder(agent_final_emb)
        entity_enc  = self.entity_enter_encoder(entity_final_emb)
        cluster_enc = self.cluster_enter_encoder(cluster_final_emb)

        for i in range(self.K_EMod):
            cluster_enc = self.extraction_modules[i](cluster_enc=cluster_enc, right_enc=agent_enc, left_enc=entity_enc)
            cluster_enc = self.clusters_internal_attention[i](cluster_enc=cluster_enc)

        mat_cluster_agent = self.agent_cluster_networking(q=cluster_enc, k=agent_enc, just_attn_weight=True)
        mat_cluster_entity = self.agent_cluster_networking(q=cluster_enc, k=entity_enc, just_attn_weight=True)

        logits_agent_cluster = mat_cluster_agent.transpose(1,2)
        logits_cluster_entity = mat_cluster_entity
        value_top, value_bottom = self._get_state_value(cluster_enc)

        top_feature, bottom_feature = self.feature_extraction_4_action(cluster_enc=cluster_enc, right_enc=agent_enc, left_enc=entity_enc)
        return top_feature, logits_cluster_entity, value_top, value_bottom, {'编码':None, '解码':None}

    def get_fifo_act_new(self, logits_agent_cluster, eval_mode, eval_actions=None):
        act_dist = Categorical(logits = logits_agent_cluster)
        if not self.test_mode:
            act = act_dist.sample() if not eval_mode else eval_actions
        else:
            act = torch.argmax(act_dist.probs, axis=2)
        actLogProbs = self._get_act_log_probs(act_dist, act)
        # sum up the log prob of all agents
        actLogProbs = actLogProbs.sum(1)
        distEntropy = act_dist.entropy().mean() if eval_mode else None
        return act, actLogProbs, distEntropy, act_dist.probs

    def get_fifo_act(self, net, feature, eval_mode, eval_actions=None):
        i_act_logits = net['INet'](feature) # attn_enc $thread.$_n_cluster.$h_dim
        assert i_act_logits.shape[-1] == 1
        i_act_logits = i_act_logits.squeeze(-1)

        i_act_dist = Categorical(logits=i_act_logits)
        if not self.test_mode:
            i_act = i_act_dist.sample() if not eval_mode else eval_actions[:, 0]
        else:
            i_act = torch.argmax(i_act_dist.probs, axis=1)
        i_act_oh = torch.nn.functional.one_hot(i_act, i_act_logits.shape[-1])
        o_act_logits = net['ONet'](torch.cat([feature, i_act_oh.unsqueeze(-1)], dim=-1))
        assert o_act_logits.shape[-1] == 1
        o_act_logits = o_act_logits.squeeze(-1)

        o_act_dist = Categorical(logits=o_act_logits)
        if not self.test_mode:
            o_act = o_act_dist.sample() if not eval_mode else eval_actions[:, 1]
        else:
            o_act = torch.argmax(o_act_dist.probs, axis=1)
        act = torch.cat(tensors=[i_act.unsqueeze(-1), o_act.unsqueeze(-1)], dim=1)
        actLogProbs = self._get_act_log_probs(i_act_dist, i_act) + self._get_act_log_probs(o_act_dist, o_act)
        distEntropy = i_act_dist.entropy().mean() + o_act_dist.entropy().mean() if eval_mode else None
        return act, actLogProbs, distEntropy, o_act_dist.probs

    @Args2tensor_Return2numpy
    def act(self, all_emb):
        return self._act(all_emb)

    @Args2tensor
    def evaluate_actions(self, embedding, action):
        return self._act(embedding, True, eval_actions=action)

    def _act(self, all_emb, eval_mode=False, eval_actions=None):
        eval_top_act, eval_bottom_act = (eval_actions[:, (0, 1)], eval_actions[:,(2,3)]) if eval_mode \
                                   else (None,                    None                 )

        top_feature, bottom_feature, value_top, value_bottom = self.get_feature(all_emb)

        act_top, actLogProbs_top, distEntropy_top, probs_top = self.get_fifo_act(net=self.fifo_net_top, feature=top_feature,
                                                                                eval_mode=eval_mode, eval_actions=eval_top_act)

        act_bottom, actLogProbs_bottom, distEntropy_bottom, probs_bottom = self.get_fifo_act(net=self.fifo_net_bottom, feature=bottom_feature,
                                                                                eval_mode=eval_mode, eval_actions=eval_bottom_act)

        act = torch.cat(tensors=[act_top, act_bottom], dim=1)

        if not eval_mode:
            return act, value_top, value_bottom, actLogProbs_top, actLogProbs_bottom
        else:
            return value_top, actLogProbs_top, distEntropy_top, probs_top, \
                value_bottom, actLogProbs_bottom, distEntropy_bottom, probs_bottom

    def chain_acts(self):
        return

    def get_fifo_act(self, net, feature, eval_mode, eval_actions=None):
        i_act_logits = net['INet'](feature)
        i_act_dist = Categorical(logits=i_act_logits)
        i_act = i_act_dist.sample() if not eval_mode else eval_actions[:, 0]
        i_act_oh = torch.nn.functional.one_hot(i_act, i_act_logits.shape[-1])
        o_act_logits = net['ONet'](torch.cat([feature, i_act_oh], dim=-1))
        o_act_dist = Categorical(logits=o_act_logits)
        o_act = o_act_dist.sample() if not eval_mode else eval_actions[:, 1]
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

    def forward(self, q, k=None, v=None, mask=None, return_attn=False, just_attn_weight=False):
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
        if just_attn_weight:
            assert self.n_heads == 1
            return compatibility.squeeze(0)
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


class LinearBiasInit(nn.Module):

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(LinearBiasInit, self).__init__()
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
    def __init__(self, input_size, only_for_last_dim, exclude_one_hot=True):
        super().__init__()
        assert only_for_last_dim
        self.exclude_one_hot = exclude_one_hot
        self.mean = nn.Parameter(torch.zeros(input_size, requires_grad=False), requires_grad=False)
        self.var = nn.Parameter(torch.ones(input_size, requires_grad=False), requires_grad=False)
        self.n_sample = nn.Parameter(torch.zeros(1, requires_grad=False, dtype=torch.long), requires_grad=False)
        if self.exclude_one_hot: 
            self.one_hot_filter = nn.Parameter(torch.ones(input_size, requires_grad=False, dtype=torch.bool), requires_grad=False)
        self.input_size = input_size

    def forward(self, x, get_mu_var=False):
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
                if self.exclude_one_hot:    # 滤除-1,0和1的点位
                    self.one_hot_filter.data &= ~(((_2dx != 0) & (_2dx != 1) & (_2dx != -1)).any(dim=0))
                self.mean.data = torch.where(self.one_hot_filter, self.mean, new_mean) if self.exclude_one_hot else new_mean # new_mean
                new_var_clip = torch.clamp(new_var, min=0.01, max=1000)
                self.var.data = torch.where(self.one_hot_filter, self.var, new_var_clip) if self.exclude_one_hot else new_var_clip
                self.n_sample.data = tot_count
        if get_mu_var:
            return self.mean, self.var
        x = torch.clip((x - self.mean) / torch.sqrt(self.var + 1e-8), -10, 10)
        return x


    @Args2tensor_Return2numpy
    def get_mean_var(self, x):
        return self.forward(x, get_mu_var=True)

'''
python main.py --note norm-no-limit-1
python main.py --note norm-no-limit-2
python main.py --note norm-no-limit-3
'''