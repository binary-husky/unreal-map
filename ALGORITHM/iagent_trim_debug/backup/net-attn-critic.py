import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn.modules.linear import Linear
from ..commom.attention import MultiHeadAttention
from ..commom.norm import DynamicNorm
from ..commom.mlp import LinearFinal, SimpleMLP, ResLinear
from UTILS.colorful import print亮紫
from UTILS.tensor_ops import my_view, Args2tensor_Return2numpy, Args2tensor, repeat_at


def weights_init(m):
    def init_Linear(m, final_layer=False):
        nn.init.orthogonal_(m.weight.data)
        if final_layer:nn.init.orthogonal_(m.weight.data, gain=0.01)
        if m.bias is not None: m.bias.data.fill_(0)

    initial_fn_dict = {
        'Net': None, 'DataParallel':None, 'BatchNorm1d':None, 
        'Pnet':None,'Sequential':None,'DataParallel':None,'Tanh':None,
        'ModuleList':None,'ModuleDict':None,'MultiHeadAttention':None,
        'SimpleMLP':None,'Extraction_Module':None,'SelfAttention_Module':None,
        'ReLU':None,'Softmax':None,'DynamicNorm':None,'EXTRACT':None,
        'LinearFinal':lambda m:init_Linear(m, final_layer=True),
        'Linear':init_Linear, 'ResLinear':None, 'LeakyReLU':None
    }

    classname = m.__class__.__name__
    assert classname in initial_fn_dict.keys(), ('how to handle the initialization of this class? ', classname)
    init_fn = initial_fn_dict[classname]
    if init_fn is None: return
    init_fn(m)


class EXTRACT(nn.Module):
    def __init__(self, K_EMod, h_dim):
        super().__init__()
        print亮紫('no agent wise attention')
        self.K_EMod = K_EMod
        self.extract_mod = nn.ModuleList(
            [nn.Sequential(  nn.Linear(h_dim, h_dim), nn.ReLU(inplace=True) )  for i in range(self.K_EMod-1)]+
            [nn.Linear(h_dim, h_dim)] )

    def forward(self, x):
        for i in range(self.K_EMod):
            x = self.extract_mod[i](x)
        return x





"""
    network initialize
"""
class Net(nn.Module):
    def __init__(self, num_agents, n_basic_dim, obs_dim, n_action, K_EMod=2, hidden_dim=128, use_m_gpu=None, use_normalization=True, seperate_critic=False):
        super().__init__()
        n_agent = num_agents
        self.n_action = n_action
        self.n_agent = n_agent
        h_dim = hidden_dim
        basic_vec_len = n_basic_dim
        lkrelu = nn.ReLU
        self.K_EMod = K_EMod

        # part0
        self.use_normalization = use_normalization
        if self.use_normalization:
            self._batch_norm = DynamicNorm(obs_dim, only_for_last_dim=True, exclude_one_hot=True, exclude_nan=True)
        
        # part1
        self.AE_encode = ResLinear(io_dim=8, h_dim=32, need_input_tf=True, input_tf_dim=n_basic_dim)
        self.AE_decode = nn.Sequential(
                            ResLinear(io_dim=basic_vec_len, h_dim=32, need_input_tf=True, input_tf_dim=8),
                            nn.Linear(basic_vec_len, basic_vec_len))

        # part2
        # self.CT_encode = ResLinear(io_dim=32, h_dim=64, need_input_tf=True, input_tf_dim=15*12, inplace_relu=False)
        self.CT_encode = nn.Sequential(
                            nn.Linear(15, 64), lkrelu(),
                            nn.Linear(64, 16))
        self.CT_attention = MultiHeadAttention(n_heads=1, input_dim=16, embed_dim=16)
        self.CT_getvalue = nn.Sequential(
            Linear(16*2, 16), lkrelu(inplace=True),
            Linear(16, 1),
        )
        # part3
        self.AT_obs_encoder = SimpleMLP(in_dim=12*15, out_dim=h_dim, use_normalization=False)
        self.AT_extract_mod = EXTRACT(self.K_EMod, h_dim)
        # self.AT_encode = ResLinear(io_dim=8, h_dim=32, need_input_tf=True, input_tf_dim=n_basic_dim)
        self.AT_get_logit = nn.Sequential( nn.Linear(h_dim, h_dim // 4), nn.ReLU(inplace=True),
                                LinearFinal(h_dim // 4, self.n_action))
        if use_m_gpu is not None:
            assert False

        self.is_recurrent = False
        self.apply(weights_init)
        return



    def logit2act(self, logits_agent_cluster, eval_mode, test_mode, eval_actions=None):
        # dead_logits = torch.isnan(logits_agent_cluster).any(-1)

        act_dist = Categorical(logits = logits_agent_cluster)
        if not test_mode:  act = act_dist.sample() if not eval_mode else eval_actions
        else:              act = torch.argmax(act_dist.probs, axis=2)
        def _get_act_log_probs(distribution, action):
            return distribution.log_prob(action.squeeze(-1)).unsqueeze(-1)
        actLogProbs = _get_act_log_probs(act_dist, act) # the policy gradient loss will feedback from here
        # sum up the log prob of all agents
        distEntropy = act_dist.entropy().mean(-1) if eval_mode else None
        return act, actLogProbs, distEntropy, act_dist.probs


    @Args2tensor_Return2numpy
    def act(self, obs, test_mode):
        return self._act(obs, test_mode)

    @Args2tensor
    def evaluate_actions(self, obs, action, test_mode):
        return self._act(obs, test_mode, eval_mode=True, eval_actions=action)

    def _act(self, obs, test_mode, eval_mode=False, eval_actions=None):
        if self.use_normalization:
            obs = self._batch_norm(obs)

        others = {}
        eval_act = eval_actions if eval_mode else None
        obs = obs[:,:,:12]   # obs use torch.nan as placeholder !
        mask_dead = torch.isnan(obs).any(-1)


        # ! <<< Auto encoder network >>>
        obs_nan2zero = torch.nan_to_num_(obs, 0)
        phi = self.AE_encode(obs_nan2zero)
        obs_hat = self.AE_decode(phi)

        ## !! detach gradient
        phi_detached = phi.detach()

        # ! <<< Critic network >>>
        ct_phi = self.CT_encode(obs)    # maybe here
        ct_phi_self = ct_phi[:,:,:1]    ; ct_phi_self_mask = mask_dead[:,:,:1]
        ct_phi_others = ct_phi[:,:,1:]  ; ct_phi_others_mask = mask_dead[:,:,1:]
        mask = ct_phi_others_mask.unsqueeze(-2)
        # assert (ct_phi_others_mask == torch.isnan(ct_phi_others).any(-1)).all()
        ct_enc, m_atten_softmax = self.CT_attention(q=ct_phi_self,k=ct_phi_others, mask=mask, return_attn=True)
        value = self.CT_getvalue(
            torch.cat((ct_phi_self.squeeze(-2), ct_enc.squeeze(-2)), -1)
        ) # self.CT_getvalue(ct_phi)#



        ## !! detach gradient
        # m_atten_softmax_detached = m_atten_softmax.detach()

 
        # ! <<< Actor network >>>
        # attn_sort = torch.argsort(m_atten_softmax_detached, dim=-1, descending=True) #.cpu().numpy()
        # attn_top2 = attn_sort[..., :2]
        # atten_index_alias = repeat_at(attn_top2, insert_dim=-1, n_times=obs.shape[-1])
        # obs_focus_ = torch.gather(obs[:,:,1:], dim=-2, index=atten_index_alias)
        # # use torch.nan as placeholder, replace t_agent nan->0, but dead agent still processes NaN in phi[:,:,:1]
        # obs_focus = torch.cat((obs[:,:,:1], obs_focus_), dim=-2)

        # phi2_focus = self.AT_encode(obs_focus)
        # phi2_focus = my_view(phi2_focus, [0, 0, -1])
        # logits = self.AT_get_logit(phi2_focus)
        agent_enc   = self.AT_obs_encoder(my_view(obs, [0, 0, -1]) )
        agent_enc = self.AT_extract_mod(agent_enc)
        logits = self.AT_get_logit(agent_enc)

        act, actLogProbs, distEntropy, probs = \
                    self.logit2act(logits, eval_mode=eval_mode, test_mode=test_mode, eval_actions=eval_act)


        others['ae_io'] = {'obs':obs[~mask_dead], 'obs_hat':obs_hat[~mask_dead]}
        # assert not torch.isnan(others['ae_io']['obs']).any()
        # assert not torch.isnan(others['ae_io']['obs_hat']).any()
        if not eval_mode:
            return act, value, actLogProbs
        else:
            return value, actLogProbs, distEntropy, probs, others




class Extraction_Module(nn.Module): # merge by MLP version
    def __init__(self, hidden_dim=128, activate_output=False):
        super().__init__()
        self.activation_func = nn.ReLU
        h_dim = hidden_dim
        self.attn = MultiHeadAttention(n_heads=1, input_dim=h_dim, embed_dim=h_dim)
        if activate_output:
            self.MLP = nn.Sequential(nn.Linear(h_dim * 2, h_dim), self.activation_func(inplace=True), nn.Linear(h_dim, h_dim), self.activation_func(inplace=True))
            print("activate_output")
        else:
            self.MLP = nn.Sequential(nn.Linear(h_dim * 2, h_dim), self.activation_func(inplace=True), nn.Linear(h_dim, h_dim))
            print("no activate_output")

    def forward(self, agent_enc):
        attn_out = self.attn(q=agent_enc, k=agent_enc, v=agent_enc)
        concated_attn_result = torch.cat(tensors=(agent_enc, attn_out), dim=2)   # dim = 3*H
        return self.MLP(concated_attn_result)   # dim = 1*H
