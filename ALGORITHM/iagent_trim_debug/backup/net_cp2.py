import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn.modules.linear import Linear
from ..commom.attention import MultiHeadAttention
from ..commom.norm import DynamicNorm
from ..commom.mlp import LinearFinal, SimpleMLP
from UTIL.colorful import print亮紫
from UTIL.tensor_ops import my_view, Args2tensor_Return2numpy, Args2tensor, repeat_at


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


class Pnet(nn.Module):
    def __init__(self, num_agents, n_basic_dim, obs_dim, n_action, K_EMod=2 ,hidden_dim=128, use_m_gpu=None, seperate_critic=False):
        super().__init__()
        from .foundation import AlgorithmConfig
        self.use_agent_wise_attention = AlgorithmConfig.agent_wise_attention

        self.obs_dim = obs_dim*12
        self.n_action = n_action
        n_agent = num_agents
        self.n_agent = n_agent
        h_dim = hidden_dim
        activation_func = nn.ReLU
        self.K_EMod = K_EMod
        self.seperate_critic = seperate_critic
        print亮紫('是否使用actor-critic网络分离？:',seperate_critic)

        self.AT_obs_encoder = SimpleMLP(in_dim=self.obs_dim, out_dim=h_dim, use_normalization=False)
        self.AT_extract_mod = EXTRACT(self.K_EMod, h_dim)


        if self.seperate_critic:
            self.CT_obs_encoder = SimpleMLP(in_dim=self.obs_dim, out_dim=h_dim, use_normalization=False)
            self.CT_extract_mod = EXTRACT(self.K_EMod, h_dim)


        self.AT_get_logit = nn.Sequential( nn.Linear(h_dim, h_dim // 4), activation_func(inplace=True),
                                            LinearFinal(h_dim // 4, self.n_action))
        
        self.CT_get_value = nn.Sequential( nn.Linear(h_dim, h_dim // 4), activation_func(inplace=True),
                                           LinearFinal(h_dim // 4, 1))
        self.CT_encode = nn.Sequential(
                                    nn.Linear(15*12, 64), nn.ReLU(),
                                    nn.Linear(64, 15))
        self.CT_getvalue = nn.Sequential(
            Linear(15, 16), nn.ReLU(inplace=True),
            Linear(16, 1),
        )
        n_possible_reward = 4

  # OUTPUT: reward_pred ($n_thread, $n_agent, $n_agent)

    def get_feature(self, obs, eval_mode):
        agent_enc   = self.AT_obs_encoder(obs)
        agent_enc = self.AT_extract_mod(agent_enc)

        logits = self.AT_get_logit(agent_enc)

        # agent_enc_ct = self.CT_obs_encoder(obs)
        # agent_enc_ct = self.CT_extract_mod(agent_enc_ct)
        # value = self.CT_get_value(agent_enc_ct)
        ct_phi = self.CT_encode(obs) 
        value = self.CT_getvalue(ct_phi)
        

        return logits, value, None

    @staticmethod
    def get_binary_array(n_int, n_bits=8, dtype=torch.float32):
        arr = torch.zeros(size=(*n_int.shape, n_bits), dtype=dtype, device=n_int.device)
        for i in range(8):
            arr[:, i] = (n_int%2==1).int()
            n_int = n_int / 2
            n_int = n_int.int()
        return arr

    def logit2act(self, logits_agent_cluster, eval_mode, test_mode, eval_actions=None):
        act_dist = Categorical(logits = logits_agent_cluster)
        if not test_mode:
            act = act_dist.sample() if not eval_mode else eval_actions
        else:
            act = torch.argmax(act_dist.probs, axis=2)
        actLogProbs = self._get_act_log_probs(act_dist, act)
        # sum up the log prob of all agents
        distEntropy = act_dist.entropy().mean(-1) if eval_mode else None
        return act, actLogProbs, distEntropy, act_dist.probs


    def _get_act_log_probs(self, distribution, action):
        return distribution.log_prob(action.squeeze(-1)).unsqueeze(-1)

    """
        main forward
    """
    def forward(self, obs, eval_mode, test_mode, eval_act):
        logits, value, _ = self.get_feature(obs, eval_mode)
        act, actLogProbs, distEntropy, probs= \
            self.logit2act(logits, eval_mode=eval_mode, test_mode=test_mode, eval_actions=eval_act)
        if not eval_mode:
            return act, value, actLogProbs
        else:
            others={}
            others['ae_io'] = {'obs':obs, 'obs_hat': obs}
            return value, actLogProbs, distEntropy, probs, others




"""
    network initialize
"""
class Net(nn.Module):
    def __init__(self, num_agents, n_basic_dim, obs_dim, n_action, K_EMod=2, hidden_dim=128, use_m_gpu=None, use_normalization=True, seperate_critic=False):
        super().__init__()
        n_agent = num_agents
        self.n_agent = n_agent
        h_dim = hidden_dim
        basic_vec_len = n_basic_dim
        activation_func = nn.ReLU
        self.K_EMod = K_EMod
        self.use_normalization = use_normalization
        if self.use_normalization:
            print亮紫('Net use normalization')
            self._batch_norm = DynamicNorm(obs_dim, only_for_last_dim=True, exclude_one_hot=True, exclude_nan=True)
        self.pnet = Pnet(num_agents, n_basic_dim, obs_dim, n_action, K_EMod ,hidden_dim, use_m_gpu, seperate_critic)
        if use_m_gpu is not None:
            P = lambda module: torch.nn.DataParallel(module, device_ids=use_m_gpu) 
            self.pnet = P(self.pnet)

        self.is_recurrent = False
        self.apply(weights_init)
        return

    @Args2tensor_Return2numpy
    def act(self, obs, test_mode):
        return self._act(obs, test_mode)

    @Args2tensor
    def evaluate_actions(self, obs, action, test_mode):
        return self._act(obs, test_mode, eval_mode=True, eval_actions=action)

    def _act(self, obs, test_mode, eval_mode=False, eval_actions=None):
        if self.use_normalization:
            obs = self._batch_norm(obs)
        obs = torch.nan_to_num_(obs[:,:,:12], 0)
        obs = my_view(obs, [0, 0, -1])

        eval_act = eval_actions if eval_mode else None
        return self.pnet(obs, eval_mode, test_mode, eval_act)



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
