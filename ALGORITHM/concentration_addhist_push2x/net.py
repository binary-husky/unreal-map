import math
import torch,time,random
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn.modules.linear import Linear
from ..commom.attention import MultiHeadAttention
from ..commom.norm import DynamicNorm
from ..commom.mlp import LinearFinal, SimpleMLP, ResLinear
from UTILS.colorful import print亮紫
from UTILS.tensor_ops import my_view, Args2tensor_Return2numpy, Args2tensor, __hash__, __hashn__, pad_at_dim
from UTILS.tensor_ops import _2cpu2numpy, one_hot_with_nan, gather_righthand, pt_inf


def weights_init(m):
    def init_Linear(m, final_layer=False):
        nn.init.orthogonal_(m.weight.data)
        if final_layer:nn.init.orthogonal_(m.weight.data, gain=0.01)
        if m.bias is not None: nn.init.uniform_(m.bias.data, a=-0.02, b=0.02)

    initial_fn_dict = {
        'Net': None, 'DataParallel':None, 'BatchNorm1d':None, 'Concentration':None,
        'Pnet':None,'Sequential':None,'DataParallel':None,'Tanh':None,
        'ModuleList':None,'ModuleDict':None,'MultiHeadAttention':None,
        'SimpleMLP':None,'Extraction_Module':None,'SelfAttention_Module':None,
        'ReLU':None,'Softmax':None,'DynamicNorm':None,'EXTRACT':None,
        'LinearFinal':lambda m:init_Linear(m, final_layer=True),
        'Linear':init_Linear, 'ResLinear':None, 'LeakyReLU':None,'SimpleAttention':None
    }

    classname = m.__class__.__name__
    assert classname in initial_fn_dict.keys(), ('how to handle the initialization of this class? ', classname)
    init_fn = initial_fn_dict[classname]
    if init_fn is None: return
    init_fn(m)

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
            self.AT_Attention = Extraction_Module(hidden_dim=self.skip_dim, activate_output=True)
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

    # @staticmethod
    # def mask_1d_to_2d(ve_dead_C):
    #     vs_alive = torch.zeros_like(ve_dead_C[...,(0,)]).bool()
    #     v_dead_stack = torch.cat((vs_alive, ve_dead_C), dim=-1)
    #     v_dead_stack = repeat_at(v_dead_stack,insert_dim=-1,n_times=v_dead_stack.shape[-1])
    #     v_dead_stack = v_dead_stack | v_dead_stack.transpose(2,3)
    #     return v_dead_stack

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
        compat = norm_factor * torch.matmul(Q, K.transpose(2, 3)) 
        if mask is not None: compat[mask.bool()] = -math.inf
        score = torch.nan_to_num(F.softmax(compat, dim=-1), 0)
        # ----------- motivational brach -------------
        return torch.matmul(score, V) 


class Extraction_Module(nn.Module): # merge by MLP version
    def __init__(self, hidden_dim=128, activate_output=False):
        super().__init__()
        h_dim = hidden_dim
        from .foundation import AlgorithmConfig
        if AlgorithmConfig.use_my_attn:
            self.attn = SimpleAttention(h_dim=h_dim)
            print('use my attn')

        if activate_output:
            self.MLP = nn.Sequential(nn.Linear(h_dim * 2, h_dim), nn.ReLU(inplace=True))
            print("activate_output")
        else:
            self.MLP = nn.Sequential(nn.Linear(h_dim * 2, h_dim))
            print("no activate_output")

    def forward(self, agent_enc, mask=None):
        attn_out = self.attn(q=agent_enc, k=agent_enc, v=agent_enc, mask=mask)
        concated_attn_result = torch.cat(tensors=(agent_enc, attn_out), dim=-1)
        return self.MLP(concated_attn_result)



"""
    network initialize
"""
class Net(nn.Module):
    def __init__(self, 
                    rawob_dim, 
                    n_action, 
                    use_m_gpu=None, 
                    use_normalization=True, 
                    n_focus_on=None, 
                    actor_attn_mod=False,
                    dual_conc=True):
        super().__init__()
        h_dim = 48
        self.skip_connect = True
        self.n_action = n_action
        assert use_m_gpu is None
        self.use_normalization = use_normalization
        self.n_focus_on = n_focus_on
        self.adopt_selfattn = actor_attn_mod
        self.dual_conc = dual_conc
        from .foundation import AlgorithmConfig
        self.alternative_critic = AlgorithmConfig.alternative_critic
        # observation normalization
        if self.use_normalization:
            self._batch_norm = DynamicNorm(rawob_dim, only_for_last_dim=True, exclude_one_hot=True, exclude_nan=True)

        self.AT_obs_encoder = nn.Sequential(nn.Linear(rawob_dim, h_dim), nn.ReLU(inplace=True), nn.Linear(h_dim, h_dim))

        if self.dual_conc:
            self.MIX_conc_core_f = Concentration(
                            n_focus_on=self.n_focus_on-1, h_dim=h_dim, 
                            skip_connect=self.skip_connect, 
                            skip_connect_dim=rawob_dim, 
                            adopt_selfattn=actor_attn_mod)
            self.MIX_conc_core_h = Concentration(
                            n_focus_on=self.n_focus_on, h_dim=h_dim, 
                            skip_connect=self.skip_connect, 
                            skip_connect_dim=rawob_dim, 
                            adopt_selfattn=actor_attn_mod)
        else:
            self.MIX_conc_core = Concentration(
                            n_focus_on=self.n_focus_on, h_dim=h_dim, 
                            skip_connect=self.skip_connect, 
                            skip_connect_dim=rawob_dim, 
                            adopt_selfattn=actor_attn_mod)
 
        tmp_dim = h_dim if not self.dual_conc else h_dim*2
        self.CT_get_value = nn.Sequential(Linear(tmp_dim, h_dim), nn.ReLU(inplace=True),Linear(h_dim, 1))
        self.CT_get_threat = nn.Sequential(Linear(tmp_dim, h_dim), nn.ReLU(inplace=True),Linear(h_dim, 1))

        if self.alternative_critic:
            self.CT_get_value_alternative_critic = nn.Sequential(Linear(tmp_dim, h_dim), nn.ReLU(inplace=True),Linear(h_dim, 1))

        # part3
        self.check_n = self.n_focus_on*2
        self.AT_get_logit_db = nn.Sequential(  
            nn.Linear(tmp_dim, h_dim), nn.ReLU(inplace=True),
            nn.Linear(h_dim, h_dim//2), nn.ReLU(inplace=True),
            LinearFinal(h_dim//2, self.n_action))

        self.is_recurrent = False
        self.apply(weights_init)
        return

    # two ways to support avail_act, but which one is better?
    def logit2act(self, logits_agent_cluster, eval_mode, test_mode, eval_actions=None, avail_act=None):
        if avail_act is not None: logits_agent_cluster = torch.where(avail_act>0, logits_agent_cluster, -pt_inf())
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
    def act(self, *args, **kargs):
        act = self._act if self.dual_conc else self._act_singlec
        return act(*args, **kargs)

    @Args2tensor
    def evaluate_actions(self, *args, **kargs):
        act = self._act if self.dual_conc else self._act_singlec
        return act(*args, **kargs, eval_mode=True)

    def div_entity(self, mat, type=[(0,), # self
                                    (1, 2, 3, 4, 5, 6,  7, 8, 9, 10,11),     # current
                                    (12,13,14,15,16,17, 18,19,20,21,22,23),  # history
                                    ],   
                                    n=24
                                    ):
        # def div_entity(self, mat, type=[(0,), (1, 2, 3, 4, 5), (6, 7, 8, 9, 10, 11)], n=24):
        # def div_entity(self, mat, type=[(0,), (1, 2, 3, 4, 5), (6, 7, 8, 9, 10, 11)], n=12):
        if mat.shape[-2]==n:
            tmp = (mat[..., t, :] for t in type)
        elif mat.shape[-1]==n:
            tmp = (mat[..., t] for t in type)
        return tmp

    def _act(self, obs, test_mode, eval_mode=False, eval_actions=None, avail_act=None):
        eval_act = eval_actions if eval_mode else None
        others = {}
        if self.use_normalization:
            obs = self._batch_norm(obs)
        # obs = obs[:,:,:12]
        mask_dead = torch.isnan(obs).any(-1)    # find dead agents
        obs = torch.nan_to_num_(obs, 0)         # replace dead agents' obs, from NaN to 0
        v = self.AT_obs_encoder(obs)


        zs, ze_f, ze_h          = self.div_entity(obs)        
        vs, ve_f, ve_h          = self.div_entity(v)          
        _, ve_f_dead, ve_h_dead = self.div_entity(mask_dead)  

        # concentration module
        vh_C, vh_M = self.MIX_conc_core_h(vs=vs, ve=ve_h, ve_dead=ve_h_dead, skip_connect_ze=ze_h, skip_connect_zs=zs)
        vf_C, vf_M = self.MIX_conc_core_f(vs=vs, ve=ve_f, ve_dead=ve_f_dead, skip_connect_ze=ze_f, skip_connect_zs=zs)

        # fuse forward path
        v_C_fuse = torch.cat((vf_C, vh_C), dim=-1)  # (vs + vs + check_n + check_n)
        logits = self.AT_get_logit_db(v_C_fuse) # diverge here


        # motivation encoding fusion
        v_M_fuse = torch.cat((vf_M, vh_M), dim=-1)
        # motivation objectives
        value = self.CT_get_value(v_M_fuse)
        threat = self.CT_get_threat(v_M_fuse)

        assert not self.alternative_critic


        act, actLogProbs, distEntropy, probs = self.logit2act(logits, eval_mode=eval_mode, 
                                                                test_mode=test_mode, eval_actions=eval_act, avail_act=avail_act)

        def re_scale(t):
            SAFE_LIMIT = 11
            r = 1. /2. * SAFE_LIMIT
            return (torch.tanh_(t/r) + 1.) * r

        others['threat'] = re_scale(threat)
        if not eval_mode: return act, value, actLogProbs
        else:             return value, actLogProbs, distEntropy, probs, others

