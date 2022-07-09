import math
import torch,time
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn.modules.linear import Linear
from ..commom.attention import MultiHeadAttention
from ..commom.norm import DynamicNorm
from ..commom.mlp import LinearFinal, SimpleMLP, ResLinear
from UTIL.colorful import print亮紫
from UTIL.tensor_ops import my_view, Args2tensor_Return2numpy, Args2tensor, __hash__, repeat_at


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
        # self.top_n = 11
        self.check_n = 12 # self.top_n+1
        self.check_who = range(12)
        # self.AT_obs_encoder_old = SimpleMLP(in_dim=12*15, out_dim=h_dim, use_normalization=False)


        # self.AT_obs_encoder = nn.Sequential(  
        #     nn.Linear(15*self.check_n, 128), nn.ReLU(inplace=True),
        #     nn.Linear(128, 128))
        # self.AT_extract_mod =  nn.Sequential(  
        #     nn.Linear(128, 128), nn.ReLU(inplace=True),
        #     nn.Linear(128, 128))
        # self.AT_get_logit = nn.Sequential( nn.Linear(128, 32), nn.ReLU(inplace=True),
        #                         LinearFinal(32, self.n_action))
        self.AT_encoder = nn.Sequential(nn.Linear(15, 32), nn.ReLU(inplace=True))
        self.AT_attention = MultiHeadAttention(n_heads=1, input_dim=32, embed_dim=32)

        self.AT_get_logit_db = nn.Sequential(
            nn.Linear(32, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 32), nn.ReLU(inplace=True),
            LinearFinal(32, self.n_action))

        # # self.AT_encode = ResLinear(io_dim=8, h_dim=32, need_input_tf=True, input_tf_dim=n_basic_dim)
        # self.AT_get_logit = nn.Sequential(  
        #     nn.Linear(128, 32), nn.ReLU(inplace=True),
        #     nn.Linear(32, self.n_action))
        # nn.Sequential( nn.Linear(h_dim, h_dim // 4), nn.ReLU(inplace=True),
        #                         LinearFinal(h_dim // 4, self.n_action))


        # # self.AT_extract_mod = nn.Sequential(  
        # #     nn.Linear(128, 128), nn.ReLU(inplace=True),
        # #     nn.Linear(128, 128) 
        # # )
        # # # self.AT_encode = ResLinear(io_dim=8, h_dim=32, need_input_tf=True, input_tf_dim=n_basic_dim)
        # # self.AT_get_logit = nn.Sequential(
        # #     nn.Linear(128, 32), nn.ReLU(inplace=True),
        # #     LinearFinal(32, self.n_action))
        # #############################################################


        # self.AT_get_logit_eff = nn.Sequential(  
        #     nn.Linear(15*self.check_n, 128), nn.ReLU(inplace=True),
        #     nn.Linear(128, 128), nn.ReLU(inplace=True),
        #     nn.Linear(128, 32), nn.ReLU(inplace=True),
        #     LinearFinal(32, self.n_action)
        # )
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

    def showoff(self, atten, obs):
        obs = torch.nan_to_num_(obs, 0)
        import numpy as np
        from config import GlobalConfig as cfg
        from UTIL.tensor_ops import _2tensor, _2cpu2numpy

        if cfg.matlab_logger is None:
            from VISUALIZE.mcom import mcom
            mcv = mcom(path='%s/v2d_logger/'%cfg.logdir,
                        digit=16, rapid_flush=True, draw_mode='Native')
            mcv.v2d_init()
            cfg.matlab_logger = mcv
        mcv = cfg.matlab_logger
        mcv.v2d_clear()
        def reverse_binary(binary, bit=8):
            assert bit==8
            mask_and_id = np.zeros_like(binary)[..., 0]
            for i in range(8):
                mask_and_id += binary[..., i]* 2**i
            return mask_and_id

        _atten = atten[0]
        _atten = _2cpu2numpy(_atten.squeeze())
        _obs = _2cpu2numpy(obs[0])
        _id = _obs[:,:,7:]
        __id = reverse_binary(_2cpu2numpy(_id))
        dead_hs = (_obs==0).all(-1) # 因为序号0 和空观测0 都会导致0， 
        __id = np.where(dead_hs, -1, __id)
        _pos = _obs[:,:,1:3]
        _atten_sort = np.argsort(-_atten, axis=-1)   # 需要用降序排列，先取倒数
        _atten_sort_top = _atten_sort[..., :3]
        obs_focus_ = np.take_along_axis(__id, axis=-1, indices=_atten_sort_top)

        share_id = __id.flatten()
        share_obs = my_view(_pos, [-1, 2])
        for i in range(len(share_id)):
            if share_id[i]==0 and (share_obs[i]==0).all(): continue
            if share_id[i]<0: continue
            if share_id[i] >=50:
                mcv.v2dx('cir|%d|r|0.04'%(share_id[i]), share_obs[i][0], share_obs[i][1])
            else:
                mcv.v2dx('cir|%d|g|0.04'%(share_id[i]), share_obs[i][0], share_obs[i][1])
        dead = (_atten==0).all(-1)
        for i in range(50): 
            if dead[i]: continue    
            for j in range(3):
                if obs_focus_[i,j]<0: continue
                mcv.v2d_line_object(i, obs_focus_[i,j])
        # for index, guard in enumerate(self.guards()):
        #     self.mcv.v2dx('cir|%d|b|0.04'%(index), guard.state.p_pos[0], guard.state.p_pos[1])
        #     if not guard.alive:
        #         self.mcv.v2dx('cir|%d|k|0.04'%(index), guard.state.p_pos[0], guard.state.p_pos[1])
        # for index, attacker in enumerate(self.attackers()):
        #     self.mcv.v2dx('cir|%d|r|0.04'%(index+50), attacker.state.p_pos[0], attacker.state.p_pos[1])
        #     if not attacker.alive:
        #         self.mcv.v2dx('cir|%d|k|0.04'%(index+50), attacker.state.p_pos[0], attacker.state.p_pos[1])
        mcv.v2d_show()
        time.sleep(0.5)
        return

    def _act(self, obs, test_mode, eval_mode=False, eval_actions=None):
        obs_raw = obs
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

        def get_self(x):
            assert x.dim()==4 or x.shape[2] == 12
            return x[:,:,:1]
        def get_other(x):
            assert x.dim()==4 or x.shape[2] == 12
            return x[:,:,1:]

        # ! <<< Critic network >>>
        ct_phi = self.CT_encode(obs)    # maybe here
        ct_phi_self = get_self(ct_phi)    ; ct_phi_self_mask = get_self(mask_dead)
        ct_phi_others = get_other(ct_phi)  ; ct_phi_others_mask = get_other(mask_dead)
        mask = (ct_phi_others_mask | ct_phi_self_mask).unsqueeze(-2)

        # assert (ct_phi_others_mask == torch.isnan(ct_phi_others).any(-1)).all()
        ct_enc, m_atten_softmax = self.CT_attention(q=ct_phi_self,k=ct_phi_others, mask=mask, return_attn=True)
        value = self.CT_getvalue(
            torch.cat((ct_phi_self.squeeze(-2), ct_enc.squeeze(-2)), -1)
        ) # self.CT_getvalue(ct_phi)#
        # self.showoff(m_atten_softmax, obs_raw[:,:,:12])


        ## !! detach gradient
        m_atten_softmax_detached = m_atten_softmax.squeeze(-2).detach()

 
        # ! <<< Actor network >>>
        '''
        attn_sort = torch.argsort(m_atten_softmax_detached, dim=-1, descending=True) #.cpu().numpy()
        attn_top = attn_sort[..., :self.top_n]
        atten_index_alias = repeat_at(attn_top, insert_dim=-1, n_times=obs.shape[-1])
        obs_focus_others = torch.gather(get_other(obs), dim=-2, index=atten_index_alias)
        obs_focus = torch.cat((get_self(obs), obs_focus_others), dim=-2)
        '''
         
        mask_dead = torch.isnan(obs).any(-1)
        self_maskdead = get_self(mask_dead)
        others_maskdead = get_other(mask_dead)
        mask = (self_maskdead | others_maskdead).unsqueeze(-2)

        enc = self.AT_encoder(obs)
        enc = self.AT_attention(q=get_self(enc), k=get_other(enc), mask=mask).squeeze(-2)
        logits = self.AT_get_logit_db(enc)

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
