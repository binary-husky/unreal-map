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
from UTILS.tensor_ops import my_view, Args2tensor_Return2numpy, Args2tensor, __hash__, repeat_at, _2cpu2numpy, one_hot_with_nan, gather_righthand


def weights_init(m):
    def init_Linear(m, final_layer=False):
        nn.init.orthogonal_(m.weight.data)
        if final_layer:nn.init.orthogonal_(m.weight.data, gain=0.01)
        if m.bias is not None: nn.init.uniform_(m.bias.data, a=-0.02, b=0.02)

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

class Extraction_Module(nn.Module): # merge by MLP version
    def __init__(self, hidden_dim=128, activate_output=False):
        super().__init__()
        self.activation_func = nn.ReLU
        h_dim = hidden_dim
        self.attn = MultiHeadAttention(n_heads=1, input_dim=h_dim, embed_dim=h_dim)
        if activate_output:
            self.MLP = nn.Sequential(nn.Linear(h_dim * 2, h_dim), self.activation_func(inplace=True), 
                nn.Linear(h_dim, h_dim), self.activation_func(inplace=True))
            print("activate_output")
        else:
            self.MLP = nn.Sequential(nn.Linear(h_dim * 2, h_dim), self.activation_func(inplace=True), 
                nn.Linear(h_dim, h_dim))
            print("no activate_output")

    # input: entity feature vector
    # output: entity feature vector
    def forward(self, agent_enc):
        # print(agent_enc.shape)
        attn_out = self.attn(q=agent_enc, k=agent_enc, v=agent_enc)
        # print(attn_out.shape)
        concated_attn_result = torch.cat(tensors=(agent_enc, attn_out), dim=-1)
        # print(concated_attn_result.shape)
        return self.MLP(concated_attn_result)




"""
    network initialize
"""
class Net(nn.Module):
    def __init__(self, num_agents, n_basic_dim, obs_dim, n_action, K_EMod=2, hidden_dim=None, 
                    use_m_gpu=None, use_normalization=True, seperate_critic=False, n_focus_on=None, actor_attn_mod=False):
        super().__init__()
        n_agent = num_agents
        self.n_action = n_action
        self.n_agent = n_agent
        basic_vec_len = n_basic_dim
        lkrelu = nn.ReLU
        self.K_EMod = K_EMod
        self.n_focus_on = n_focus_on
        # part0
        self.use_normalization = use_normalization
        if self.use_normalization:
            self._batch_norm = DynamicNorm(obs_dim, only_for_last_dim=True, exclude_one_hot=True, exclude_nan=True)

        # # part1
        # self.AE_encode = ResLinear(io_dim=8, h_dim=32, need_input_tf=True, input_tf_dim=n_basic_dim)
        # self.AE_decode = nn.Sequential(
        #                     ResLinear(io_dim=basic_vec_len, h_dim=32, need_input_tf=True, input_tf_dim=8),
        #                     nn.Linear(basic_vec_len, basic_vec_len))

        h_dim = 48

        # part2
        self.AT_encode = nn.Sequential(
                            nn.Linear(n_basic_dim, h_dim), lkrelu(inplace=True),
                            nn.Linear(h_dim, h_dim))
        self.CT_friend_attention = MultiHeadAttention(n_heads=1, input_dim=h_dim, embed_dim=h_dim)
        self.CT_hostile_attention = MultiHeadAttention(n_heads=1, input_dim=h_dim, embed_dim=h_dim)
        self.CT_get_value = nn.Sequential(
            Linear(h_dim*3, h_dim), lkrelu(inplace=True),
            Linear(h_dim, 1),
        )
        self.CT_get_threat = nn.Sequential(
            Linear(h_dim*3, h_dim), lkrelu(inplace=True),
            Linear(h_dim, 1),
        )
        self.CT_est_friend_threat = nn.Sequential(
            Linear(h_dim, h_dim//2), lkrelu(inplace=True),
            Linear(h_dim//2, 1),
        )
        # part3
        self.apply_focus = True
        self.check_n = 12 if not self.apply_focus else self.n_focus_on*2
        self.check_who = range(12)

        self.AT_get_logit_db = nn.Sequential(  
            nn.Linear((h_dim+n_basic_dim)*self.check_n, h_dim), nn.ReLU(inplace=True),
            nn.Linear(h_dim, h_dim//2), nn.ReLU(inplace=True),
            LinearFinal(h_dim//2, self.n_action))

        self.actor_attn_mod = actor_attn_mod
        if self.actor_attn_mod:
            print('警告！使用actor attention')
            self.AT_Attention = Extraction_Module(hidden_dim=(h_dim+n_basic_dim), activate_output=True)

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

    # obs[torch.isnan(my_view(obs, [0,0, -1])).all(-1)]
    def _act(self, obs, test_mode, eval_mode=False, eval_actions=None):
        eval_act = eval_actions if eval_mode else None
        others = {}
        # if obs.dim()==3:
        #     print('处理过滤')
        # nan2zero
        obs_raw = obs
        if self.use_normalization:
            obs = self._batch_norm(obs)
        obs = obs[...,:12,:]   #倒数第二维 # obs use torch.nan as placeholder !
        mask_dead = torch.isnan(obs).any(-1)
        obs = torch.nan_to_num_(obs, 0)


        def get_self(x):
            if x.dtype == torch.bool: 
                assert x.shape[-1] == 12
                return x[...,:1]
            assert x.shape[-2] == 12
            return x[...,:1,:] 
        def get_friend(x):
            # assert x.dim()==4 or x.shape[2] == 12
            if x.dtype == torch.bool: 
                assert x.shape[-1] == 12
                return x[...,1:6]
            assert x.shape[-2] == 12
            return x[...,1:6,:]
        def get_hostile(x):
            # assert x.dim()==4 or x.shape[2] == 12
            if x.dtype == torch.bool: 
                assert x.shape[-1] == 12
                return x[...,6:12]
            assert x.shape[-2] == 12
            return x[...,6:12,:]


        # ! <<< Critic network >>> 
        # ct_phi = self.CT_encode(obs)
        obs_self = get_self(obs)
        # obs = obs - obs_self
        ct_phi = self.AT_encode(obs)    # __hash__(ct_phi) = '1ef4cf80c75f3a4d68a3c33c83059be5'
        def get_friend_enc(ct_phi, mask_dead):
            ct_phi_self = get_self(ct_phi)
            ct_phi_friend = get_friend(ct_phi)
            ct_phi_friend_mask = get_friend(mask_dead)
            mask = ct_phi_friend_mask.unsqueeze(-2)
            phi_friend_enc, atten_friend = self.CT_friend_attention(q=ct_phi_self, k=ct_phi_friend, mask=mask, return_attn=True)
            return phi_friend_enc.squeeze(-2), atten_friend.squeeze(-2).detach()

        def get_hostile_enc(ct_phi, mask_dead):
            ct_phi_self = get_self(ct_phi)
            ct_phi_hostile = get_hostile(ct_phi)
            ct_phi_hostile_mask = get_hostile(mask_dead)
            mask = ct_phi_hostile_mask.unsqueeze(-2)
            phi_hostile_enc, atten_hostile = self.CT_hostile_attention(q=ct_phi_self, k=ct_phi_hostile, mask=mask, return_attn=True)
            return phi_hostile_enc.squeeze(-2), atten_hostile.squeeze(-2).detach()

        f_enc, f_attn_matrix = get_friend_enc(ct_phi, mask_dead)
        h_enc, h_attn_matrix = get_hostile_enc(ct_phi, mask_dead)
        # self.showoff(h_attn_matrix, obs_raw[:,:,:12], attn_offset=+6)
        ct_phi_self = get_self(ct_phi).squeeze(-2)
        # merge attn result
        phi_all = torch.cat((ct_phi_self, f_enc, h_enc), -1)    #__hash__(phi_all) 'a613051af9ccc851bf8ed72d342f8d13'
        value = self.CT_get_value(phi_all)  # 'e9a139681cb5569e6d49f0b52a6aa4a4'

        # estimate self threat
        threat = self.CT_get_threat(phi_all) # 'fe9ae2f8c9eeb3df4904471ed04e4aa7'
        SAFE_LIMIT = 11
        r = 1. /2. * SAFE_LIMIT
        threat = (torch.tanh_(threat/r) + 1.) * r
        others['threat'] = threat

        # produce friend's threat estimation
        # ct_phi_friend = get_friend(ct_phi)
        # friend_id = get_friend_id(obs_raw)
        # phi_rep = repeat_at(phi_all, insert_dim=-2, n_times=5)
        # phi_friend = torch.cat((ct_phi_friend, phi_rep), -1)    # 16+48
        # friend_threat = self.CT_est_friend_threat(phi_friend)*5 # 50,5
        # others['friend_threat'] = my_view(friend_threat,[0,-1])
        # others['friend_threat_id_with_nan'] = my_view(friend_id,[0,-1])

        at_phi = ct_phi #.detach()
        obs_ = torch.cat((at_phi, obs), -1)


        # actor part
        if self.apply_focus:
            # ! <<< Actor network >>>
            attn_sort_f = torch.argsort(f_attn_matrix, dim=-1, descending=True)     # torch.Size([64, 50, 5, ...])
            attn_sort_h = torch.argsort(h_attn_matrix, dim=-1, descending=True)     # torch.Size([64, 50, 5, ...])
            attn_top_f = attn_sort_f[..., :self.n_focus_on-1]   # top 2 attention friend, torch.Size([64, 50, 2, ...])
            attn_top_h = attn_sort_h[..., :self.n_focus_on]    # top 3 attention hostile, torch.Size([64, 50, 3, ...])

            obs_focus_others_f = gather_righthand(src=get_friend(obs_),  index = attn_top_f, check=False)
            obs_focus_others_h = gather_righthand(src=get_hostile(obs_),  index = attn_top_h, check=False)
            # __hash__(obs_focus) = 'da385436fa62f26677b5786cfd065939'
            obs_focus = torch.cat((get_self(obs_), obs_focus_others_f, obs_focus_others_h), dim=-2)
            if self.actor_attn_mod:
                obs_focus = self.AT_Attention(obs_focus)
            flat = my_view(obs_focus, [0, 0, -1]) if obs_focus.dim()==4 else my_view(obs_focus, [0, -1]) # 'da385436fa62f26677b5786cfd065939'
            logits = self.AT_get_logit_db(flat) # '07a8400581e7873425b11c294cab2081'
        else:
            assert False
            logits = self.AT_get_logit_db(my_view(obs_, [0, 0, -1]) )

        act, actLogProbs, distEntropy, probs = \
                    self.logit2act(logits, eval_mode=eval_mode, test_mode=test_mode, eval_actions=eval_act)

        if not eval_mode:
            return act, value, actLogProbs
        else:
            return value, actLogProbs, distEntropy, probs, others

        
        '''
        # seperate self obs and other obs
        ct_phi_self = get_self(ct_phi)
        ct_phi_others = get_other(ct_phi)

        # get vis mask for attn
        ct_phi_self_mask = get_self(mask_dead)
        ct_phi_others_mask = get_other(mask_dead)
        mask = (ct_phi_others_mask | ct_phi_self_mask).unsqueeze(-2) # 
        assert (ct_phi_others_mask == torch.isnan(ct_phi_others).any(-1)).all() # 这里涉及到本体dead的情况下，对于其他的空位是如何pad的，如果正确的情况下，应当全部观测不到
        phi_h_enc, m_atten_softmax = self.CT_attention(q=ct_phi_self, k=ct_phi_others, mask=mask, return_attn=True)
        value = self.CT_getvalue(
            torch.cat((ct_phi_self.squeeze(-2), phi_h_enc.squeeze(-2)), -1)
        )
        # self.showoff(m_atten_softmax, obs_raw[:,:,:12])


        ## !! detach gradient
        m_atten_softmax_detached = m_atten_softmax.squeeze(-2).detach()
        
        '''
 
        # ! <<< Actor network >>>
        '''
        attn_sort_f = torch.argsort(m_atten_softmax_detached[..., :5], dim=-1, descending=True) 
        attn_sort_h = torch.argsort(m_atten_softmax_detached[..., -6:], dim=-1, descending=True) 
        attn_top_f = attn_sort_f[..., :3]
        attn_top_h = attn_sort_h[..., :4]
        atten_index_alias_f = repeat_at(attn_top_f, insert_dim=-1, n_times=obs.shape[-1])
        atten_index_alias_h = repeat_at(attn_top_h, insert_dim=-1, n_times=obs.shape[-1])
        obs_focus_others_f = torch.gather(get_other(obs), dim=-2, index=atten_index_alias_f)
        obs_focus_others_h = torch.gather(get_other(obs), dim=-2, index=atten_index_alias_h)
        obs_focus = torch.cat((get_self(obs), obs_focus_others_f, obs_focus_others_h), dim=-2)
        '''
        ## 


    def showoff(self, atten, obs, attn_offset=0, top_n=1):
        obs = torch.nan_to_num_(obs, 0)
        import numpy as np
        from config import GlobalConfig as cfg
        from UTILS.tensor_ops import _2tensor, _2cpu2numpy

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
        _atten_sort_top = _atten_sort[..., :top_n] + attn_offset
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
            for j in range(top_n):
                if obs_focus_[i,j]<0: continue
                mcv.v2d_line_object(i, obs_focus_[i,j])

        mcv.v2d_show()
        time.sleep(0.5)
        return

