from attr import has
import torch, math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .ccategorical import CCategorical
from torch.distributions.categorical import Categorical
from torch.nn.modules.linear import Linear
from ..commom.norm import DynamicNorm
from UTIL.tensor_ops import Args2tensor_Return2numpy, Args2tensor, _2cpu2numpy, __hashn__
from UTIL.tensor_ops import pt_inf
from .foundation import AlgorithmConfig
from ..commom.conc import Concentration
from ..commom.net_manifest import weights_init


class PolicyGroupRollBuffer():
    def __init__(self, n_hete_types, n_rollbuffer_size) -> None:
        super().__init__()
        self.n_rollbuffer_size = n_rollbuffer_size
        self.n_hete_types = n_hete_types
        self._array_policy_group = []

    def push_policy_group(self, policy_group):
        self._array_policy_group.push(
            [p.state_dict() for p in policy_group]
        )
        if len(self._array_policy_group)>self.n_rollbuffer_size:
            self.pop_policy_group()
            assert len(self._array_policy_group) == self.n_rollbuffer_size

    def pop_policy_group(self):
        self._array_policy_group.pop(0)

    def link_policy_group(self, nets, n_sample_point):
        assert len(self._array_policy_group) == self.n_rollbuffer_size
        # [ ... ], [ ... ], [ ... ]
        batch = math.ceil(self.n_rollbuffer_size/n_sample_point)
        pick = []
        for i in range(n_sample_point):
            low = batch * i
            high = batch * (i + 1) if batch * (i + 1) <= self.n_rollbuffer_size else self.n_rollbuffer_size
            pick.append(np.random.randint(low=low, high=high))

        for i, p in enumerate(pick):
            nets[i].load_state_dict(self._array_policy_group[p], strict=True)

def _count_list_type(x):
    type_cnt = {}
    for xx in x:
        if xx not in type_cnt: type_cnt[xx] = 0
        type_cnt[xx] += 1
    return len(type_cnt)

def _list_type(x):
    type_cnt = {}
    for xx in x:
        if xx not in type_cnt: type_cnt[xx] = 0
        type_cnt[xx] += 1
    return type_cnt.keys()


class HeteNet(nn.Module):
    def __init__(self, rawob_dim, n_action, hete_type):
        super().__init__()
        self.rawob_dim = rawob_dim
        self.n_action = n_action
        self.hete_type = hete_type
        self.n_hete_types = _count_list_type(self.hete_type)
        self.n_policy_groups = AlgorithmConfig.n_policy_groups
        self.n_rollbuffer_size = 100
        self.pgrb = PolicyGroupRollBuffer(self.n_hete_types, self.n_rollbuffer_size)
        self.use_normalization = AlgorithmConfig.use_normalization

        self.frontend_nets = torch.nn.ModuleList(modules=[
            Net(rawob_dim, n_action) for _ in range(
                self.n_hete_types * 1
            )  
        ])
        self.static_nets = torch.nn.ModuleList(modules=[
            Net(rawob_dim, n_action) for _ in range(
                self.n_hete_types * (self.n_policy_groups - 1)
            )  
        ])

        
    def acquire_net(self, i):
        if i < self.n_hete_types:
            return self.frontend_nets[i]
        else:
            assert i < self.n_hete_types * self.n_policy_groups
            return self.static_nets[i-self.n_hete_types]


        # self.pgrb.push_policy_group( self.frontend_nets )
        # self.pgrb.link_policy_group( 
        #     nets = self.static_nets, 
        #     n_sample_point = self.n_types * (self.n_policy_group - 1)
        # )

    def access_net(self, nth_type, nth_policy_group):
        assert False
        index = nth_type * self.n_policy_group + nth_policy_group
        return self.nets[index]


    # def distribute_computation(bool_mask, fn, *args, **kargs):
        


    def act_ll(self, obs, test_mode, eval_mode=False, eval_actions=None, avail_act=None, hete_pick=None):
        eval_act = eval_actions if eval_mode else None
        invo_hete_types = _list_type(_2cpu2numpy(hete_pick.flatten()))
        g_out = []
        n_threads = obs.shape[0]
        n_agents = obs.shape[1]
        print(hete_pick)
        print('*********************')
        for hete_type in invo_hete_types:
            self.acquire_net(hete_type).eval()
            ret_tuple = self.acquire_net(hete_type).act(obs, test_mode, eval_mode, eval_actions, avail_act, agent_ids=None)
            print(ret_tuple[1].squeeze(-1))
        print('*********************')
        
        for hete_type in invo_hete_types:
            sel_bool = (hete_pick == hete_type)
            obs_tmp = obs[sel_bool] # shape (thread, agent, ...) --> (composed, ...)
            obs_tmp = obs_tmp.unsqueeze(0)# shape (composed, ...) --> (1, composed, ...)
            agent_ids = torch.where(sel_bool)[1]
            agent_ids = agent_ids.unsqueeze(0)
            ret_tuple = self.acquire_net(hete_type).act(obs_tmp, test_mode, eval_mode, eval_actions, avail_act, agent_ids)
            
            def register_g_out(i, x, sel_bool):
                if len(g_out) < (i+1):
                    g_out.append(None)
                    assert len(g_out)==(i+1)
                if g_out[i] is None:
                    g_out[i] = torch.zeros(size=(n_threads, n_agents, *x.shape[2:]), device=x.device, dtype=x.dtype)
                g_out[i][sel_bool] = x.squeeze(0)
            
            def deal_single(i, x):
                if hasattr(x, '__len__'):
                    register_g_out(i, x, sel_bool)
                else:
                    assert False
                return x
            
            if isinstance(ret_tuple, tuple):
                for i, ret in enumerate(ret_tuple):
                    deal_single(i, ret)
            else:
                deal_single(0, ret_tuple)
                
        print(g_out[1].squeeze(-1))

        return tuple(g_out)


    @Args2tensor_Return2numpy
    def act(self, *args, **kargs):
        return self.act_ll(*args, **kargs)

    @Args2tensor
    def evaluate_actions(self, *args, **kargs):
        return self.act_ll(*args, **kargs, eval_mode=True)




"""
    network initialize
"""
class Net(nn.Module):
    def __init__(self, rawob_dim, n_action):
        super().__init__()


        self.use_normalization = AlgorithmConfig.use_normalization
        self.n_focus_on = AlgorithmConfig.n_focus_on
        self.dual_conc = AlgorithmConfig.dual_conc
        self.n_entity_placeholder = AlgorithmConfig.n_entity_placeholder
        h_dim = AlgorithmConfig.net_hdim

        self.skip_connect = True
        self.n_action = n_action
        self.UseDivTree = AlgorithmConfig.UseDivTree

        # observation normalization
        if self.use_normalization:
            self._batch_norm = DynamicNorm(rawob_dim, only_for_last_dim=True, exclude_one_hot=True, exclude_nan=True)

        self.AT_obs_encoder = nn.Sequential(nn.Linear(rawob_dim, h_dim), nn.ReLU(inplace=True), nn.Linear(h_dim, h_dim))

        if self.dual_conc:
            self.MIX_conc_core_f = Concentration(
                            n_focus_on=self.n_focus_on-1, h_dim=h_dim, 
                            skip_connect=self.skip_connect, 
                            skip_connect_dim=rawob_dim, 
                            adopt_selfattn=False)
            self.MIX_conc_core_h = Concentration(
                            n_focus_on=self.n_focus_on, h_dim=h_dim, 
                            skip_connect=self.skip_connect, 
                            skip_connect_dim=rawob_dim, 
                            adopt_selfattn=False)
        else:
            self.MIX_conc_core = Concentration(
                            n_focus_on=self.n_focus_on, h_dim=h_dim, 
                            skip_connect=self.skip_connect, 
                            skip_connect_dim=rawob_dim, 
                            adopt_selfattn=False)

        tmp_dim = h_dim if not self.dual_conc else h_dim*2
        self.CT_get_value = nn.Sequential(Linear(tmp_dim, h_dim), nn.ReLU(inplace=True),Linear(h_dim, 1))
        self.CT_get_threat = nn.Sequential(Linear(tmp_dim, h_dim), nn.ReLU(inplace=True),Linear(h_dim, 1))


        # part
        if self.UseDivTree:
            self.AT_get_logit_db = nn.Sequential(
                nn.Linear(tmp_dim, h_dim), nn.ReLU(inplace=True),
                nn.Linear(h_dim, h_dim), nn.ReLU(inplace=True),
                nn.Linear(h_dim, h_dim))
            from .div_tree import DivTree
            self.AT_div_tree = DivTree(input_dim=h_dim, h_dim=h_dim, n_action=self.n_action)
        else:
            self.AT_get_logit_db = nn.Sequential(
                nn.Linear(tmp_dim, h_dim), nn.ReLU(inplace=True),
                nn.Linear(h_dim, h_dim//2), nn.ReLU(inplace=True),
                nn.Linear(h_dim//2, self.n_action))

        self.ccategorical = CCategorical()

        self.is_recurrent = False
        self.apply(weights_init)
        return

    # two ways to support avail_act, but which one is better?
    def logit2act_old(self, logits_agent_cluster, eval_mode, test_mode, eval_actions=None, avail_act=None):
        if avail_act is not None: logits_agent_cluster = torch.where(avail_act>0, logits_agent_cluster, -pt_inf())
        act_dist = Categorical(logits = logits_agent_cluster)
        if not test_mode:  act = act_dist.sample() if not eval_mode else eval_actions
        else:              act = torch.argmax(act_dist.probs, axis=2)
        actLogProbs = self._get_act_log_probs(act_dist, act) # the policy gradient loss will feedback from here
        # sum up the log prob of all agents
        distEntropy = act_dist.entropy().mean(-1) if eval_mode else None
        return act, actLogProbs, distEntropy, act_dist.probs


    def act(self, *args, **kargs):
        act = self._act if self.dual_conc else self._act_singlec
        return act(*args, **kargs)

    def evaluate_actions(self, *args, **kargs):
        act = self._act if self.dual_conc else self._act_singlec
        return act(*args, **kargs, eval_mode=True)

    # div entity for DualConc models, distincting friend or hostile (present or history)
    def div_entity(self, mat, type=[(0,), # self
                                        (1, 2, 3, 4, 5, 6, 7, 8, 9,10),  # current
                                    (11,12,13,14,15,16,17,18,19,20,21),  # history
                                    ],
                                    n=22):
        assert n == self.n_entity_placeholder
        if mat.shape[-2]==n:
            tmp = (mat[..., t, :] for t in type)
            assert mat.shape[-1]!=n
        elif mat.shape[-1]==n:
            tmp = (mat[..., t] for t in type)
            assert mat.shape[-2]!=n
        else:
            assert False, "please make sure that the number of entities is correct, should be %d"%mat.shape[-2]
        return tmp


    def _act(self, obs, test_mode, eval_mode=False, eval_actions=None, avail_act=None, agent_ids=None):
        eval_act = eval_actions if eval_mode else None
        others = {}
        if self.use_normalization:
            obs = self._batch_norm(obs)
        mask_dead = torch.isnan(obs).any(-1)    # find dead agents
        obs = torch.nan_to_num_(obs, 0)         # replace dead agents' obs, from NaN to 0
        v = self.AT_obs_encoder(obs)

        zs, ze_f, ze_h          = self.div_entity(obs,       n=self.n_entity_placeholder)
        vs, ve_f, ve_h          = self.div_entity(v,         n=self.n_entity_placeholder)
        _, ve_f_dead, ve_h_dead = self.div_entity(mask_dead, n=self.n_entity_placeholder)

        # concentration module
        vh_C, vh_M = self.MIX_conc_core_h(vs=vs, ve=ve_h, ve_dead=ve_h_dead, skip_connect_ze=ze_h, skip_connect_zs=zs)
        vf_C, vf_M = self.MIX_conc_core_f(vs=vs, ve=ve_f, ve_dead=ve_f_dead, skip_connect_ze=ze_f, skip_connect_zs=zs)

        # fuse forward path
        v_C_fuse = torch.cat((vf_C, vh_C), dim=-1)  # (vs + vs + check_n + check_n)
        if self.UseDivTree:
            pre_logits = self.AT_get_logit_db(v_C_fuse)
            logits, confact_info = self.AT_div_tree(pre_logits, agent_ids)   # ($thread, $agent, $coredim)
        else:
            logits = self.AT_get_logit_db(v_C_fuse)

        # motivation encoding fusion
        v_M_fuse = torch.cat((vf_M, vh_M), dim=-1)

        # motivation objectives
        value = self.CT_get_value(v_M_fuse)
        if eval_mode: threat = self.CT_get_threat(v_M_fuse)


        logit2act = self.logit2act if AlgorithmConfig.PR_ACTIVATE else self.logit2act_old
        act, actLogProbs, distEntropy, probs = logit2act(logits, eval_mode=eval_mode, 
                                                            test_mode=test_mode, eval_actions=eval_act, avail_act=avail_act)

        if eval_mode: others['threat'] = self.re_scale(threat, limit=12)

        if not eval_mode: return act, value, actLogProbs
        else:             return value, actLogProbs, distEntropy, probs, others


    @staticmethod
    def re_scale(t, limit):
        r = 1. /2. * limit
        return (torch.tanh_(t/r) + 1.) * r



    def logit2act(self, logits_agent_cluster, eval_mode, test_mode, eval_actions=None, avail_act=None):
        if avail_act is not None: logits_agent_cluster = torch.where(avail_act>0, logits_agent_cluster, -pt_inf())

        act_dist = self.ccategorical.feed_logits(logits_agent_cluster)

        if not test_mode:  
            act = self.ccategorical.sample(act_dist) if not eval_mode else eval_actions
        else:              
            act = torch.argmax(act_dist.probs, axis=2)

        # the policy gradient loss will feedback from here
        actLogProbs = self._get_act_log_probs(act_dist, act) 

        # sum up the log prob of all agents
        distEntropy = act_dist.entropy().mean(-1) if eval_mode else None
        
        return act, actLogProbs, distEntropy, act_dist.probs



    @staticmethod
    def _get_act_log_probs(distribution, action):
        return distribution.log_prob(action.squeeze(-1)).unsqueeze(-1)