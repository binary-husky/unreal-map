import torch, math, copy
import numpy as np
import torch.nn as nn
from torch.distributions.categorical import Categorical
from UTIL.colorful import print亮绿
from UTIL.tensor_ops import Args2tensor_Return2numpy, Args2tensor, __hashn__
from UTIL.tensor_ops import pt_inf
from UTIL.exp_helper import changed
from .ccategorical import CCategorical
from .foundation import AlgorithmConfig
from ..commom.norm import DynamicNormFix
from ..commom.conc import Concentration
from ..commom.net_manifest import weights_init



"""
    network initialize
"""
class Net(nn.Module):
    def __init__(self, rawob_dim, n_action, **kwargs):
        super().__init__()
        self.update_cnt = nn.Parameter(
            torch.zeros(1, requires_grad=False, dtype=torch.long), requires_grad=False)
        self.use_normalization = AlgorithmConfig.use_normalization
        self.use_policy_resonance = AlgorithmConfig.policy_resonance
        self.n_focus_on = AlgorithmConfig.n_focus_on
        self.dual_conc = AlgorithmConfig.dual_conc
        self.n_entity_placeholder = AlgorithmConfig.n_entity_placeholder
        h_dim = AlgorithmConfig.net_hdim
        if self.use_policy_resonance:
            self.ccategorical = CCategorical(kwargs['stage_planner'])
            self.is_resonance_active = lambda: kwargs['stage_planner'].is_resonance_active()

        self.skip_connect = True
        self.n_action = n_action

        # observation normalization
        if self.use_normalization:
            self._batch_norm = DynamicNormFix(rawob_dim, only_for_last_dim=True, exclude_one_hot=True, exclude_nan=True)

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
        self.CT_get_value = nn.Sequential(nn.Linear(tmp_dim, h_dim), nn.ReLU(inplace=True),nn.Linear(h_dim, 1))
        self.CT_get_threat = nn.Sequential(nn.Linear(tmp_dim, h_dim), nn.ReLU(inplace=True),nn.Linear(h_dim, 1))

        self.AT_policy_head = nn.Sequential(
            nn.Linear(tmp_dim, h_dim), nn.ReLU(inplace=True),
            nn.Linear(h_dim, h_dim//2), nn.ReLU(inplace=True),
            nn.Linear(h_dim//2, self.n_action))

        self.is_recurrent = False
        self.apply(weights_init)
        return

    def act(self, *args, **kargs):
        act = self._act if self.dual_conc else self._act_singlec
        return act(*args, **kargs)

    def evaluate_actions(self, *args, **kargs):
        act = self._act if self.dual_conc else self._act_singlec
        return act(*args, **kargs, eval_mode=True)

    # div entity for DualConc models, distincting friend or hostile (present or history)
    def div_entity(self, mat, n=22, core_dim=None):
        assert n == self.n_entity_placeholder
        assert n == mat.shape[core_dim]
        type =  AlgorithmConfig.entity_distinct
        if core_dim == -2:
            tmp = (mat[..., t, :] for t in type)
        elif core_dim == -1:
            tmp = (mat[..., t] for t in type)
        else:
            assert False, "please make sure that the number of entities is correct, should be %d"%mat.shape[-2]
        return tmp

    def _act(self, obs=None, test_mode=None, eval_mode=False, eval_actions=None, avail_act=None, agent_ids=None, eprsn=None):
        assert not (self.forbidden)
        if self.static:
            assert self.gp >=1
        # if not test_mode: assert not self.forbidden
        eval_act = eval_actions if eval_mode else None
        others = {}
        if self.use_normalization:
            if torch.isnan(obs).all():
                pass # 某一种类型的智能体全体阵亡
            else:
                obs = self._batch_norm(obs, freeze=(eval_mode or test_mode or self.static))

        mask_dead = torch.isnan(obs).any(-1)    # find dead agents
        obs = torch.nan_to_num_(obs, 0)         # replace dead agents' obs, from NaN to 0
        v = self.AT_obs_encoder(obs)

        zs, ze_f, ze_h          = self.div_entity(obs,       n=self.n_entity_placeholder, core_dim=-2)
        vs, ve_f, ve_h          = self.div_entity(v,         n=self.n_entity_placeholder, core_dim=-2)
        _, ve_f_dead, ve_h_dead = self.div_entity(mask_dead, n=self.n_entity_placeholder, core_dim=-1)

        # concentration module
        vh_C, vh_M = self.MIX_conc_core_h(vs=vs, ve=ve_h, ve_dead=ve_h_dead, skip_connect_ze=ze_h, skip_connect_zs=zs)
        vf_C, vf_M = self.MIX_conc_core_f(vs=vs, ve=ve_f, ve_dead=ve_f_dead, skip_connect_ze=ze_f, skip_connect_zs=zs)

        # fuse forward path
        v_C_fuse = torch.cat((vf_C, vh_C), dim=-1)  # (vs + vs + check_n + check_n)
        logits = self.AT_policy_head(v_C_fuse)

        # motivation encoding fusion
        v_M_fuse = torch.cat((vf_M, vh_M), dim=-1)

        # motivation objectives
        if eval_mode: 
            threat = self.CT_get_threat(v_M_fuse)
            value = self.CT_get_value(v_M_fuse)
            others['threat'] = self.re_scale(threat, limit=12)
            others['value'] = value
            
        logit2act = self._logit2act
        if self.use_policy_resonance and self.is_resonance_active():
            logit2act = self._logit2act_rsn
            
        act, actLogProbs, distEntropy, probs = logit2act(   logits, eval_mode=eval_mode,
                                                            test_mode=(test_mode or self.static), 
                                                            eval_actions=eval_act, 
                                                            avail_act=avail_act,
                                                            eprsn=eprsn)



        if not eval_mode: return act, 'vph', actLogProbs
        else:             return 'vph', actLogProbs, distEntropy, probs, others

    @staticmethod
    def re_scale(t, limit):
        r = 1. /2. * limit
        return (torch.tanh_(t/r) + 1.) * r

    def _logit2act_rsn(self, logits_agent_cluster, eval_mode, test_mode, eval_actions=None, avail_act=None, eprsn=None):
        if avail_act is not None: logits_agent_cluster = torch.where(avail_act>0, logits_agent_cluster, -pt_inf())
        act_dist = self.ccategorical.feed_logits(logits_agent_cluster)
        
        if not test_mode: act = self.ccategorical.sample(act_dist, eprsn) if not eval_mode else eval_actions
        else:             act = torch.argmax(act_dist.probs, axis=2)
        # the policy gradient loss will feedback from here
        actLogProbs = self._get_act_log_probs(act_dist, act) 
        # sum up the log prob of all agents
        distEntropy = act_dist.entropy().mean(-1) if eval_mode else None
        return act, actLogProbs, distEntropy, act_dist.probs

    def _logit2act(self, logits_agent_cluster, eval_mode, test_mode, eval_actions=None, avail_act=None, **kwargs):
        if avail_act is not None: logits_agent_cluster = torch.where(avail_act>0, logits_agent_cluster, -pt_inf())
        act_dist = Categorical(logits = logits_agent_cluster)
        if not test_mode:  act = act_dist.sample() if not eval_mode else eval_actions
        else:              act = torch.argmax(act_dist.probs, axis=2)
        actLogProbs = self._get_act_log_probs(act_dist, act) # the policy gradient loss will feedback from here
        # sum up the log prob of all agents
        distEntropy = act_dist.entropy().mean(-1) if eval_mode else None
        return act, actLogProbs, distEntropy, act_dist.probs

    @staticmethod
    def _get_act_log_probs(distribution, action):
        return distribution.log_prob(action.squeeze(-1)).unsqueeze(-1)
    
    

    
    
    
"""
    network initialize
"""
class NetCentralCritic(nn.Module):
    def __init__(self, rawob_dim, n_action, **kwargs):
        super().__init__()

        self.use_normalization = AlgorithmConfig.use_normalization
        self.use_policy_resonance = AlgorithmConfig.policy_resonance
        self.n_focus_on = AlgorithmConfig.n_focus_on
        self.dual_conc = AlgorithmConfig.dual_conc
        self.n_entity_placeholder = AlgorithmConfig.n_entity_placeholder
        h_dim = AlgorithmConfig.net_hdim
        if self.use_policy_resonance:
            self.ccategorical = CCategorical(kwargs['stage_planner'])
            self.is_resonance_active = lambda: kwargs['stage_planner'].is_resonance_active()

        self.skip_connect = True
        self.n_action = n_action

        # observation normalization
        if self.use_normalization:
            self._batch_norm = DynamicNormFix(rawob_dim, only_for_last_dim=True, exclude_one_hot=True, exclude_nan=True)

        self.CT_obs_encoder = nn.Sequential(nn.Linear(rawob_dim, h_dim), nn.ReLU(inplace=True), nn.Linear(h_dim, h_dim))

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
        self.CT_get_value = nn.Sequential(nn.Linear(tmp_dim, h_dim), nn.ReLU(inplace=True),nn.Linear(h_dim, 1))
        self.CT_get_threat = nn.Sequential(nn.Linear(tmp_dim, h_dim), nn.ReLU(inplace=True),nn.Linear(h_dim, 1))


        self.is_recurrent = False
        self.apply(weights_init)
        return

    # div entity for DualConc models, distincting friend or hostile (present or history)
    def div_entity(self, mat, n=22, core_dim=None):
        assert n == self.n_entity_placeholder
        assert n == mat.shape[core_dim]
        type =  AlgorithmConfig.entity_distinct
        if core_dim == -2:
            tmp = (mat[..., t, :] for t in type)
        elif core_dim == -1:
            tmp = (mat[..., t] for t in type)
        else:
            assert False, "please make sure that the number of entities is correct, should be %d"%mat.shape[-2]
        return tmp


    def estimate_state(self, obs=None, test_mode=None, eval_mode=False, eval_actions=None, avail_act=None, agent_ids=None, eprsn=None):
        if self.use_normalization:
            if torch.isnan(obs).all():
                pass # 某一种类型的智能体全体阵亡
            else:
                obs = self._batch_norm(obs, freeze=(eval_mode or test_mode))

        mask_dead = torch.isnan(obs).any(-1)    # find dead agents
        obs = torch.nan_to_num_(obs, 0)         # replace dead agents' obs, from NaN to 0
        v = self.CT_obs_encoder(obs)

        zs, ze_f, ze_h          = self.div_entity(obs,       n=self.n_entity_placeholder, core_dim=-2)
        vs, ve_f, ve_h          = self.div_entity(v,         n=self.n_entity_placeholder, core_dim=-2)
        _, ve_f_dead, ve_h_dead = self.div_entity(mask_dead, n=self.n_entity_placeholder, core_dim=-1)

        # concentration module
        _, vh_M = self.MIX_conc_core_h(vs=vs, ve=ve_h, ve_dead=ve_h_dead, skip_connect_ze=ze_h, skip_connect_zs=zs)
        _, vf_M = self.MIX_conc_core_f(vs=vs, ve=ve_f, ve_dead=ve_f_dead, skip_connect_ze=ze_f, skip_connect_zs=zs)

        # motivation encoding fusion
        v_M_fuse = torch.cat((vf_M, vh_M), dim=-1)

        # motivation objectives
        value = self.CT_get_value(v_M_fuse)

        return value



