import torch, math, copy
import numpy as np
import torch.nn as nn
from torch.distributions.categorical import Categorical
from UTIL.colorful import print亮绿
from UTIL.tensor_ops import Args2tensor_Return2numpy, Args2tensor, __hashn__, my_view
from UTIL.tensor_ops import pt_inf
from UTIL.exp_helper import changed
from .ccategorical import CCategorical
from .foundation import AlgorithmConfig
from ALGORITHM.common.attention import SimpleAttention
from ALGORITHM.common.norm import DynamicNormFix
from ALGORITHM.common.net_manifest import weights_init



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
        self.n_action = n_action
        
        
        if self.use_policy_resonance:
            self.ccategorical = CCategorical(kwargs['stage_planner'])
            self.is_resonance_active = lambda: kwargs['stage_planner'].is_resonance_active()

        h_dim = AlgorithmConfig.net_hdim

        # observation normalization
        if self.use_normalization:
            self._batch_norm = DynamicNormFix(rawob_dim, only_for_last_dim=True, exclude_one_hot=True, exclude_nan=True)

        n_entity = AlgorithmConfig.n_entity_placeholder
        
        # # # # # # # # # #  actor-critic share # # # # # # # # # # # #
        self.obs_encoder = nn.Sequential(nn.Linear(rawob_dim, h_dim), nn.ReLU(inplace=True), nn.Linear(h_dim, h_dim))
        self.attention_layer = SimpleAttention(h_dim=h_dim)
        # # # # # # # # # #        actor        # # # # # # # # # # # #
        _size = n_entity * h_dim
        self.policy_head = nn.Sequential(
            nn.Linear(_size, h_dim), nn.ReLU(inplace=True),
            nn.Linear(h_dim, h_dim//2), nn.ReLU(inplace=True),
            nn.Linear(h_dim//2, self.n_action))
        # # # # # # # # # # critic # # # # # # # # # # # #
        
        _size = n_entity * h_dim
        self.ct_encoder = nn.Sequential(nn.Linear(_size, h_dim), nn.ReLU(inplace=True), nn.Linear(h_dim, h_dim))
        self.ct_attention_layer = SimpleAttention(h_dim=h_dim)
        self.get_value = nn.Sequential(nn.Linear(h_dim, h_dim), nn.ReLU(inplace=True),nn.Linear(h_dim, 1))


        self.is_recurrent = False
        self.apply(weights_init)
        return
    
    @Args2tensor_Return2numpy
    def act(self, *args, **kargs):
        return self._act(*args, **kargs)
    
    @Args2tensor
    def evaluate_actions(self, *args, **kargs):
        return self._act(*args, **kargs, eval_mode=True)

    def _act(self, obs=None, test_mode=None, eval_mode=False, eval_actions=None, avail_act=None, agent_ids=None, eprsn=None):
        eval_act = eval_actions if eval_mode else None
        others = {}
        if self.use_normalization:
            if torch.isnan(obs).all(): pass
            else: obs = self._batch_norm(obs, freeze=(eval_mode or test_mode))

        mask_dead = torch.isnan(obs).any(-1)
        obs = torch.nan_to_num_(obs, 0)         # replace dead agents' obs, from NaN to 0
        
        # # # # # # # # # # actor-critic share # # # # # # # # # # # #
        baec = self.obs_encoder(obs)
        baec = self.attention_layer(k=baec,q=baec,v=baec, mask=mask_dead)

        # # # # # # # # # # actor # # # # # # # # # # # #
        at_bac = my_view(baec,[0,0,-1])
        logits = self.policy_head(at_bac)
        
        # choose action selector
        logit2act = self._logit2act_rsn if self.use_policy_resonance and self.is_resonance_active() else self._logit2act
        
        # apply action selector
        act, actLogProbs, distEntropy, probs = logit2act(   logits, 
                                                            eval_mode=eval_mode,
                                                            test_mode=test_mode, 
                                                            eval_actions=eval_act, 
                                                            avail_act=avail_act,
                                                            eprsn=eprsn)
        
        
        # # # # # # # # # # critic # # # # # # # # # # # #
        ct_bac = my_view(baec,[0,0,-1])
        ct_bac = self.ct_encoder(ct_bac)
        ct_bac = self.ct_attention_layer(k=ct_bac,q=ct_bac,v=ct_bac)
        value = self.get_value(ct_bac)
        
        if not eval_mode: return act, value, actLogProbs
        else:             return value, actLogProbs, distEntropy, probs, others

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
    
    

    
    