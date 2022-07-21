import torch, math
import numpy as np
import torch.nn as nn
from torch.distributions.categorical import Categorical
from UTIL.tensor_ops import Args2tensor_Return2numpy, Args2tensor, __hashn__
from UTIL.tensor_ops import pt_inf
from UTIL.data_struct import UniqueList
from .ccategorical import CCategorical
from .foundation import AlgorithmConfig
from ..commom.norm import DynamicNormFix
from ..commom.conc import Concentration
from ..commom.net_manifest import weights_init



    
class no_context():
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc_value, traceback):
        return False
    
class PolicyGroupRollBuffer():
    def __init__(self, n_hete_types, n_rollbuffer_size):
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





def _create_tensor_ph_or_fill_(ref, pt, offset, *args):
    n_threads, n_agents, mask = args
    if pt[offset] is None: 
        pt[offset] = torch.zeros(size=(n_threads*n_agents, *ref.shape[2:]), device=ref.device, dtype=ref.dtype)
    pt[offset][mask] = ref.squeeze(0)
    
def _tensor_expand_thread_dim_v2_(ref, pt, offset, *args):
    # undo dim collapse
    n_threads, n_agents = args
    v = pt[offset]
    pt[offset] = v.view(n_threads, n_agents, *v.shape[1:])



def dfs_create_and_fn(ref, pt, offset, fn, *args):
    '''
        ref: target to sync
        pt: mutable list
        offset: mutable list index
        fn: function to be executed at leaf nodes
        args: anything needed
    '''
    if ref is None:
        ref = pt[offset]
        
    if isinstance(ref, tuple) or isinstance(ref, list):
        if pt[offset] is None: pt[offset] = [None for item in ref]
        for i, item in enumerate(ref):
            dfs_create_and_fn(item, pt[offset], i, fn, *args)
    elif isinstance(ref, dict):
        if pt[offset] is None: pt[offset] = {key:None for key in ref}
        for key in ref:
            dfs_create_and_fn(ref[key], pt[offset], key, fn, *args)
    elif isinstance(ref, torch.Tensor):
        fn(ref, pt, offset, *args)
    else:
        assert False




def _deal_single_in(x, mask_flatten):
    if isinstance(x, torch.Tensor):
        # collapse first two dims
        return x.view(-1, *x.shape[2:])[mask_flatten].unsqueeze(0)
    else:
        return x

# todo: https://pytorch.org/tutorials/advanced/torch-script-parallelism.html?highlight=parallel
def distribute_compute(fn_arr, mask_arr, **kwargs):
    # python don't have pointers, 
    # however, a list is a mutable type in python, that's all we need
    g_out = [None]  
    
    n_threads = mask_arr[0].shape[0]
    n_agents = mask_arr[0].shape[1]
    
    for fn, mask in zip(fn_arr, mask_arr):
        assert mask.dim()==2
        mask_flatten = mask.flatten()
        
        agent_ids = torch.where(mask)[1]
        agent_ids = agent_ids.unsqueeze(0) # fake an extral dimension
       
        _kwargs = {key:_deal_single_in(kwargs[key], mask_flatten) for key in kwargs}
        
        # def _act(self, obs, test_mode, eval_mode=False, eval_actions=None, avail_act=None, agent_ids=None):
        with torch.no_grad() if fn.static else no_context() as gs:
            ret_tuple = fn._act(agent_ids=agent_ids, **_kwargs)
            
        dfs_create_and_fn(ret_tuple, g_out, 0, _create_tensor_ph_or_fill_, n_threads, n_agents, mask_flatten)

    dfs_create_and_fn(None, g_out, 0, _tensor_expand_thread_dim_v2_, n_threads, n_agents)
    return tuple(g_out[0])




class HeteNet(nn.Module):
    def __init__(self, rawob_dim, n_action, hete_type):
        super().__init__()
        self.rawob_dim = rawob_dim
        self.n_action = n_action
        self.hete_type = hete_type
        self.n_hete_types = _count_list_type(self.hete_type)
        self.n_policy_groups = AlgorithmConfig.n_policy_groups
        self.n_rollbuffer_size = 100
        # self.pgrb = PolicyGroupRollBuffer(self.n_hete_types, self.n_rollbuffer_size)
        self.use_normalization = AlgorithmConfig.use_normalization

        n_tp = self.n_hete_types
        n_gp = self.n_policy_groups
        get_placeholder = lambda type, group: group*n_tp + type
        get_type_group = lambda ph: (ph%n_tp, ph//n_tp)
        
        self._nets_flat_placeholder_ = torch.nn.ModuleList(modules=[
            Net(rawob_dim, n_action) for _ in range(
                n_tp * n_gp
            )  
        ])
        
        self.nets = [[self._nets_flat_placeholder_[get_placeholder(type=tp, group=gp)] for tp in range(n_tp)] for gp in range(n_gp)]
        for gp, n_arr in enumerate(self.nets): 
            for tp, n in enumerate(n_arr):
                n.hete_valid = True                                     # hete_valid 指历史节点已经就绪
                n.static = not AlgorithmConfig.hete_type_trainable[tp]  # static 即静态网络，不可以训练，torch.no_grad()
                if gp!=0: n.hete_valid = False
                if gp!=0: n.static = True
                if n.static:
                    n.eval()
                n.hete_tag = 'group:%d,type:%d,valid:%s,training:%s'%(gp, tp, str(n.hete_valid), str(n.training))

        print('')

        # self.pgrb.push_policy_group( self.frontend_nets )
        # self.pgrb.link_policy_group( 
        #     nets = self.static_nets, 
        #     n_sample_point = self.n_types * (self.n_policy_group - 1)
        # )

    def redirect_ph(self, i):
        n_tp = self.n_hete_types
        n_gp = self.n_policy_groups
        # get_placeholder = lambda type, group: group*n_tp + type
        
        # <get type and group>
        # get_type_group = lambda ph: (ph%n_tp, ph//n_tp)
        # type, group = get_type_group(i)
        
        type, group = (i%n_tp, i//n_tp)
        assert group < n_gp
        # net = self._nets_flat_placeholder_[get_placeholder(type=type, group=group)]
        net = self._nets_flat_placeholder_[group*n_tp + type]

        if group==0:
            return group*n_tp + type
            # return get_placeholder(type=type, group=group)
        else:
            if net.hete_valid:
                return group*n_tp + type
                # return get_placeholder(type=type, group=group)
            else:
                return 0*n_tp + type
                # return get_placeholder(type=type, group=0)


    def acquire_net(self, i):
        return self._nets_flat_placeholder_[self.redirect_ph(i)]
        

    def act_lowlevel(self, hete_pick=None, **kargs):
        # if eval_mode:
        #     for net in self._nets_flat_placeholder_:
        #         print(__hashn__(net.parameters()))
        hete_pick = hete_pick.cpu().apply_(self.redirect_ph)    # map policy selection
        invo_hete_types = [i for i in range(self.n_hete_types*self.n_hete_types) if (i in hete_pick)]
        
        return distribute_compute(
            fn_arr = [self.acquire_net(hete_type) for hete_type in invo_hete_types],
            # [self.acquire_net(hete_type)._act for hete_type in invo_hete_types],
            mask_arr = [(hete_pick == hete_type) for hete_type in invo_hete_types],
            **kargs
        )


    @Args2tensor_Return2numpy
    def act(self, **kargs):
        return self.act_lowlevel(**kargs)

    @Args2tensor
    def evaluate_actions(self, **kargs):
        return self.act_lowlevel(**kargs, eval_mode=True)




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
        if avail_act is not None: 
            logits_agent_cluster = torch.where(avail_act>0, logits_agent_cluster, -pt_inf())
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


    def _act(self, obs=None, test_mode=None, eval_mode=False, eval_actions=None, avail_act=None, agent_ids=None):

        eval_act = eval_actions if eval_mode else None
        others = {}
        if self.use_normalization:
            if torch.isnan(obs).all():
                pass # 某一种类型的智能体全体阵亡
            else:
                obs = self._batch_norm(obs)
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
        logits = self.AT_get_logit_db(v_C_fuse)

        # motivation encoding fusion
        v_M_fuse = torch.cat((vf_M, vh_M), dim=-1)

        # motivation objectives
        value = self.CT_get_value(v_M_fuse)
        if eval_mode: threat = self.CT_get_threat(v_M_fuse)


        logit2act = self.logit2act_old
        act, actLogProbs, distEntropy, probs = logit2act(logits, eval_mode=eval_mode,
                                                            test_mode=test_mode, eval_actions=eval_act, avail_act=avail_act)

        if eval_mode: others['threat'] = self.re_scale(threat, limit=12)

        if not eval_mode: return act, value, actLogProbs
        else:             return value, actLogProbs, distEntropy, probs, others


    @staticmethod
    def re_scale(t, limit):
        r = 1. /2. * limit
        return (torch.tanh_(t/r) + 1.) * r



    # def logit2act(self, logits_agent_cluster, eval_mode, test_mode, eval_actions=None, avail_act=None):
    #     if avail_act is not None: logits_agent_cluster = torch.where(avail_act>0, logits_agent_cluster, -pt_inf())

    #     act_dist = self.ccategorical.feed_logits(logits_agent_cluster)

    #     if not test_mode:  
    #         act = self.ccategorical.sample(act_dist) if not eval_mode else eval_actions
    #     else:              
    #         act = torch.argmax(act_dist.probs, axis=2)

    #     # the policy gradient loss will feedback from here
    #     actLogProbs = self._get_act_log_probs(act_dist, act) 

    #     # sum up the log prob of all agents
    #     distEntropy = act_dist.entropy().mean(-1) if eval_mode else None
        
    #     return act, actLogProbs, distEntropy, act_dist.probs



    @staticmethod
    def _get_act_log_probs(distribution, action):
        return distribution.log_prob(action.squeeze(-1)).unsqueeze(-1)