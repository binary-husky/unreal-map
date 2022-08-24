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
from .net import Net, NetCentralCritic


    
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
        self._array_policy_group.append(
            copy.deepcopy([p.state_dict() for p in policy_group])
        )
        if len(self._array_policy_group)>self.n_rollbuffer_size:
            self.pop_policy_group()
            assert len(self._array_policy_group) == self.n_rollbuffer_size

    def __len__(self):
        return len(self._array_policy_group)

    def is_full(self):
        return len(self._array_policy_group) == self.n_rollbuffer_size

    def pop_policy_group(self):
        self._array_policy_group.pop(0)

    def random_link(self, static_groups):
        # 随机选择 n_gp - 1 个时间点
        ar = np.arange(len(self._array_policy_group))
        np.random.shuffle(ar)
        indices = ar[:len(static_groups)]
        # 加载参数
        for i, static_nets in enumerate(static_groups):
            pick = indices[i]
            for k, p in enumerate(self._array_policy_group[pick]):
                static_nets[k].load_state_dict(p, strict=True)
                assert static_nets[k].static
                assert not static_nets[k].forbidden

def _count_list_type(x):
    type_cnt = {}
    for xx in x:
        if xx not in type_cnt: type_cnt[xx] = 0
        type_cnt[xx] += 1
    return len(type_cnt)


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

    if ref is None:
        return
    elif ref == 'vph':
        pt[offset] = 'vph'
        return
    elif isinstance(ref, tuple) or isinstance(ref, list):
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
        
        with torch.no_grad() if fn.static else no_context() as gs:
            ret_tuple = fn._act(agent_ids=agent_ids, **_kwargs)

        dfs_create_and_fn(ret_tuple, g_out, 0, _create_tensor_ph_or_fill_, n_threads, n_agents, mask_flatten)

    dfs_create_and_fn(None, g_out, 0, _tensor_expand_thread_dim_v2_, n_threads, n_agents)
    return tuple(g_out[0])




class HeteNet(nn.Module):
    def __init__(self, rawob_dim, n_action, hete_type, **kwargs):
        super().__init__()
        self.rawob_dim = rawob_dim
        self.n_action = n_action
        self.hete_type = hete_type
        self.n_hete_types = _count_list_type(self.hete_type)
        self.n_online_policy_groups = AlgorithmConfig.n_online_policy_groups
        self.n_rollbuffer_size = AlgorithmConfig.rollbuffer_size
        self.pgrb = PolicyGroupRollBuffer(self.n_hete_types, self.n_rollbuffer_size)
        self.use_normalization = AlgorithmConfig.use_normalization

        n_tp = self.n_hete_types
        n_gp = self.n_online_policy_groups
        get_placeholder = lambda type, group: group*n_tp + type

        self._nets_flat_placeholder_ = torch.nn.ModuleList(modules=[
            Net(rawob_dim, n_action, **kwargs) for _ in range(
                n_tp * n_gp
            )  
        ])
        self._critic_central = NetCentralCritic(rawob_dim, n_action, **kwargs)
        self.nets = [[self._nets_flat_placeholder_[get_placeholder(type=tp, group=gp)] for tp in range(n_tp)] for gp in range(n_gp)]
        self.frontend_nets = self.nets[0]
        self.static_nets = self.nets[1:]
        for gp, n_arr in enumerate(self.nets): 
            for tp, n in enumerate(n_arr):
                index = get_placeholder(tp, gp)
                n.forbidden = False
                if gp!=0: 
                    self.lock_net(index)
                else:
                    self.unlock_net(index)
                n.hete_tag = 'group:%d,type:%d'%(gp, tp)



    def lock_net(self, i, forbidden=False):
        n = self._nets_flat_placeholder_[i]
        n.static = True
        n.eval()
        if forbidden:
            n.forbidden = True

    def unlock_net(self, i):
        n = self._nets_flat_placeholder_[i]
        n.static = False
        n.forbidden = False
        n.train()


    # called after training update
    def on_update(self):
        self.pgrb.push_policy_group( self.frontend_nets )
        if self.pgrb.is_full():
            self.pgrb.random_link(self.static_nets)
            

    def redirect_to_frontend(self, i):
        return i%self.n_hete_types

    def acquire_net(self, i):
        if self.pgrb.is_full():
            # print('buffer full mode')
            return self._nets_flat_placeholder_[i]
        else:
            # print('buffer empty mode')
            return self._nets_flat_placeholder_[self.redirect_to_frontend(i)]


    def act_lowlevel(self, hete_pick=None, **kargs):
        # if eval_mode:
        #     for net in self._nets_flat_placeholder_:
        #         print(__hashn__(net.parameters()))
        # hete_pick = hete_pick.cpu().apply_(self.redirect_ph)    # map policy selection
        # if len(hete_pick)==16:  # collecting samples, thread all alive
        #     if changed().check(hete_pick, key='hete_pick'): 
        #         print亮绿('redirected hete pick', hete_pick)
        invo_hete_types = [i for i in range(self.n_hete_types*self.n_hete_types) if (i in hete_pick)]
        # run self.net(input)
        actor_result = distribute_compute(
            fn_arr = [self.acquire_net(hete_type) for hete_type in invo_hete_types],
            # [self.acquire_net(hete_type)._act for hete_type in invo_hete_types],
            mask_arr = [(hete_pick == hete_type) for hete_type in invo_hete_types],
            **kargs
        )
        critic_result = self._critic_central.estimate_state(**kargs)
        
        actor_result = list(actor_result)
        for i, item in enumerate(actor_result):
            if item=='vph': 
                actor_result[i] = critic_result
        
        return tuple(actor_result)


    @Args2tensor_Return2numpy
    def act(self, **kargs):
        return self.act_lowlevel(**kargs)

    @Args2tensor
    def evaluate_actions(self, **kargs):
        return self.act_lowlevel(**kargs, eval_mode=True)


def tuple_ops():
    return

