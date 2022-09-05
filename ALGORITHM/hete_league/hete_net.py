import torch, math, copy
import numpy as np
import torch.nn as nn
from torch.distributions.categorical import Categorical
from UTIL.colorful import print亮绿
from UTIL.tensor_ops import Args2tensor_Return2numpy, Args2tensor, __hashn__, cat_last_dim, __hash__, repeat_at, scatter_righthand, gather_righthand
from UTIL.tensor_ops import pt_inf
from UTIL.exp_helper import changed
from .ccategorical import CCategorical
from .foundation import AlgorithmConfig
from ..commom.norm import DynamicNormFix
from ..commom.conc import Concentration
from ..commom.net_manifest import weights_init
from .net import Net, NetCentralCritic
from operator import methodcaller

    
class no_context():
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc_value, traceback):
        return False


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
    if ref is None: # there is nothing to sync, instead, do something at leaf node only
        ref = pt[offset]

    if ref == 'vph':
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
    
    # calculated result will be gathered into ret_tuple_gather
    ret_tuple_gather = []
    for fn, mask in zip(fn_arr, mask_arr):
        assert mask.dim()==2
        mask_flatten = mask.flatten()
        
        agent_ids = torch.where(mask)[1]
        agent_ids = agent_ids.unsqueeze(0) # fake an extral dimension
        _kwargs = {key:_deal_single_in(kwargs[key], mask_flatten) for key in kwargs}
        
        with torch.no_grad() if fn.static else no_context() as gs:  # no_grad is already declared outside in act mode
            ret_tuple = fn._act(agent_ids=agent_ids, **_kwargs)
            ret_tuple_gather.append(ret_tuple)
            
    # stack ret_tuple_gather into g_out
    for ret_tuple, fn, mask in zip(ret_tuple_gather, fn_arr, mask_arr):
        mask_flatten = mask.flatten()
        dfs_create_and_fn(ret_tuple, g_out, 0, _create_tensor_ph_or_fill_, n_threads, n_agents, mask_flatten)
        
    # reshape the tensor
    dfs_create_and_fn(None, g_out, 0, _tensor_expand_thread_dim_v2_, n_threads, n_agents)
    return tuple(g_out[0])

def popgetter(*items):
    def g(obj): return tuple(obj.pop(item) if item in obj else None for item in items)
    return g


class HeteNet(nn.Module):
    def __init__(self, rawob_dim, n_action, hete_type, **kwargs):
        super().__init__()
        self.rawob_dim = rawob_dim
        self.n_action = n_action
        self.hete_type = hete_type
        self.n_hete_types = _count_list_type(self.hete_type)
        self.hete_n_net_placeholder = AlgorithmConfig.hete_n_net_placeholder
        self.n_rollbuffer_size = AlgorithmConfig.hete_rollbuffer_size
        # self.pgrb = PolicyGroupRollBuffer(1, self.n_rollbuffer_size)
        self.use_normalization = AlgorithmConfig.use_normalization

        self.n_tp = self.n_hete_types
        self.n_gp = self.hete_n_net_placeholder
        get_placeholder = lambda type, group: group*self.n_tp + type
        self.get_type_group = lambda ph: (ph%self.n_hete_types, ph//self.n_hete_types)

        self._nets_flat_placeholder_ = torch.nn.ModuleList(modules=[
            Net(rawob_dim+self.n_tp+1, n_action, **kwargs) for _ in range(
                self.n_tp * self.n_gp
            )  
        ])
        self._critic_central = NetCentralCritic(rawob_dim+self.n_tp+1, n_action, **kwargs)
        self.nets = [[self._nets_flat_placeholder_[get_placeholder(type=tp, group=gp)] for tp in range(self.n_tp)] for gp in range(self.n_gp)]
        self.frontend_nets = self.nets[0]
        self.static_nets = self.nets[1:]
        for gp, n_arr in enumerate(self.nets): 
            for tp, n in enumerate(n_arr):
                index = get_placeholder(tp, gp)
                n.forbidden = False
                n.gp = gp
                n.tp = tp
                if gp!=0: 
                    self.lock_net(index)
                else:
                    self.unlock_net(index)
                n.hete_tag = 'group:%d,type:%d'%(gp, tp)
                
        if AlgorithmConfig.debug:
            from VISUALIZE.mcom import mcom
            self.threejs_bridge = mcom(path='TEMP/v2d_logger/', rapid_flush=False, draw_mode='Threejs')
            self.threejs_bridge.v2d_init()
            self.threejs_bridge.set_style('font', fontPath='/examples/fonts/ttf/FZYTK.TTF', fontLineHeight=1500)
            self.threejs_bridge.geometry_rotate_scale_translate('box',   0, 0, 0,       1, 1, 1,         0, 0, 0)
        
        self.ckp_info = []

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

    def register_ckp(self, win_rate, cpk_path):
        self.ckp_info.apppend({
            'win_rate': win_rate, 
            'cpk_path': cpk_path,
            'model': copy.deepcopy([p.state_dict() for p in self.frontend_nets]),
            'feature': None,
            })
        self.ckp_info.sort(key=lambda x:x['win_rate'])

    def random_select(self, exclude_frontend):
        rand_winrate = np.random.rand()
        for i in self.ckp_info:
            t = self.ckp_info[i]
            winrate = t['win_rate']
            e = abs(winrate-rand_winrate)
        

    # called after training update
    def on_update(self, update_cnt):
        return 
        # if update_cnt % AlgorithmConfig.hete_rollbuffer_interval==0:
        #     self.pgrb.push_policy_group(self.frontend_nets)
        # if not self.pgrb.is_empty():
        #     self.pgrb.random_link(self.static_nets)
        

    def redirect_to_frontend(self, i):
        return i%self.n_tp

    def acquire_net(self, i):
        if not self.pgrb.is_empty():
            return self._nets_flat_placeholder_[i]
        else:
            return self._nets_flat_placeholder_[self.redirect_to_frontend(i)]

    def acquire_net_redirect(self, i):
        if not self.pgrb.is_empty():
            return i
        else:
            return self.redirect_to_frontend(i)

    # def debug_visual(self, hete_pick, n_thread, n_agents, thread_indices):
    #     if n_thread <= 32:
    #         # # rollbuffer elements
    #         # for t, p_dict_L in enumerate(self.pgrb._array_policy_group):
    #         #     for tp, p_dict in enumerate(p_dict_L):
    #         #         update_cnt = p_dict['update_cnt'][0].item()
    #         #         self.threejs_bridge.v2dx(
    #         #             'box|%d|%s|%.3f'%(-1000-t*len(p_dict_L) - tp, 
    #         #                             'yellow', 
    #         #                             0.2),
    #         #             tp-10, t+10, 2,
    #         #             label='U-%d'%(update_cnt),
    #         #             label_color='white',
    #         #         )
            
    #         # net place holder
    #         for i, net in enumerate(self._nets_flat_placeholder_):
    #             update_cnt = net.update_cnt[0].item()
    #             self.threejs_bridge.v2dx(
    #                 'box|%d|%s|%.3f'%(-10-i, 
    #                                 'white' if not net.static else 'red', 
    #                                 0.2),
    #                 net.tp, net.gp, 10,
    #                 label='G-%d, T-%d, U-%d'%(net.gp, net.tp, update_cnt),
    #                 label_color='white',
    #             )
        
    #         # agents 
    #         for t in range(n_thread):
    #             thread_index = thread_indices[t].item()
    #             for agent_index in range(n_agents):
    #                 hete_type = hete_pick[t, agent_index]
                    
    #                 net = self.acquire_net(hete_type)
    #                 net_redirect = self.acquire_net_redirect(hete_type).item()
    #                 self.threejs_bridge.v2dx(
    #                     'box|%d|%s|%.3f'%(thread_index*n_agents+agent_index, 
    #                                     'white' if not net.static else 'red', 
    #                                     0.2),
    #                     agent_index, thread_index, 1,
    #                     label='G-%d, U-%d'%(net.gp, net.update_cnt[0].item()),
    #                     label_color='white',
    #                 )
    #                 if net.static:
    #                     self.threejs_bridge.发射光束(
    #                         'beam',         # 有 beam 和 lightning 两种选择
    #                         src=thread_index*n_agents+agent_index,   # 发射者的几何体的唯一ID标识
    #                         dst=-10-net_redirect,  # 接收者的几何体的唯一ID标识
    #                         dur=0.5,        # 光束持续时间，单位秒，绝对时间，不受播放fps的影响
    #                         size=0.03,      # 光束粗细
    #                         color='DeepSkyBlue' # 光束颜色
    #                     )
                    
    #         self.threejs_bridge.v2d_show()
    #     else:
    #         pass
    #         # print('too many threads')

    def exe(self, hete_pick=None, **kargs):
        # pop items from kargs
        n_thread = hete_pick.shape[0]
        n_agents = hete_pick.shape[1]
        

        gp_sel_summary, thread_indices, hete_type = popgetter('gp_sel_summary', 'thread_index', 'hete_type')(kargs)
        
        
        gp = repeat_at(gp_sel_summary, 1, n_agents)
        
        # add the hete type info into obs
        gp_sel_summary_obs = scatter_righthand(
            scatter_into = gp,
            src = hete_type.new_ones(n_thread,n_agents,1) * -1,
            index = hete_type.unsqueeze(-1), check=True
        )
        kargs['obs'] = cat_last_dim(kargs['obs'], hete_type.unsqueeze(-1))
        kargs['obs'] = cat_last_dim(kargs['obs'], gp_sel_summary_obs)
        
        invo_hete_types = [i for i in range(self.n_tp*self.n_gp) if (i in hete_pick)]
        running_nets = [self.acquire_net(hete_type) for hete_type in invo_hete_types]
        
        # make sure all nets under testing is frontend
        if 'test_mode' in kargs and kargs['test_mode']:
             for net in running_nets: assert not net.static
                
        # debug visual

        # if AlgorithmConfig.debug: self.debug_visual(hete_pick, n_thread, n_agents, thread_indices)

            
        # run actor policy networks
        actor_result = distribute_compute(
            fn_arr = running_nets,
            mask_arr = [(hete_pick == hete_type) for hete_type in invo_hete_types],
            **kargs
        )
        
        # run critic network
        critic_result = self._critic_central.estimate_state(**kargs)
        
        # replace value estimation
        actor_result = list(actor_result)
        for i, item in enumerate(actor_result):
            if item=='vph': actor_result[i] = critic_result
        
        return tuple(actor_result)


    @Args2tensor_Return2numpy
    def act(self, **kargs):
        return self.exe(**kargs)

    @Args2tensor
    def evaluate_actions(self, **kargs):
        return self.exe(**kargs, eval_mode=True)


def tuple_ops():
    return

