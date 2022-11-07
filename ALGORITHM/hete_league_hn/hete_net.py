import torch, math, copy
import numpy as np
import torch.nn as nn
from config import GlobalConfig as cfg
from torch.distributions.categorical import Categorical
from UTIL.colorful import print亮绿
from UTIL.tensor_ops import Args2tensor_Return2numpy, Args2tensor, __hashn__, cat_last_dim, __hash__, repeat_at, scatter_righthand, gather_righthand, _2cpu2numpy, my_view
from .foundation import AlgorithmConfig
from ..commom.pca import pca
from ..commom.net_manifest import weights_init
from .net import Net, NetCentralCritic

def popgetter(*items):
    def g(obj): return tuple(obj.pop(item) if item in obj else None for item in items)
    return g

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
    """compute on each network

    Args:
        fn_arr : a list of forwarding networks
        mask_arr : mask of kwargs

    Returns:
        tuple tensors: the result of networks
    """
    # python don't have pointers, 
    # however, a list is a mutable type in python, that's all we need
    g_out = [None]  
    
    n_threads = mask_arr[0].shape[0]
    n_agents = mask_arr[0].shape[1]
    
    # calculated result will be gathered into ret_tuple_gather
    ret_tuple_gather = []
    
    # one by one we compute the result
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



class HeteNet(nn.Module):
    def __init__(self, rawob_dim, n_action, hete_type, **kwargs):
        super().__init__()
        self.rawob_dim = rawob_dim
        self.n_action = n_action
        self.hete_type = hete_type
        self.n_hete_types = _count_list_type(self.hete_type)
        self.hete_n_net_placeholder = AlgorithmConfig.hete_n_net_placeholder
        self.use_normalization = AlgorithmConfig.use_normalization

        self.n_tp = self.n_hete_types
        self.n_gp = self.hete_n_net_placeholder
        self.n_agent_each_tp = [sum(self.hete_type==i) for i in range(self.n_hete_types)]
        self.n_agents = len(self.hete_type)
        # convertion between placeholder index and type-group index
        self.tpgp_2_ph = lambda type, group: group*self.n_tp + type
        self.ph_2_tpgp = lambda ph: (ph%self.n_hete_types, ph//self.n_hete_types)

        # initialize net placeholders
        self._nets_flat_placeholder_ = torch.nn.ModuleList(modules=[
            Net(rawob_dim, n_action, **kwargs) for _ in range(
                self.n_tp * self.n_gp
            )
        ])
        # initialize critic
        self._critic_central = NetCentralCritic(rawob_dim, n_action, **kwargs)
        # reshape the handle of networks
        self.nets = [[self._nets_flat_placeholder_[self.tpgp_2_ph(type=tp, group=gp)] for tp in range(self.n_tp)] for gp in range(self.n_gp)]
        # the frontier nets
        self.frontend_nets = self.nets[0]
        # the static nets
        self.static_nets = self.nets[1:]
        # heterogeneous feature dimension
        self.hete_feature_dim = 1
        # add flags to each nets
        for gp, n_arr in enumerate(self.nets): 
            for tp, n in enumerate(n_arr):
                ph_index = self.tpgp_2_ph(tp, gp)
                n.gp = gp
                n.tp = tp
                n.lr_div = self.n_agent_each_tp[tp] / self.n_agents
                n.feature = np.zeros(self.hete_feature_dim)
                if gp!=0: 
                    # the frontier nets are ready
                    n.ready_to_go = False
                    self.lock_net(ph_index)
                else:
                    # the static nets are not loaded yet
                    n.ready_to_go = True
                    self.unlock_net(ph_index)
                    
        # a list to trace the vital checkpoints
        self.ckpg_info = []
        # track the number of checkpoints commited
        self.ckpg_input_cnt = 0
        # feature array, arranged according to placeholders
        self.ph_to_feature = torch.tensor(np.array([n.feature for n in self._nets_flat_placeholder_]), dtype=torch.float, device=cfg.device)
        # debug visually
        if AlgorithmConfig.debug:
            from VISUALIZE.mcom import mcom
            self.threejs_bridge = mcom(path='TEMP/v2d_logger/', rapid_flush=False, draw_mode='Threejs')
            self.threejs_bridge.v2d_init()
            self.threejs_bridge.set_style('font', fontPath='/examples/fonts/ttf/FZYTK.TTF', fontLineHeight=1500)
            self.threejs_bridge.geometry_rotate_scale_translate('box',   0, 0, 0,       1, 1, 1,         0, 0, 0)

    def lock_net(self, i):
        n = self._nets_flat_placeholder_[i]
        n.static = True
        n.eval()

    def unlock_net(self, i):
        n = self._nets_flat_placeholder_[i]
        n.static = False
        n.train()

    def register_ckp(self, win_rate, cpk_path, mean_reward):
        # deal with new checkpoint
        self.ckpg_input_cnt += 1
        # get previous win rates
        prev_win_rate = [self.ckpg_info[i]['win_rate'] for i in range(len(self.ckpg_info))]
        # if the winrate is not a breakthough, give up
        if len(prev_win_rate)>0 and win_rate <= max(prev_win_rate): 
            return
        
        # list the infomation about this checkpoint
        self.ckpg_info.append({
            'win_rate': win_rate, 
            'mean_reward': mean_reward,
            'ckpg_cnt': self.ckpg_input_cnt,
            'cpk_path': cpk_path,
            'model': copy.deepcopy([p.state_dict() for p in self.frontend_nets]),
            'feature': [
                np.array([win_rate]),
                np.array([win_rate]),
                np.array([win_rate]),
            ],
            })
        
        # sort according to win rate
        self.ckpg_info.sort(key=lambda x:x['win_rate'])
        # remove a checkpoint that is too close to its neighbor
        self.trim_ckp()
        
        print('ckp register change!')
        print([self.ckpg_info[i]['win_rate'] for i in range(len(self.ckpg_info))])
        print([self.ckpg_info[i]['ckpg_cnt'] for i in range(len(self.ckpg_info))])
        
        # reload parameters
        for i, static_nets in enumerate(self.static_nets):
            # some net cannot be loaded with parameters yet, because ckpg_info has not collect enough samples
            if i >= len(self.ckpg_info): continue
            for k, net in enumerate(static_nets):
                # load parameters
                net.load_state_dict(self.ckpg_info[i]['model'][k], strict=True)
                # the net must be static
                assert static_nets[k].static
                # now the net is ready
                static_nets[k].ready_to_go = True
                static_nets[k].feature = self.ckpg_info[i]['feature'][k]

        # reload the net features
        self.ph_to_feature = torch.tensor(np.array([n.feature for n in self._nets_flat_placeholder_]), dtype=torch.float, device=cfg.device)
        print('parameters reloaded')

    def random_select(self, *args, **kwargs):
        """randomly select a group index

        Args:
            AlgorithmConfig.hete_same_prob: a probability about choosing the frontier net as the teammate

        Returns:
            int: a group index
        """
        # redirect to frontier if so
        if np.random.rand() < AlgorithmConfig.hete_same_prob:
            return 0
        
        # choose randomly among existing nets
        n_option = len(self.ckpg_info)
        if n_option > 0:
            rand_sel = np.random.randint(low=1, high=n_option+1)
            return rand_sel
        else:
            return 0
    
    # def random_select(self, rand_ops=None):
    #     """randomly select a group index

    #     Args:
    #         AlgorithmConfig.hete_same_prob: a probability about choosing the frontier net as the teammate

    #     Returns:
    #         int: a group index
    #     """
    #     # when random win rate is high, direct to frontend nets
    #     if np.random.rand() < AlgorithmConfig.hete_same_prob:
    #         return 0
        
    #     # randomly select ckp
    #     if rand_ops is not None:
    #         # improve efficiency by limiting the number of active net
    #         rand_winrate = np.random.choice(rand_ops)
    #     else:
    #         rand_winrate = np.random.rand()
            
    #     # find nearest
    #     e_min = float('inf')
    #     e_min_index = -1 # default return 0
    #     for i, t in enumerate(self.ckpg_info):
    #         winrate = t['win_rate']
    #         e = abs(winrate-rand_winrate)
    #         if e < e_min:
    #             e_min = e
    #             e_min_index = i
        
    #     if e_min_index >= 0: # not empty
    #         # print亮绿('given', rand_winrate, 'return', e_min_index + 1)
    #         return e_min_index + 1
    #     else: # self.ckpg_info is empty
    #         # print亮绿('given', rand_winrate, 'return', 0)
    #         return 0

    # called after training update
    def on_update(self, update_cnt):
        return

    def redirect_to_frontend(self, i):
        return i%self.n_tp

    def acquire_net(self, i):
        return self._nets_flat_placeholder_[i]

    def debug_visual(self, hete_pick, n_thread, n_agents, thread_indices, running_nets):
        if n_thread > 8: return 
        
        for i, net in enumerate(self._nets_flat_placeholder_):
            update_cnt = net.update_cnt[0].item()
            self.threejs_bridge.v2dx(
                'box|%d|%s|0.2'%(-10-i, 'white' if not net.static else 'red' ),
                net.tp, net.gp, 10,
                label='G-%d, T-%d, U-%d'%(net.gp, net.tp, update_cnt),
                label_color='white',
            )
        for t in range(n_thread):
            thread_index = thread_indices[t].item()
            for agent_index in range(n_agents):
                hete_type = hete_pick[t, agent_index]
                
                net = self.acquire_net(hete_type)
                net_redirect = hete_type.item()
                self.threejs_bridge.v2dx(
                    'box|%d|%s|%.3f'%(thread_index*n_agents+agent_index, 
                                    'white' if not net.static else 'red', 
                                    0.2),
                    agent_index, thread_index, 1,
                    label='G-%d, U-%d'%(net.gp, net.update_cnt[0].item()),
                    label_color='white',
                )
                if net.static:
                    self.threejs_bridge.发射光束(
                        'beam',         # 有 beam 和 lightning 两种选择
                        src=thread_index*n_agents+agent_index,   # 发射者的几何体的唯一ID标识
                        dst=-10-net_redirect,  # 接收者的几何体的唯一ID标识
                        dur=0.5,        # 光束持续时间，单位秒，绝对时间，不受播放fps的影响
                        size=0.03,      # 光束粗细
                        color='DeepSkyBlue' # 光束颜色
                    )
                    

        self.threejs_bridge.v2d_show()

                
                
                
    def exe(self, hete_pick=None, **kargs):
        # shape
        n_thread = hete_pick.shape[0]
        n_agents = hete_pick.shape[1]
        
        # pop items from kargs
        gp_sel_summary, thread_indices, hete_type = popgetter('gp_sel_summary', 'thread_index', 'hete_type')(kargs)
        
        # get ph_feature
        _012345 = torch.arange(self.n_tp, device=kargs['obs'].device, dtype=torch.int64)
        ph_sel = gp_sel_summary*self.n_tp + repeat_at(_012345, 0, n_thread)   # group * self.n_tp + tp
        ph_feature = self.ph_to_feature[ph_sel]  # my_view(, [0, -1])
        ph_feature_cp = repeat_at(ph_feature, 1, n_agents)
        
        # reshape ph_feature
        ph_feature_cp_obs = scatter_righthand( # ph_feature_cp_obs.shape = torch.Size([n_thread=16, n_agents=10, n_tp=3, core_dim=4])
            scatter_into = ph_feature_cp,
            src = ph_feature_cp.new_ones(n_thread,n_agents, 1, self.hete_feature_dim) * -1,
            index = hete_type.unsqueeze(-1), check=True
        ) 
        ph_feature_cp_obs = my_view(ph_feature_cp_obs, [0,0,-1]) # ph_feature_cp_obs.shape = torch.Size([n_thread=16, n_agents=10, core_dim=12])
        
        # add ph_feature to kwargs
        kargs['obs_hfeature'] = ph_feature_cp_obs
        
        # get a manifest of running nets
        invo_hete_types = [i for i in range(self.n_tp*self.n_gp) if (i in hete_pick)]
        running_nets = [self.acquire_net(hete_type) for hete_type in invo_hete_types]
        
        # make sure all nets under testing is frontend / frontier
        if 'test_mode' in kargs and kargs['test_mode']: 
            for net in running_nets: assert not net.static
                
        # debug visual
        # if AlgorithmConfig.debug: self.debug_visual(hete_pick, n_thread, n_agents, thread_indices, running_nets)

        # run actor policy networks
        actor_result = distribute_compute(
            fn_arr = running_nets,
            mask_arr = [(hete_pick == hete_type) for hete_type in invo_hete_types],
            **kargs
        )
        
        # run critic network
        critic_result = self._critic_central.estimate_state(**kargs)
        
        # combine actor_result and critic_result
        actor_result = list(actor_result)
        for i, item in enumerate(actor_result):
            if item=='vph': actor_result[i] = critic_result
        
        # done !
        return tuple(actor_result)


    @Args2tensor_Return2numpy
    def act(self, **kargs):
        return self.exe(**kargs)

    @Args2tensor
    def evaluate_actions(self, **kargs):
        return self.exe(**kargs, eval_mode=True)

    def trim_ckp(self):
        RemoveNew = True
        max_static_gp = self.n_gp - 1
        if len(self.ckpg_info) <= max_static_gp:
            return
        else:
            assert len(self.ckpg_info) == max_static_gp+1
            # find two ckp with nearest 
            winrate_list = np.array([self.ckpg_info[i]['win_rate'] for i in range(len(self.ckpg_info))])
            winrate_list = np.abs(winrate_list[1:] - winrate_list[:-1])
            index = np.argmin(winrate_list)
            old_index = index
            new_index = index + 1
            if self.ckpg_info[new_index]['ckpg_cnt'] < self.ckpg_info[old_index]['ckpg_cnt']:
                new_index, old_index = old_index, new_index
            
            if RemoveNew:
                self.ckpg_info.pop(new_index)
            else:
                self.ckpg_info.pop(old_index)
            assert len(self.ckpg_info) == max_static_gp
                
        pass
    