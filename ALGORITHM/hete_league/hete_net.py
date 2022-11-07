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
        self.use_normalization = AlgorithmConfig.use_normalization

        self.n_tp = self.n_hete_types
        self.n_gp = self.hete_n_net_placeholder
        get_placeholder = lambda type, group: group*self.n_tp + type
        self.get_type_group = lambda ph: (ph%self.n_hete_types, ph//self.n_hete_types)

        addon_dim = 12 
        self._nets_flat_placeholder_ = torch.nn.ModuleList(modules=[
            Net(rawob_dim+addon_dim, n_action, **kwargs) for _ in range(
                self.n_tp * self.n_gp
            )  
        ])
        self._critic_central = NetCentralCritic(rawob_dim+addon_dim, n_action, **kwargs)
        self.nets = [[self._nets_flat_placeholder_[get_placeholder(type=tp, group=gp)] for tp in range(self.n_tp)] for gp in range(self.n_gp)]
        self.frontend_nets = self.nets[0]
        self.static_nets = self.nets[1:]
        for gp, n_arr in enumerate(self.nets): 
            for tp, n in enumerate(n_arr):
                index = get_placeholder(tp, gp)
                n.forbidden = False
                n.gp = gp
                n.tp = tp
                n.feature = np.zeros(4)
                if gp!=0: 
                    n.forbidden = True
                    self.lock_net(index)
                else:
                    self.unlock_net(index)
                n.hete_tag = 'group:%d,type:%d'%(gp, tp)

        self.ckpg_info = []
        self.map_ckpg_phg = {}
        self.ckpg_input_cnt = 0
        self.hete_feature_dim = 10
        self.ph_to_feature = torch.tensor(np.array([n.feature for n in self._nets_flat_placeholder_]), dtype=torch.float, device=cfg.device)
        
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

    # def re_calculate_nets_feature(self):
    #     drop_out_interval = 50
    #     down_dim = 10

    #     pca_data = []
    #     for d in self.ckpg_info: 
    #         for dd in d['model']:
    #             pca_data.append(torch.cat([v.flatten()[::drop_out_interval] for v in dd.values()]))
                
    #     pca_data = _2cpu2numpy(torch.stack(pca_data)) 
    #     downd_data = pca(pca_data, down_dim)
        
    #     downd_data = downd_data.reshape(len(self.ckpg_info), self.n_tp, down_dim)
        
    #     for i, d in enumerate(self.ckpg_info): 
    #         self.ckpg_info[i]['feature'] = [v for v in downd_data[i]]
        

    def register_ckp(self, win_rate, cpk_path, mean_reward):
        self.ckpg_input_cnt += 1
        prev_win_rate = [self.ckpg_info[i]['win_rate'] for i in range(len(self.ckpg_info))]
        if len(prev_win_rate)>0 and win_rate <= max(prev_win_rate): 
            return  # give up
        
        # net_feature = np.concatenate(([win_rate], np.zeros(10)))
        
        self.ckpg_info.append({
            'win_rate': win_rate, 
            'mean_reward': mean_reward,
            'ckpg_cnt': self.ckpg_input_cnt,
            'cpk_path': cpk_path,
            'model': copy.deepcopy([p.state_dict() for p in self.frontend_nets]),
            'feature': [
                np.array([0,  win_rate,  mean_reward,  self.ckpg_input_cnt/100]),
                np.array([1,  win_rate,  mean_reward,  self.ckpg_input_cnt/100]),
                np.array([2,  win_rate,  mean_reward,  self.ckpg_input_cnt/100]),
            ],
            })
        
        self.ckpg_info.sort(key=lambda x:x['win_rate'])
        self.trim_ckp()
        
        print('ckp register change!')
        print([self.ckpg_info[i]['win_rate'] for i in range(len(self.ckpg_info))])
        print([self.ckpg_info[i]['ckpg_cnt'] for i in range(len(self.ckpg_info))])
        
        # load parameters
        for i, static_nets in enumerate(self.static_nets):
            if i >= len(self.ckpg_info): continue
            for k, net in enumerate(static_nets):
                net.load_state_dict(self.ckpg_info[i]['model'][k], strict=True)
                assert static_nets[k].static
                static_nets[k].forbidden = False
                static_nets[k].feature = self.ckpg_info[i]['feature'][k]
                
        self.ph_to_feature = torch.tensor(np.array([n.feature for n in self._nets_flat_placeholder_]), dtype=torch.float, device=cfg.device)
        print('parameters reloaded')


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


    def random_select(self, rand_ops=None):
        
        
        # when random win rate is high, direct to frontend nets
        if np.random.rand() < AlgorithmConfig.hete_same_prob:
            # print亮绿('frontend return', 0)
            return 0
        
        # randomly select ckp
        if rand_ops is not None:
            # improve efficiency by limiting the number of active net
            rand_winrate = np.random.choice(rand_ops)
        else:
            rand_winrate = np.random.rand()
            
        # find nearest
        e_min = float('inf')
        e_min_index = -1 # default return 0
        for i, t in enumerate(self.ckpg_info):
            winrate = t['win_rate']
            e = abs(winrate-rand_winrate)
            if e < e_min:
                e_min = e
                e_min_index = i
        
        if e_min_index >= 0: # not empty
            # print亮绿('given', rand_winrate, 'return', e_min_index + 1)
            return e_min_index + 1
        else: # self.ckpg_info is empty
            # print亮绿('given', rand_winrate, 'return', 0)
            return 0

    # called after training update
    def on_update(self, update_cnt):
        return

    def redirect_to_frontend(self, i):
        return i%self.n_tp

    def acquire_net(self, i):
        return self._nets_flat_placeholder_[i]

    def exe(self, hete_pick=None, **kargs):
        # pop items from kargs
        n_thread = hete_pick.shape[0]
        n_agents = hete_pick.shape[1]
        
        gp_sel_summary, thread_indices, hete_type = popgetter('gp_sel_summary', 'thread_index', 'hete_type')(kargs)
        
        ph_sel = gp_sel_summary + repeat_at(torch.arange(self.n_tp, device=kargs['obs'].device, dtype=torch.int64), 0, gp_sel_summary.shape[0])
        ph_feature = self.ph_to_feature[ph_sel]  # my_view(, [0, -1])
        ph_feature_cp = repeat_at(ph_feature, 1, n_agents)
        
        # add the hete type info into obs
        ph_feature_cp_obs = scatter_righthand(
            scatter_into = ph_feature_cp,
            src = ph_feature_cp.new_ones(n_thread,n_agents,1, 4) * -1,
            index = hete_type.unsqueeze(-1), check=True
        )
        ph_feature_cp_obs = my_view(ph_feature_cp_obs, [0,0,-1])
        
        kargs['obs'] = cat_last_dim(kargs['obs'], ph_feature_cp_obs)
        
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

