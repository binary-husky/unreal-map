import torch
import torch.nn as nn
import numpy as np
from ..commom.mlp import LinearFinal
from UTILS.tensor_ops import add_onehot_id_at_last_dim, repeat_at, _2tensor, gather_righthand, scatter_righthand


    
class DivTree(nn.Module): # merge by MLP version
    def __init__(self, input_dim, h_dim, n_action):
        super().__init__()

        # to design a division tree, I need to get the total number of agents
        from .foundation import AlgorithmConfig
        self.n_agent = AlgorithmConfig.n_agent
        self.div_tree = get_division_tree(self.n_agent)
        self.n_level = len(self.div_tree)
        self.max_level = len(self.div_tree) - 1
        self.current_level = 0
        self.init_level = AlgorithmConfig.div_tree_init_level
        self.current_level_floating = 0.0



        get_net = lambda: nn.Sequential(
            nn.Linear(h_dim+self.n_agent, h_dim), 
            nn.ReLU(inplace=True),
            LinearFinal(h_dim, n_action)
        )
        # Note: this is NOT net defining for each agent
        # Instead, all agents starts from self.nets[0]
        self.nets = torch.nn.ModuleList(modules=[
            get_net() for i in range(self.n_agent)  
        ])

    def set_to_init_level(self, auto_transfer=True):
        if self.init_level!=self.current_level:
            for i in range(self.current_level, self.init_level):
                self.change_div_tree_level(i+1, auto_transfer)


    def handle_update(self, update_cnt):
        pass 


    def change_div_tree_level(self, level, auto_transfer=True):
        print('performing div tree level change (%d -> %d/%d) \n'%(self.current_level, level, self.max_level))
        self.current_level = level
        assert len(self.div_tree) > self.current_level, ('Reach max level already!')
        if not auto_transfer: return
        transfer_list = []
        for i in range(self.n_agent):
            previous_net_index = self.div_tree[self.current_level-1, i]
            post_net_index = self.div_tree[self.current_level, i]
            if post_net_index!=previous_net_index:
                transfer = (previous_net_index, post_net_index)
                if transfer not in transfer_list:
                    transfer_list.append(transfer)
        for transfer in transfer_list:
            from_which_net = transfer[0]
            to_which_net = transfer[1]
            self.nets[to_which_net].load_state_dict(self.nets[from_which_net].state_dict())
            print('transfering model parameters from %d-th net to %d-th net'%(from_which_net, to_which_net))
        return 

    def forward(self, x_in, req_confact_info):  # x0: shape = (?,...,?, n_agent, core_dim)
        x0 = add_onehot_id_at_last_dim(x_in)
        # x1 = self.shared_net(x0)
        res = []
        for i in range(self.n_agent):
            use_which_net = self.div_tree[self.current_level, i]
            res.append(self.nets[use_which_net](x0[..., i, :]))
        x2 = torch.stack(res, -2)
        # x22 = self.nets[0](x1)
        
        if req_confact_info:
            ######### forward twice: shuffle forward
            perm_index = torch.randperm(self.n_agent, device=x_in.device)   # shape = (n_agent)
            perm_index = perm_index.expand(x_in.shape[:-1]) # shape = (...?, n_agent)
            # x_in shape = (...?, n_agent, coredim)
            perm_x_in = gather_righthand(src=x_in, index=perm_index, check=False)
            perm_x0 = add_onehot_id_at_last_dim(perm_x_in)
            perm_res = []
            for i in range(self.n_agent):
                use_which_net = self.div_tree[self.current_level, i]
                perm_res.append(self.nets[use_which_net](perm_x0[..., i, :]))
            perm_x2 = torch.stack(perm_res, -2)

            confact_x2 = torch.zeros_like(perm_x2) + np.nan
            confact_x2 = scatter_righthand(scatter_into=confact_x2, src=perm_x2, index=perm_index)
            assert not torch.isnan(confact_x2).any()
            confact_info = {'orig_out':x2, 'confact_out':confact_x2, 'perm_index':perm_index}
        else:
            confact_info = None
            
        return x2, confact_info

    def get_blood_distance(self):
        current_level = self.current_level

        # not first run, read buffer
        if hasattr(self, 'blood_distance_each_level'):
            return self.blood_distance_each_level[current_level]

        # on first run
        tree = self.div_tree
        n_agent = tree.shape[-1]
        self.blood_distance_each_level = np.ones(shape=(self.n_level, n_agent, n_agent), dtype=np.float64) * np.nan
        for m in range(self.n_level):
            n_agent = tree.shape[-1]
            blood_distance = np.ones(shape=(n_agent,n_agent), dtype=np.float64) * np.nan
            for i in range(n_agent):
                for j in range(n_agent):
                    if i==j:blood_distance[i,j] = 0
                    if not np.isnan(blood_distance[i,j]):
                        continue
                    for t in range(m+1):
                        investigate_level = (m) - t
                        if tree[investigate_level, i] == tree[investigate_level, j]:
                            blood_distance[i,j] = t
                            blood_distance[j,i] = t
                            break
            self.blood_distance_each_level[m, :] = blood_distance

        assert not np.isnan(self.blood_distance_each_level).any()
        self.blood_distance_each_level = _2tensor(self.blood_distance_each_level)

        if hasattr(self, 'blood_distance_each_level'):
            return self.blood_distance_each_level[current_level]
        else:
            assert False

    def get_weighted_kld(self, kld_matrix):
        blood_distance = self.get_blood_distance()
        return (kld_matrix*blood_distance).mean()
    # def forward_try_parallel(self, x0):  # x0: shape = (?,...,?, n_agent, core_dim)
    #     x1 = self.shared_net(x0)
    #     stream = []
    #     res = []
    #     for i in range(self.n_agent):
    #         stream.append(torch.cuda.Stream())
        
    #     torch.cuda.synchronize()
    #     for i in range(self.n_agent):
    #         use_which_net = self.div_tree[self.current_level, i]
    #         with torch.cuda.stream(stream[i]):
    #             res.append(self.nets[use_which_net](x1[..., i, :]))
    #             print(res[i])

    #     # s1 = torch.cuda.Stream()
    #     # s2 = torch.cuda.Stream()
    #     # # Wait for the above tensors to initialise.
    #     # torch.cuda.synchronize()
    #     # with torch.cuda.stream(s1):
    #     #     C = torch.mm(A, A)
    #     # with torch.cuda.stream(s2):
    #     #     D = torch.mm(B, B)
    #     # Wait for C and D to be computed.
    #     torch.cuda.synchronize()
    #     # Do stuff with C and D.

    #     x2 = torch.stack(res, -2)

    #     return x2



def _2div(arr):
    arr_res = arr.copy()
    arr_pieces = []
    pa = 0
    st = 0
    needdivcnt = 0
    for i, a in enumerate(arr):
        if a!=pa:
            arr_pieces.append([st, i])
            if (i-st)!=1: needdivcnt+=1
            pa = a
            st = i

    arr_pieces.append([st, len(arr)])
    if (len(arr)-st)!=1: needdivcnt+=1

    offset = range(len(arr_pieces), len(arr_pieces)+needdivcnt)
    p=0
    for arr_p in arr_pieces:
        length = arr_p[1] - arr_p[0]
        if length == 1: continue
        half_len = int(np.ceil(length / 2))
        for j in range(arr_p[0]+half_len, arr_p[1]):
            try:
                arr_res[j] = offset[p]
            except:
                print('wtf')
        p+=1
    return arr_res

def get_division_tree(n_agents):
    agent2divitreeindex = np.arange(n_agents)
    np.random.shuffle(agent2divitreeindex)
    max_div = np.ceil(np.log2(n_agents)).astype(int)
    levels = np.zeros(shape=(max_div+1, n_agents), dtype=int)
    tree_of_agent = []*(max_div+1)
    for ith, level in enumerate(levels):
        if ith == 0: continue
        res = _2div(levels[ith-1,:])
        levels[ith,:] = res
    res_levels = levels.copy()
    for i, div_tree_index in enumerate(agent2divitreeindex):
        res_levels[:, i] = levels[:, div_tree_index]
    return res_levels