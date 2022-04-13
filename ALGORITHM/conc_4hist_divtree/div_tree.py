import torch
import torch.nn as nn
import numpy as np
from ..commom.mlp import LinearFinal

class DivTree(nn.Module): # merge by MLP version
    def __init__(self, input_dim, h_dim, n_action):
        super().__init__()

        # to design a division tree, I need to get the total number of agents
        from .foundation import AlgorithmConfig
        self.n_agent = AlgorithmConfig.n_agent
        self.current_level = 0
        self.div_tree = get_division_tree(self.n_agent)
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, h_dim), 
            nn.ReLU(inplace=True)
        )

        get_net = lambda: nn.Sequential(
            nn.Linear(h_dim, h_dim), 
            nn.ReLU(inplace=True),
            LinearFinal(h_dim, n_action)
        )
        # Note: this is NOT net defining for each agent
        # Instead, all agents starts from self.nets[0]
        self.nets = torch.nn.ModuleList(modules=[
            get_net() for i in range(self.n_agent)  
        ])


    def change_div_tree_level(self, level, auto_transfer=True):
        print('performing div tree level change (%d -> %d), div tree:\n'%(self.current_level, level), self.div_tree)
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

    def forward(self, x0):  # x0: shape = (?,...,?, n_agent, core_dim)
        x1 = self.shared_net(x0)
        res = []
        for i in range(self.n_agent):
            use_which_net = self.div_tree[self.current_level, i]
            res.append(self.nets[use_which_net](x1[..., i, :]))
        x2 = torch.stack(res, -2)
        # x22 = self.nets[0](x1)
        return x2

    def forward_try_parallel(self, x0):  # x0: shape = (?,...,?, n_agent, core_dim)
        x1 = self.shared_net(x0)
        stream = []
        res = []
        for i in range(self.n_agent):
            stream.append(torch.cuda.Stream())
        
        torch.cuda.synchronize()
        for i in range(self.n_agent):
            use_which_net = self.div_tree[self.current_level, i]
            with torch.cuda.stream(stream[i]):
                res.append(self.nets[use_which_net](x1[..., i, :]))
                print(res[i])

        # s1 = torch.cuda.Stream()
        # s2 = torch.cuda.Stream()
        # # Wait for the above tensors to initialise.
        # torch.cuda.synchronize()
        # with torch.cuda.stream(s1):
        #     C = torch.mm(A, A)
        # with torch.cuda.stream(s2):
        #     D = torch.mm(B, B)
        # Wait for C and D to be computed.
        torch.cuda.synchronize()
        # Do stuff with C and D.

        x2 = torch.stack(res, -2)

        return x2



def get_division_tree(n_agents):
    agent2divitreeindex = np.arange(n_agents)
    np.random.shuffle(agent2divitreeindex)
    max_div = np.ceil(np.log2(n_agents)).astype(int)
    levels = np.zeros(shape=(max_div+1, n_agents), dtype=int)
    tree_of_agent = []*(max_div+1)
    for ith, level in enumerate(levels):
        if ith == 0: continue
        levels[ith,:]  = levels[ith-1,:]
        for j in range(n_agents):
            seg = j // ( n_agents /2**ith)
            if seg%2==1:
                levels[ith,j] += 2**(ith-1)
    res_levels = levels.copy()
    for i, div_tree_index in enumerate(agent2divitreeindex):
        res_levels[:, i] = levels[:, div_tree_index]
    return res_levels
