import copy
import numpy as np
from UTIL.tensor_ops import my_view, __hash__, repeat_at, gather_righthand
from .foundation import AlgorithmConfig

def random_group(policy, n_thread, hete_type, n_hete_types, n_group, selected_tps):
    n_agent = hete_type.shape[-1]
    res = np.zeros(shape=(n_thread, n_agent), dtype=int)
    gp_sel_summary = []
    MaximumActiveNets = AlgorithmConfig.hete_max_active_groups
    rand_ops = [np.random.rand() for _ in range(MaximumActiveNets)]
    for i in range(n_thread):
        # include
        group_assignment = np.array([
                policy.random_select(
                    rand_ops=rand_ops
                ) 
            for _ in range(n_hete_types)])
        
        # group_assignment = np.random.randint(low=low_group, high=n_group, size=(n_hete_types))
        group_assignment[selected_tps[i]]=0
        gp_sel_summary.append(copy.deepcopy(group_assignment))
        for ht, group in enumerate(group_assignment):
            mask = (hete_type == ht) # bool mask, find ht type agents, 1D=[n_agent,]
            res[i,mask] = group
    return res, np.stack(gp_sel_summary)

# select_nets_for_shellenv(n_types=n_types, 
#                          policy=self.rl_functional,
#                          hete_type_list=self.hete_type,
#                          n_thread = n_thread,
#                          n_gp=AlgorithmConfig.hete_n_net_placeholder
#                          )   

# 在每次环境重置时
def select_nets_for_shellenv(n_types, policy, hete_type_list, n_thread, n_gp, testing):
    # choose one hete type
    n_alive_frontend = AlgorithmConfig.hete_n_alive_frontend


    # 锁定没有用到的frontend网络，但一般在一个大batch中都会用到
    # for i in range(n_types): policy.lock_net(i, forbidden=True)
    # policy.unlock_net(selected_type)
    tmp = np.arange(n_types)
    selected_types = np.stack([
        np.random.choice(
            a=tmp,
            size=(n_alive_frontend),
            replace=False,
            p=None)
        for _ in range(n_thread)
    ])
    if testing: selected_types = np.stack([np.arange(n_types) for _ in range(n_thread)])
    
    # generate a random group selection array
    group_sel_arr, gp_sel_summary = random_group(policy=policy, n_thread=n_thread, hete_type=hete_type_list, n_hete_types=n_types, n_group=n_gp, selected_tps=selected_types)
    # group to net index
    n_tp = n_types
    get_placeholder = lambda type, group: group*n_tp + type
    get_type_group = lambda ph: (ph%n_tp, ph//n_tp)
    hete_type_arr = repeat_at(hete_type_list, 0, n_thread)
    selected_nets = get_placeholder(type=hete_type_arr, group=group_sel_arr)
    

    # # selected_type = np.random.randint(low=0, high=n_types, size=())
    # # select corrisponding agents
    # selected_agent_bool_1d = np.array([(i in selected_types) for i in hete_type_list])
    # # selected_agent_bool = repeat_at(selected_agent_bool_1d, 0, n_thread)
    # # replace chosen agents
    # selected_nets[:, selected_agent_bool_1d] = repeat_at(hete_type_list[selected_agent_bool_1d], 0, n_thread) # (selected_type) #  + group*n_tp (group=0)
    return selected_nets, gp_sel_summary


    
# [[n.grad for n in net.parameters()] for net in self.policy_and_critic._nets_flat_placeholder_]


# [net.training for net in self.policy_and_critic._nets_flat_placeholder_]
# [net.lock for net in self.policy_and_critic._nets_flat_placeholder_]
# [net.forbidden for net in self.policy_and_critic._nets_flat_placeholder_]



