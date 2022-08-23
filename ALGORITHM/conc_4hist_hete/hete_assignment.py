import numpy as np
from UTIL.tensor_ops import my_view, __hash__, repeat_at, gather_righthand
EXCLUDE_FRONTEND = True
def random_group(n_thread, hete_type, n_hete_types, n_group):
    n_agent = hete_type.shape[-1]
    res = np.zeros(shape=(n_thread, n_agent), dtype=int)
    for i in range(n_thread):
        low_group = 1 if EXCLUDE_FRONTEND else 0
        group_assignment = np.random.randint(low=low_group, high=n_group, size=(n_hete_types))
        for ht, group in enumerate(group_assignment):
            mask = (hete_type == ht) # bool mask, find ht type agents, 1D=[n_agent,]
            res[i,mask] = group
    return res
            
# select_nets_for_shellenv(n_types=n_types, 
#                          policy=self.RL_functional,
#                          hete_type_list=self.hete_type,
#                          n_thread = n_thread,
#                          n_gp=AlgorithmConfig.n_policy_groups
#                          )   


def select_nets_for_shellenv(n_types, policy, hete_type_list, n_thread,n_gp):
    # choose one hete type
    selected_type = np.random.randint(low=0, high=n_types, size=())
    for i in range(n_types): policy.lock_frontend_type(i)
    policy.unlock_frontend_type(selected_type)
    # select corrisponding agents
    selected_agent_bool = (hete_type_list==selected_type)
    selected_agent_bool = repeat_at(selected_agent_bool, 0, n_thread)
    # generate a random group selection array
    group_sel_arr = random_group(n_thread=n_thread, hete_type=hete_type_list, n_hete_types=n_types, n_group=n_gp)
    # group to net index
    n_tp = n_types
    get_placeholder = lambda type, group: group*n_tp + type
    get_type_group = lambda ph: (ph%n_tp, ph//n_tp)
    hete_type_arr = repeat_at(hete_type_list, 0, n_thread)
    selected_nets = get_placeholder(type=hete_type_arr, group=group_sel_arr)
    # replace chosen agents
    selected_nets[selected_agent_bool] = (selected_type)
    return selected_nets


    
# [[n.grad for n in net.parameters()] for net in self.policy_and_critic._nets_flat_placeholder_]


# [net.training for net in self.policy_and_critic._nets_flat_placeholder_]
# [net.lock for net in self.policy_and_critic._nets_flat_placeholder_]



