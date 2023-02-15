import copy
import numpy as np
from UTIL.tensor_ops import my_view, __hash__, repeat_at, gather_righthand
from .foundation import AlgorithmConfig



def random_group(random_select_fn, n_thread, hete_type, n_hete_types, n_group, selected_tps, testing):
    n_agent = hete_type.shape[-1]
    group_sel_arr = np.zeros(shape=(n_thread, n_agent), dtype=int)
    gp_sel_summary = []
    for i in range(n_thread):
        group_assignment = np.array([ 
            random_select_fn(testing) 
            if type not in selected_tps[i] else 0
            for type in range(n_hete_types)
        ])
        assert (group_assignment[selected_tps[i]]==0).all()
        gp_sel_summary.append(copy.deepcopy(group_assignment))
        for ht, group in enumerate(group_assignment):
            mask = (hete_type == ht)
            group_sel_arr[i,mask] = group
    return group_sel_arr, np.stack(gp_sel_summary).astype(np.int64)




def select_nets_for_shellenv(n_types, policy, hete_type_list, n_thread, n_gp, testing):
    if (not testing) or (AlgorithmConfig.policy_matrix_testing):
        n_alive_frontend = AlgorithmConfig.hete_n_alive_frontend
        tmp = np.arange(n_types)
        # select types to use frontier
        if not AlgorithmConfig.type_sel_override:
            selected_types = np.stack([
                np.random.choice(
                    a=tmp,
                    size=(n_alive_frontend),
                    replace=False,
                    p=None)
                for _ in range(n_thread)
            ])
        else:
            selected_types = np.stack([
                AlgorithmConfig.type_sel_override_list
                for _ in range(n_thread)
            ]) 
    else:
        # testing but not policy_matrix_testing: select all types to use frontier
        selected_types = np.stack([np.arange(n_types) for _ in range(n_thread)])
    
    # generate a random group selection array
    if not AlgorithmConfig.policy_matrix_testing:
        random_select_fn = policy.random_select 
    else:
        random_select_fn = policy.random_select_matrix_test

    group_sel_arr, gp_sel_summary = random_group(
        random_select_fn=random_select_fn, n_thread=n_thread, hete_type=hete_type_list, 
        n_hete_types=n_types, n_group=n_gp, selected_tps=selected_types, testing=testing)
    # group to net index
    n_tp = n_types
    get_placeholder = lambda type, group: group*n_tp + type
    hete_type_arr = repeat_at(hete_type_list, 0, n_thread)
    selected_nets = get_placeholder(type=hete_type_arr, group=group_sel_arr)
    
    return selected_nets, gp_sel_summary





