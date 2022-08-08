


import numpy as np
from UTIL.tensor_ops import distance_mat_between
from scipy.optimize import linear_sum_assignment
inf = float('inf')
opp_pos_arr = np.array([
    [1, 2], [2, 2], [3, 2], [4, 2], [5, 2], [6, 2], [7, 2], [8, 2],
    [inf, inf], [2, 1], [3, 1], [4, 1], [5, 1], [6, 1], [7, 1], [8, 1],
])

opp_id_arr = np.array([
   0, 1, 2,  3,  4,  5,  6,  7,  
   8, 9, 10, 11, 12, 13, 14, 15,
])


leader_pos_arr= np.array([
    [0, 0], 
    [7, 0],
])
leader_id_arr = np.array([
    -1, -2,
])


def assign_opponent(opp_pos_arr, opp_id_arr, leader_pos_arr, leader_id_arr):
    result = {}
    dis_mat = distance_mat_between(leader_pos_arr, opp_pos_arr)

    indices, assignments = linear_sum_assignment(dis_mat)
    for i, j, a in zip(range(len(indices)), indices, assignments):
        assert i == j
        result[leader_id_arr[i]] = opp_id_arr[a]

    return result


result = assign_opponent(opp_pos_arr, opp_id_arr, leader_pos_arr, leader_id_arr)
print(result)

