import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange
np.import_array()
ctypedef np.float32_t DTYPE_t
ctypedef fused DTYPE_intlong_t:
    np.int64_t
    np.int32_t  # to compat Windows
ctypedef np.uint8_t DTYPE_bool_t
# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.nonecheck(False)
# def new_method( np.ndarray[DTYPE_t, ndim=4] obs_feed_new, 
#                 np.ndarray[DTYPE_t, ndim=4] prev_obs_feed, 
#                 np.ndarray[DTYPE_bool_t, ndim=3] valid_mask, 
#                 np.ndarray[DTYPE_int64_t, ndim=2] N_valid, 
#                 np.ndarray[DTYPE_t, ndim=4] next_his_pool):
#     cdef int vmax = N_valid.shape[0]
#     cdef int wmax = N_valid.shape[1]
# 
#     for th in range(vmax):
#         for a in range(wmax):
#             n_v = N_valid[th,a]
#             next_his_pool[th,a,:n_v] = obs_feed_new[th,a,valid_mask[th,a]]
#             next_his_pool[th,a,n_v:] = prev_obs_feed[th,a,:(12-n_v)]
#     return next_his_pool



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def roll_hisory( DTYPE_t[:,:,:,:] obs_feed_new, 
                DTYPE_t[:,:,:,:] prev_obs_feed, 
                DTYPE_bool_t[:,:,:] valid_mask, 
                DTYPE_intlong_t[:,:] N_valid, 
                DTYPE_t[:,:,:,:] next_his_pool):
    cdef Py_ssize_t vmax = N_valid.shape[0]
    cdef Py_ssize_t wmax = N_valid.shape[1]
    cdef Py_ssize_t max_obs_entity = obs_feed_new.shape[2]
    cdef int n_v, th, a, t, k, pointer
    for th in prange(vmax, nogil=True):
        for a in range(wmax):
            pointer = 0
            for k in range(max_obs_entity):
                if valid_mask[th,a,k]:
                    next_his_pool[th,  a,  pointer] = obs_feed_new[th,a,k]
                    pointer = pointer + 1
            n_v = N_valid[th,a]
            for k in range(n_v, max_obs_entity):
                next_his_pool[th,a,k] = prev_obs_feed[th,a,k-n_v]
    return np.asarray(next_his_pool)