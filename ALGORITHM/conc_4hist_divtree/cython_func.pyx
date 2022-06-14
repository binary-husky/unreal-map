import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange
np.import_array()
ctypedef np.float32_t DTYPE_t
ctypedef fused DTYPE_int64_t:
    np.int64_t
    np.int32_t  # to compat Windows
ctypedef np.uint8_t DTYPE_bool_t


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def roll_hisory( DTYPE_t[:,:,:,:] obs_feed_new, 
                DTYPE_t[:,:,:,:] prev_obs_feed, 
                DTYPE_bool_t[:,:,:] valid_mask, 
                DTYPE_int64_t[:,:] N_valid, 
                DTYPE_t[:,:,:,:] next_his_pool):
    # how many threads
    cdef Py_ssize_t vmax = N_valid.shape[0]
    # how many agents
    cdef Py_ssize_t wmax = N_valid.shape[1]
    # how many entity subjects (including self @0)
    cdef Py_ssize_t max_obs_entity = obs_feed_new.shape[2]
    cdef int n_v, th, a, t, k, pointer
    for th in prange(vmax, nogil=True):
        # for each thread range -> prange
        for a in prange(wmax):
            # for each agent
            pointer = 0
            # step 1 fill next_his_pool[0 ~ (nv-1)] with obs_feed_new[0 ~ max_obs_entity-1]
            for k in range(max_obs_entity):
                if valid_mask[th,a,k]:
                    next_his_pool[th,a, pointer] = obs_feed_new[th,a, k]
                    pointer = pointer + 1

            # step 2 fill next_his_pool[nv ~ (max_obs_entity-1)] with prev_obs_feed[0 ~ (max_obs_entity-1-nv)]
            n_v = N_valid[th,a]
            for k in range(n_v, max_obs_entity):
                next_his_pool[th,a, k] = prev_obs_feed[th,a, k-n_v]
    return np.asarray(next_his_pool)

