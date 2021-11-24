import numpy as np
cimport numpy as np
cimport cython
np.import_array()
DTYPE = np.float32
ctypedef np.float32_t DTYPE_t
ctypedef np.int64_t DTYPE_int64_t
ctypedef np.uint8_t DTYPE_bool_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def new_method( np.ndarray[DTYPE_t, ndim=4] obs_feed_new, 
                np.ndarray[DTYPE_t, ndim=4] prev_obs_feed, 
                np.ndarray[DTYPE_bool_t, ndim=3] valid_mask, 
                np.ndarray[DTYPE_int64_t, ndim=2] N_valid, 
                np.ndarray[DTYPE_t, ndim=4] next_his_pool):
    cdef int vmax = N_valid.shape[0]
    cdef int wmax = N_valid.shape[1]

    for th in range(vmax):
        for a in range(wmax):
            n_v = N_valid[th,a]
            next_his_pool[th,a,:n_v] = obs_feed_new[th,a,valid_mask[th,a]]
            next_his_pool[th,a,n_v:] = prev_obs_feed[th,a,:(12-n_v)]
    return next_his_pool