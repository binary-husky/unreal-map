import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange
from libc.math cimport cos, atan2, abs

np.import_array()
ctypedef np.float64_t DTYPE_F64_t
ctypedef np.float32_t DTYPE_t
ctypedef fused DTYPE_int64_t:
    np.int64_t
    np.int32_t  # to compat Windows
ctypedef np.uint8_t DTYPE_bool_t

PI = np.pi
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def reg_rad_arr(DTYPE_F64_t[:] rad):
    cdef Py_ssize_t dim = rad.shape[0]
    cdef Py_ssize_t x, y
    result = np.zeros(dim, dtype=np.double)
    cdef DTYPE_F64_t[:] result_view = result
    cdef DTYPE_F64_t PI = np.pi

    for x in prange(dim, nogil=True):
        result_view[x] = (rad[x] + PI) % (2*PI) - PI
    return result


# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.nonecheck(False)
# def roll_hisory( DTYPE_t[:,:,:,:] obs_feed_new, 
#                 DTYPE_t[:,:,:,:] prev_obs_feed, 
#                 DTYPE_bool_t[:,:,:] valid_mask, 
#                 DTYPE_int64_t[:,:] N_valid, 
#                 DTYPE_t[:,:,:,:] next_his_pool):
#     cdef Py_ssize_t vmax = N_valid.shape[0]
#     cdef Py_ssize_t wmax = N_valid.shape[1]
#     cdef Py_ssize_t max_obs_entity = obs_feed_new.shape[2]
#     cdef int n_v, th, a, t, k, pointer
#     for th in prange(vmax, nogil=True):
#         for a in range(wmax):
#             pointer = 0
#             for k in range(max_obs_entity):
#                 if valid_mask[th,a,k]:
#                     next_his_pool[th,  a,  pointer] = obs_feed_new[th,a,k]
#                     pointer = pointer + 1
#             n_v = N_valid[th,a]
#             for k in range(n_v, max_obs_entity):
#                 next_his_pool[th,a,k] = prev_obs_feed[th,a,k-n_v]
#     return np.asarray(next_his_pool)


#  https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html?highlight=wraparound#compiler-directives
'''
    binding (True): Python函数的内省, 查看函数内部的细节['__class__',  '__delatrr__', ...., 'co_code', 'co_filename', 'co_argcount', 'co_varnames',...]等等
    boundscheck (True): 数组的边界检查
    wraparound (True） ： 是否支持索引倒数，如a[-1]

    initializedcheck (True / False): ?
    nonecheck (False)
    always_allow_keywords (True / False)
    profile (False):   Write hooks for Python profilers into the compiled C code. Default is False.


    infer_types (True / False): Infer types of untyped variables in function bodies. Default is None, indicating that only safe (semantically-unchanging) inferences are allowed. In particular, inferring integral types for variables used in arithmetic expressions is considered unsafe (due to possible overflow) and must be explicitly requested.

'''