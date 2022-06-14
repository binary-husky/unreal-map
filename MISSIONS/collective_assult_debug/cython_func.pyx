import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange
from libc.math cimport cos, atan2, abs

np.import_array()
ctypedef np.float64_t DTYPE_F64_t
ctypedef np.float32_t DTYPE_t
ctypedef np.uint8_t DTYPE_bool_t

PI = np.pi
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef reg_rad(DTYPE_F64_t rad):
    return (rad + PI) % (2 * PI) - PI

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def laser_hit_improve3(
    np.ndarray[DTYPE_F64_t, ndim=1] pos_o, np.ndarray[DTYPE_F64_t, ndim=1] pos_t, DTYPE_F64_t fanRadius, DTYPE_F64_t fanOpenRad, DTYPE_F64_t fanDirRad):

    cdef DTYPE_F64_t dis_square
    cdef np.ndarray[DTYPE_F64_t, ndim=1] delta

    cdef DTYPE_F64_t ori_rad_pos, ori_rad_neg, ori_2tgt

    delta = pos_t - pos_o
    dis_square = delta[0]*delta[0] + delta[1]*delta[1]

    if dis_square > fanRadius*fanRadius: return False
    
    ori_rad_pos = fanDirRad + fanOpenRad/2
    ori_rad_neg = fanDirRad - fanOpenRad/2
    
    ori_2tgt = atan2(delta[1],delta[0])
    
    d1rad = abs(reg_rad(ori_rad_pos - ori_2tgt))
    d2rad = abs(reg_rad(ori_rad_neg - ori_2tgt))
    
    if d1rad <= fanOpenRad and d2rad <= fanOpenRad:
        return True
    else:
        return False
