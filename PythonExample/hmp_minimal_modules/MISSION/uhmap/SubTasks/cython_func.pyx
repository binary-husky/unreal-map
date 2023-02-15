import numpy as np
cimport numpy as np
cimport cython
np.import_array()


ctypedef fused DTYPE_all_t:
    np.float32_t
    np.float64_t
    np.int64_t
    np.int32_t  # to compat Wi
ctypedef fused DTYPE_t:
    np.float32_t
    np.float64_t


ctypedef fused DTYPE_intlong_t:
    np.int64_t
    np.int32_t  # to compat Windows
    
    
    
# x: input
# n_digit: output dimension
# base: 进制
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def tear_number_apart(np.float64_t x, DTYPE_intlong_t n_digit, DTYPE_intlong_t base=16, DTYPE_all_t mv_left=8):
    cdef np.ndarray out = np.zeros(n_digit, dtype=float)
    cdef int p = n_digit
    cdef float tmp = 0
    reverse = x < 0
    cdef float m_init = base
    if reverse: x = -x
    m_init = m_init ** mv_left
    x = x * m_init
    for _ in range(n_digit):
        p -= 1
        if p==0: 
            out[p] = x
            break
        tmp = x % base
        out[p] = tmp
        x = x // base
    if reverse: out = -out
    return out


def tear_num_arr(arr, DTYPE_intlong_t n_digit, DTYPE_intlong_t base, DTYPE_all_t mv_left):
    return np.concatenate([tear_number_apart(x, n_digit, base, mv_left) for x in arr], axis=0)

def comb_num_back(arr, n_digit, base, mv_left):
    out = 0
    tmp = base ** (n_digit - mv_left - 1)
    for x in arr:
        out += x * tmp
        tmp = tmp/base
        
    return out


'''
test <1>

parts = tear_number_apart(255, n_digit=10, base=2, mv_left=1)
print(parts)
comb_num_back(parts, n_digit=10, base=2, mv_left=1)


test <2>

parts = tear_number_apart(255.778, n_digit=10, base=10, mv_left=-1)
print(parts)
comb_num_back(parts, n_digit=10, base=10, mv_left=-1)

test <3>

for i in range(1000):
    q = (np.random.rand() - 0.5)*1e3
    parts = tear_number_apart(q, n_digit=10, base=10, mv_left=0)
    print(q, parts)
    res = np.abs(comb_num_back(parts, n_digit=10, base=10, mv_left=0)-q) < 1e-6
    if not res:
        print('??? np.abs(comb_num_back(parts, n_digit=10, base=10, mv_left=0)-q)', np.abs(comb_num_back(parts, n_digit=10, base=10, mv_left=0)-q))
        assert False
'''