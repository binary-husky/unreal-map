import numpy as np

def distance_matrix(A):
    assert A.shape[-1] == 2 or A.shape[-1] == 3  # assert 2D situation
    n_subject = A.shape[-2] # is 2
    A = np.repeat(np.expand_dims(A,-2), n_subject, axis=-2) # =>(64, 100, 100, 2)
    At = np.swapaxes(A,-2,-3) # =>(64, 100, 100, 2)
    dis = At-A # =>(64, 100, 100, 2)
    dis = np.linalg.norm(dis, axis=-1)
    return dis

# def distance_matrix3D(A):
#     assert A.shape[-1] == 3 # assert 2D situation
#     n_subject = A.shape[-2] # is 2
#     A = np.repeat(np.expand_dims(A,-2), n_subject, axis=-2) # =>(64, 100, 100, 3)
#     At = np.swapaxes(A,-2,-3) # =>(64, 100, 100, 3)
#     dis = At-A # =>(64, 100, 100, 3)
#     dis = np.linalg.norm(dis, axis=-1)
#     return dis
