import copy, json
import numpy as np
try:
    import torch
    import torch.nn.functional as F
except:
    print('warning, pytorch not installed!')
    print('警告, 没有安装pytorch, 所有pytorch相关函数不可用!')
    class torch():
        Tensor = Exception
from functools import wraps


class ConfigCache(object):
    def __init__(self) -> None:
        super().__init__()
        self.init = False

    def read_cfg(self):
        from config import GlobalConfig
        if GlobalConfig.cfg_ready:
            self.device_ = GlobalConfig.device
            self.use_float64_ = GlobalConfig.use_float64
            self.init = True

    @property
    def device(self):
        if not self.init: self.read_cfg()
        assert self.init, ('cfg not ready!')
        return self.device_

    @property
    def use_float64(self):
        if not self.init: self.read_cfg()
        assert self.init, ('cfg not ready!')
        return self.use_float64_

cfg = ConfigCache()

def pt_inf():
    # if not cfg.init: cfg.read_cfg()
    pt_dtype = torch.float64 if cfg.use_float64 else torch.float32
    return torch.tensor(np.inf, dtype=pt_dtype, device=cfg.device)

def pt_nan():
    # if not cfg.init: cfg.read_cfg()
    pt_dtype = torch.float64 if cfg.use_float64 else torch.float32
    return torch.tensor(np.nan, dtype=pt_dtype, device=cfg.device)


def vis_mat(mat):
    mat = mat.astype(np.float)
    mat = mat - mat.min()
    mat = mat / mat.max()
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    imgplot = plt.imshow(mat)
    plt.xlabel("cols, 2rd dim")
    plt.ylabel("lines, 1st dim")
    plt.show()


"""
    improve torch.repeat / torch.expand function
    eg.1    x.shape = (4, 5, 6, 7); insert_dim = -1; n_times=666
            y = repeat_at(x, insert_dim, n_times)
            y.shape = (4, 5, 6, 7, 666)

    eg.2    x.shape = (4, 5, 6, 7); insert_dim = +1; n_times=666
            y = repeat_at(x, insert_dim, n_times)
            y.shape = (4, 666, 5, 6, 7)
"""
def repeat_at(tensor, insert_dim, n_times, copy_mem=False):
    if not isinstance(tensor, torch.Tensor):
        return np_repeat_at(tensor, insert_dim, n_times)
    tensor = tensor.unsqueeze(insert_dim)
    shape = list(tensor.shape)
    assert shape[insert_dim] == 1
    shape[insert_dim] = n_times
    if copy_mem: tensor.repeat(*shape)
    return tensor.expand(*shape)

def np_repeat_at(array, insert_dim, n_times):
    array = np.expand_dims(array, insert_dim)
    return array.repeat(axis=insert_dim, repeats=n_times)


def copy_clone(x):
    if x is None:
        return None
    return (
        x.clone()
        if hasattr(x, "clone")
        else x.copy()
        if hasattr(x, "copy")
        else copy.deepcopy(x)
    )


"""
    improve np.reshape and torch.view function
    If a dim is assigned with 0, it will keep its original dimension
    eg.1    x.shape = (4, 5, 6, 7); new_shape = [0, 0, -1]
            y = my_view(x, new_shape)
            y.shape = (4, 5, 6*7)

    eg.2    x.shape = (4, 5, 6, 7); new_shape = [-1, 0, 0]
            y = my_view(x, new_shape)
            y.shape = (4*5, 6, 7)

    eg.3    x.shape = (4, 5, 6); new_shape = [0, 0, -1, 3]
            y = my_view(x, new_shape)
            y.shape = [4, 5, 2, 3]

    eg.4    x.shape = (3, 4, 5, 6); new_shape = [0, 2, -1, 0, 0]
            y = my_view(x, new_shape)
            y.shape = [3, 2, 2, 5, 6]


"""
def my_view(x, shape):
    if -1 in shape[1:-1]: return my_view_test(x, shape)
    reverse_lookup = True if shape[0] == -1 else False
    if not reverse_lookup:
        for i, dim in enumerate(shape):
            if dim == 0:
                shape[i] = x.shape[i]
    else:
        for i in range(len(shape)):
            ni = -(i + 1)  # iter -1,-2,-3,...
            dim = shape[ni]
            if dim == 0:
                shape[ni] = x.shape[ni]
    if isinstance(x, np.ndarray):
        return x.reshape(*shape)
    return x.view(*shape)

def my_view_test(x, shape):
    # fill both way until meet -1 
    for i, dim in enumerate(shape):
        if dim == 0: shape[i] = x.shape[i]
        elif dim == -1: break
    for i in range(len(shape)):
        ni = -(i + 1); dim = shape[ni]
        if dim == 0: shape[ni] = x.shape[ni]
        elif dim == -1: break
    # print(shape)
    if isinstance(x, np.ndarray):
        return x.reshape(*shape)
    return x.view(*shape)


def add_onehot_id_at_last_dim(x):
    if isinstance(x, np.ndarray):
        return np_add_onehot_id_at_last_dim(x)
    _hot_dim = x.shape[-2]
    _identity = torch.tile(torch.eye(_hot_dim, device=x.device), (*x.shape[:-2], 1, 1))
    return torch.cat((x, _identity), -1)

def np_add_onehot_id_at_last_dim(x):
    _hot_dim = x.shape[-2]
    _identity = np.tile(np.eye(_hot_dim), (*x.shape[:-2], 1, 1))
    return np.concatenate((x, _identity), -1)





"""
    numpy corresponding to torch.nn.functional.one_hot
    x is array, e.g. x = [4,2,3,1]
    n is int, e.g. n=5
    >> np_one_hot( np.array([4,2,3,1]), n=5)
    np.array([
        [0,0,0,0,1],
        [0,0,1,0,0],
        [0,0,0,1,0],
        [0,1,0,0,0],
    ])

"""
def np_one_hot(x, n):
    return np.eye(n)[x]


def add_obs_container_subject(container_emb, subject_emb, div):
    # for subject, add one-hot embedding of its group
    n_container = container_emb.shape[1]
    subject_belonging_info = np_one_hot(div, n_container)
    subject_out_emb = np.concatenate((subject_emb, subject_belonging_info), -1)
    # for container, add add multi-hot embedding of its subjects
    container_multihot = np.concatenate(
        [np.expand_dims((div == nth_container).astype(np.long), 1) 
            for nth_container in range(n_container)],
        1,
    )
    container_out_emb = np.concatenate((container_emb, container_multihot), -1)
    return container_out_emb, subject_out_emb


def MayGoWrong(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            print('going wrong!')
            return f(*args, **kwargs)

    return decorated



"""
    Function decorate, 
    Turning numpy array to torch.Tensor, then put it on the right GPU / CPU
"""
def Args2tensor(f):
    # if not cfg.init: cfg.read_cfg()
    def _2tensor(x):
        if isinstance(x, torch.Tensor):
            return x.to(cfg.device)
        elif isinstance(x, np.ndarray):
            if (not cfg.use_float64) and x.dtype == np.float64:
                x = x.astype(np.float32)
            if cfg.use_float64 and x.dtype == np.float32:
                x = x.astype(np.float64)
            return torch.from_numpy(x).to(cfg.device)
        elif isinstance(x, dict):
            y = {}
            for key in x:
                y[key] = _2tensor(x[key])
            return y
        else:
            return x

    @wraps(f)
    def decorated(*args, **kwargs):
        for key in kwargs:
            kwargs[key] = _2tensor(kwargs[key])
        return f(*(_2tensor(arg) for arg in args), **kwargs)

    return decorated




def Return2numpy(f):

    def _2cpu2numpy(x):
        return (
            None
            if x is None
            else x
            if not isinstance(x, torch.Tensor)
            else x.detach().cpu().numpy()
            if x.requires_grad
            else x.cpu().numpy()
        )

    @wraps(f)
    def decorated(*args, **kwargs):
        ret_tuple = f(*args, **kwargs)
        if isinstance(ret_tuple, tuple):
            return (_2cpu2numpy(ret) for ret in ret_tuple)
        else:
            return _2cpu2numpy(ret_tuple)

    return decorated


"""
    Function decorate, 
    Turning numpy array to torch.Tensor, then put it on the right GPU / CPU,
    When returning, convert all torch.Tensor to numpy array
"""
def Args2tensor_Return2numpy(f):
    def _2tensor(x):
        if isinstance(x, torch.Tensor):
            return x.to(cfg.device)
        elif isinstance(x, np.ndarray) and x.dtype != 'object':
            if (not cfg.use_float64) and x.dtype == np.float64:
                x = x.astype(np.float32)
            if cfg.use_float64 and x.dtype == np.float32:
                x = x.astype(np.float64)
            return torch.from_numpy(x).to(cfg.device)
        elif isinstance(x, dict):
            y = {}
            for key in x:
                y[key] = _2tensor(x[key])
            return y
        else:
            return x

    def _2cpu2numpy(x):
        return (
            None
            if x is None
            else x
            if not isinstance(x, torch.Tensor)
            else x.detach().cpu().numpy()
            if x.requires_grad
            else x.cpu().numpy()
        )

    @wraps(f)
    def decorated(*args, **kwargs):
        for key in kwargs:
            kwargs[key] = _2tensor(kwargs[key])
        ret_tuple = f(*(_2tensor(arg) for arg in args), **kwargs)
        if not isinstance(ret_tuple, tuple):
            return _2cpu2numpy(ret_tuple)
        return (_2cpu2numpy(ret) for ret in ret_tuple)

    return decorated


"""
    Turning torch.Tensor to numpy array, put it on CPU,
"""


def _2cpu2numpy(x):
    return (
        None
        if x is None
        else x
        if not isinstance(x, torch.Tensor)
        else x.detach().cpu().numpy()
        if x.requires_grad
        else x.cpu().numpy()
    )


"""
    Convert torch.Tensor to numpy array.
    Turning numpy array to torch.Tensor, then put it on the right GPU / CPU.
"""


def _2tensor(x):
    # if not cfg.init: cfg.read_cfg()
    if isinstance(x, torch.Tensor):
        return x.to(cfg.device)
    elif isinstance(x, np.ndarray):
        if (not cfg.use_float64) and x.dtype == np.float64:
            x = x.astype(np.float32)
        if cfg.use_float64 and x.dtype == np.float32:
            x = x.astype(np.float64)
        return torch.from_numpy(x).to(cfg.device)
    elif isinstance(x, dict):
        y = {}
        for key in x:
            y[key] = _2tensor(x[key])
        return y
    elif isinstance(x, torch.nn.Module):
        x.to(cfg.device)
        return x
    else:
        return x

"""
    Stack an array whose elements with different len, pad empty place with with NaN
"""

def pad_vec_array(arr_list, max_len):
    # init to NaNs
    res = np.zeros(shape=(len(arr_list), max_len), dtype=np.double) + np.nan
    for i in range(len(arr_list)):
        if arr_list[i] is None:
            continue
        res[i, : len(arr_list[i])] = arr_list[i]
    return res


def one_hot_with_nan_np(tensr, num_classes):
    tensr = tensr.copy()
    tensr[np.isnan(tensr)] = num_classes
    Res_1MoreCol = np_one_hot(tensr.astype(np.long), num_classes + 1)
    return Res_1MoreCol[..., :-1]


def one_hot_with_nan(tensr, num_classes):
    if isinstance(tensr, np.ndarray):
        return one_hot_with_nan_np(tensr, num_classes)
    tensr = tensr.clone()
    tensr[torch.isnan(tensr)] = num_classes
    Res_1MoreCol = F.one_hot(tensr.long(), num_classes + 1)
    return Res_1MoreCol[..., :-1]


def scatter_with_nan(tensr, num_classes, out_type="binary"):
    res = one_hot_with_nan(tensr, num_classes)
    res = res.sum(-2)
    if out_type == "bool":
        res = res != 0
    return res

"""
    Not used anymore
"""
def process_space(space):
    # starcraft 环境无须特殊处理
    if not ("Box" in space["obs_space"] or "Discrete" in space["act_space"]):
        return space

    # 其他环境需要进行格式转换
    import re

    obs_dim = int(
        re.findall(
            re.compile(r"Box[(]-inf, inf, [(](.*?)[,)]", re.S), space["obs_space"]
        )[0]
    )
    print(space["obs_space"])
    space_ = {}
    space_["obs_space"] = {}
    space_["act_space"] = {}
    space_["obs_space"]["state_shape"] = 8
    space_["obs_space"]["obs_shape"] = obs_dim
    space_["act_space"]["n_actions"] = 8
    space_["obs_space"] = str(space_["obs_space"])
    space_["act_space"] = str(space_["act_space"])
    return space_

"""
    Not used anymore
"""
class Policy_shift_observer(object):
    def __init__(self, act_range, act_num):
        self.act_range = act_range  # 15
        self.act_num = act_num  # 3
        self.act_cnt_array = np.zeros(shape=(act_num, act_range))
        self.rate = None
        self.rate_history = None

    def new_sample(self, act):
        act_rec = act.shape[0]
        for act_index in range(self.act_num):
            for act_nth in range(self.act_range):
                self.act_cnt_array[act_index, act_nth] = torch.sum(
                    (act[:, act_index] == act_nth).long()
                )
        self.rate = self.act_cnt_array / act_rec
        if self.rate_history is None:
            self.rate_history = self.rate
        else:
            self.rate_history = self.rate_history * 0.9 + self.rate * 0.1
        print("rate", self.rate)
        # conclusion: the action distribution is not reinforced because the rewards signal is too weak.


"""
    Get the hash code string of an array, 
    compatable for numpy array and torch.tensor
"""
def __hash__(x):
    import hashlib
    md5 = hashlib.md5()  # ignore
    # if isinstance(x, str):
    #     md5.update(x)
    #     return md5.hexdigest()
    if hasattr(x, "cpu"):
        md5.update(x.detach().cpu().numpy().data.tobytes())
        return md5.hexdigest()
    elif hasattr(x, "numpy"):
        md5.update(x.numpy().data.tobytes())
        return md5.hexdigest()
    elif hasattr(x, "data"):
        md5.update(x.data.tobytes())
        return md5.hexdigest()
    else:
        try:
            md5.update(x.encode("utf-8"))
            return md5.hexdigest()
        except:
            return str(x)


def __hashm__(*args):
    import hashlib

    md5 = hashlib.md5()  # ignore
    for arg in args:
        x = arg
        if hasattr(x, "cpu"):
            md5.update(x.detach().cpu().numpy().data.tobytes())
        elif hasattr(x, "numpy"):
            md5.update(x.numpy().data.tobytes())
        elif hasattr(x, "data"):
            md5.update(x.data.tobytes())
        else:
            try:
                md5.update(x.encode("utf-8"))
            except:
                md5.update(str(x).encode("utf-8"))
    return md5.hexdigest()


"""
    Get the hash code string of the pytorch network parameters
    eg. 
        __hashn__(mlp_module.parameters())
"""
def __hashn__(generator):
    import hashlib

    md5 = hashlib.md5()  # ignore
    for arg in generator:
        x = arg.data
        if hasattr(x, "cpu"):
            md5.update(x.detach().cpu().numpy().data.tobytes())
        elif hasattr(x, "numpy"):
            md5.update(x.numpy().data.tobytes())
        elif hasattr(x, "data"):
            md5.update(x.data.tobytes())
        else:
            try:
                md5.update(x.encode("utf-8"))
            except:
                md5.update(str(x).encode("utf-8"))
    return md5.hexdigest()


"""
    numpy version of softmax
"""


def np_softmax(x, axis=None):
    # compute in log space for numerical stability
    return np.exp(x - logsumexp(x, axis=axis, keepdims=True))


"""
    numpy version of logsumexp
"""


def logsumexp(a, axis=None, keepdims=False, return_sign=False):
    a_max = np.amax(a, axis=axis, keepdims=True)
    if a_max.ndim > 0:
        a_max[~np.isfinite(a_max)] = 0
    elif not np.isfinite(a_max):
        a_max = 0
    tmp = np.exp(a - a_max)
    # suppress warnings about log of zero
    with np.errstate(divide="ignore"):
        s = np.sum(tmp, axis=axis, keepdims=keepdims)
        if return_sign:
            sgn = np.sign(s)
            s *= sgn  # /= makes more sense but we need zero -> zero
        out = np.log(s)
    if not keepdims:
        a_max = np.squeeze(a_max, axis=axis)
    out += a_max
    if return_sign:
        return out, sgn
    else:
        return out


"""
函数说明：在有限的、不均衡的多标签数据集中，按照预设的比例，取出尽可能多的样本
"""


def sample_balance(x, y, n_class, weight=None):
    if weight is None:
        weight = torch.ones(n_class, device=x.device)
    else:
        weight = torch.Tensor(weight).to(x.device)
    n_instance = torch.zeros(n_class, device=x.device)
    indices = [None] * n_class
    for i in range(n_class):
        indices[i] = torch.where(y == i)[0]
        n_instance[i] = len(indices[i])
    ratio = n_instance / weight
    bottle_neck = torch.argmin(n_instance / weight)
    r = ratio[bottle_neck]
    n_sample = (r * weight).long()
    # print(n_instance, n_sample)
    new_indices = [indices[i][torch.randperm(n_sample[i])] for i in range(n_class)]
    # print(new_indices)
    new_indices_ = torch.cat(new_indices)
    assert len(new_indices_) == sum(n_sample)
    return x[new_indices_], y[new_indices_]


"""
    gather tensor with index, 
    regarding all right hand dimensions as dimensions need to be gathered
    eg.1
        src = torch.Tensor([[[ 0,  1,  2], [ 3,  4,  5]],
                            [[ 6,  7,  8], [ 9, 10, 11]],
                            [[12, 13, 14], [15, 16, 17]]])
        index = torch.Tensor([[0], [1], [0]])
        src.shape = (3, 2, 3)
        src.shape = (3, 1)
        >> res = gather_righthand(src,index)
        res.shape = (3, 1, 3)
        res= tensor([[[ 0.,  1.,  2.]],
                     [[ 9., 10., 11.]],
                     [[12., 13., 14.]]])
    eg.2
        src.shape   = (64, 16, 8, 88, 888)
        index.shape = (64, 5)
        >> res = gather_righthand(src,index)
        res.shape   = (64, 5,  8, 88, 888)

    eg.3
        src.shape   = (64,  16,  88, 888)
        index.shape = (64, 777)
        >> res = gather_righthand(src,index)
        res.shape   = (64, 777,  88, 888)

"""
def gather_righthand(src, index, check=True):
    if not isinstance(src, torch.Tensor):
        return np_gather_righthand(src, index, check)
    index = index.long()
    i_dim = index.dim()
    s_dim = src.dim()
    t_dim = i_dim - 1
    if check:
        assert s_dim >= i_dim
        assert index.max() <= src.shape[t_dim] - 1
        if index.max() != src.shape[t_dim] - 1:
            print(
                "[gather_righthand] warning, index max value does not match src target dim"
            )
        assert (
            src.shape[t_dim] != index.shape[t_dim]
        ), "Do you really want to select %d item out of %d?? If so, please set check=False." % (
            index.shape[t_dim],
            src.shape[t_dim],
        )
        for d in range(0, t_dim):
            assert src.shape[d] == index.shape[d]
    index_new_shape = list(src.shape)
    index_new_shape[t_dim] = index.shape[t_dim]
    for _ in range(i_dim, s_dim):
        index = index.unsqueeze(-1)
    index_expand = index.expand(index_new_shape)  # only this two line matters
    return torch.gather(
        src, dim=t_dim, index=index_expand
    )  # only this two line matters




"""
    numpy version of 'gather_righthand'
"""
def np_gather_righthand(src, index, check=True):
    index = index.astype(np.long)
    dim = lambda x: len(x.shape)
    i_dim = dim(index)
    s_dim = dim(src)
    t_dim = i_dim - 1
    if check:
        assert s_dim >= i_dim
        assert index.max() <= src.shape[t_dim] - 1, ("\tindex.max()=", index.max(), "\tsrc.shape[t_dim]-1=", src.shape[t_dim] - 1)
        if index.max() != src.shape[t_dim] - 1:
            print(
                "[gather_righthand] warning, index max value does not match src target dim"
            )
        assert (
            src.shape[t_dim] != index.shape[t_dim]
        ), "you really want to select %d item out of %d?" % (
            index.shape[t_dim],
            src.shape[t_dim],
        )
        for d in range(0, t_dim):
            assert src.shape[d] == index.shape[d]
    tile_shape = np.array(src.shape)  # warning: careful when moving to pytorch
    tile_shape[: (t_dim + 1)] = 1
    for _ in range(i_dim, s_dim):
        index = np.expand_dims(index, -1)
    index_expand = np.tile(
        index, tile_shape
    )  # index.expand(index_new_shape)            # only this two line matters
    return np.take_along_axis(arr=src, indices=index_expand, axis=t_dim)
    # return torch.gather(src, dim=t_dim, index=index_expand) # only this two line matters

"""
    reverse operation of 'gather_righthand'
"""
def scatter_righthand(scatter_into, src, index, check=True):
    index = index.long()
    i_dim = index.dim()
    s_dim = src.dim()
    t_dim = i_dim - 1
    index_new_shape = list(src.shape)
    index_new_shape[t_dim] = index.shape[t_dim]
    for _ in range(i_dim, s_dim):
        index = index.unsqueeze(-1)
    index_expand = index.expand(index_new_shape)  # only this two line matters
    return scatter_into.scatter(t_dim, index_expand, src)


"""
    calculate distance matrix between two position vector array A and B, support 3d and 2d
    test >>
    A = np.array([  [0,0],
                    [1,1],])
    B = np.array([  [0,-1],
                    [1, 0],
                    [0, 1],])
    distance_mat_between(A, B) == [ [ 1 1 1  ], [sqrt(5), 1, 1 ]] => shape = (2,3)
"""
def distance_mat_between(A, B):
    n_subject_a = A.shape[-2]  # A (64, 3)
    n_subject_b = B.shape[-2]  # B (28, 3)
    A = np.repeat(np.expand_dims(A, -2), n_subject_b, axis=-2)  # =>(64, 28, 3)
    B = np.repeat(np.expand_dims(B, -2), n_subject_a, axis=-2)  # =>(28, 64, 3)
    B = np.swapaxes(B, -2, -3)  # =>(64, 28, 3)
    dis = A - B  # =>(64, 100, 100, 2)
    dis = np.linalg.norm(dis, axis=-1)
    return dis


"""
    calculate distance matrix for a position vector array A, support 3d and 2d
"""
def distance_matrix(A):
    n_subject = A.shape[-2]  # is 2
    A = np.repeat(np.expand_dims(A, -2), n_subject, axis=-2)  # =>(64, 100, 100, 2)
    At = np.swapaxes(A, -2, -3)  # =>(64, 100, 100, 2)
    dis = At - A  # =>(64, 100, 100, 2)
    dis = np.linalg.norm(dis, axis=-1)
    return dis

"""
    calculate delta matrix for a position vector array A
"""
def delta_matrix(A):
    n_subject = A.shape[-2]  # is 2
    A = np.repeat(np.expand_dims(A, -2), n_subject, axis=-2)  # =>(64, 100, 100, 2)
    At = np.swapaxes(A, -2, -3)  # =>(64, 100, 100, 2)
    delta = At - A  # =>(64, 100, 100, 2)
    return delta

def np_normalize_last_dim(mat):
    return mat / np.expand_dims(np.linalg.norm(mat, axis=-1) + 1e-16, axis=-1)

def dir2rad_old(delta_pos):
    result = np.empty(delta_pos.shape[:-1], dtype=complex)
    result.real = delta_pos[..., 0]
    result.imag = delta_pos[..., 1]
    rad_angle = np.angle(result)
    # assert (dir2rad_new(delta_pos)==rad_angle).all()
    return rad_angle

"""
    arctan2, but support any batch
"""
def dir2rad(delta_pos):
    return np.arctan2(delta_pos[..., 1], delta_pos[..., 0])


def dir3d_rad(delta_pos):
    assert delta_pos.shape[-1]==3
    xy = delta_pos[..., :2]
    r1 = dir2rad(xy)
    xy_norm = np.linalg.norm(xy, axis=-1)
    r2 = dir2rad(np.stack((xy_norm, delta_pos[..., 2]),-1))
    return np.stack((r1,r2), axis=-1)
    

def reg_deg(deg):
    return (deg + 180) % 360 - 180

# make angles comparable
def reg_deg_at(rad, ref):
    return reg_deg(rad-ref) + ref

def reg_rad(rad):
    # it's OK to show "RuntimeWarning: invalid value encountered in remainder"
    return (rad + np.pi) % (2 * np.pi) - np.pi

# make angles comparable
def reg_rad_at(rad, ref):
    return reg_rad(rad-ref) + ref

# the average of two angles (in rad)
def avg_rad(rad1, rad2):
    return reg_rad_at(rad1, rad2)/2 + rad2/2

def zeros_like_except_dim(array, except_dim, n):
    shape_ = list(array.shape)
    shape_[except_dim] = n
    return torch.zeros(size=shape_, device=array.device, dtype=array.dtype)


def pad_at_dim(array, dim, n):
    extra_n = n-array.shape[dim]
    padding = zeros_like_except_dim(array, except_dim=dim, n=extra_n)
    return torch.cat((array, padding), axis=dim)

def stack_vec_with_padding(arr_list):
    _len = [arr.len() for arr in arr_list]
    max_len = max(_len)
    n_subject = arr_list.len()
    dtype = arr_list[0].dtype
    arr_np = np.zeros(shape=(n_subject, max_len), dtype=dtype)
    for i, arr in enumerate(arr_list):
        arr_np[i,:_len[i]] = arr
    return arr_np


def objdump(obj):
    import pickle
    with open('objdump.tmp', 'wb+') as f:
        pickle.dump(obj, f)
    return

def objload():
    import pickle, os
    if not os.path.exists('objdump.tmp'): 
        return
    with open('objdump.tmp', 'rb') as f:
        return pickle.load(f)

def stack_padding(l, padding=np.nan):
    max_len = max([t.shape[0] for t in l])
    shape_desired = (len(l), max_len, *(l[0].shape[1:]))
    target = np.zeros(shape=shape_desired, dtype=float) + padding
    for i in range(len(l)): target[i, :len(l[i])] = l[i]
    return target

def n_item(tensor):
    n = 1
    for d in tensor.shape:
        n = n*d
    return n
