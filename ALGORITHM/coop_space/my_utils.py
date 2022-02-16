import copy
from functools import wraps
import numpy as np
import torch
from numba import jit, njit


def normalize_obs(obs, mean, std):
    if mean is not None:
        return np.divide((obs - mean), std)
    else:
        return obs


def make_env(env_id, seed, rank, num_agents, dist_threshold, arena_size, identity_size):
    def _thunk():
        env = make_multiagent_env(env_id, num_agents, dist_threshold, arena_size, identity_size, rank)
        return env
    return _thunk


def make_multiagent_env(env_id, num_agents, dist_threshold, arena_size, identity_size, rank):
    scenario = scenarios.load(env_id+".py").Scenario(num_agents=num_agents, dist_threshold=dist_threshold,
                                                     arena_size=arena_size, identity_size=identity_size, process_id=rank)
    world = scenario.make_world()

    env = MultiAgentEnv(world=world,
                        reset_callback=scenario.reset_world,
                        reward_callback=scenario.reward,
                        observation_callback=scenario.observation,
                        info_callback=scenario.info if hasattr(scenario, 'info') else None,
                        discrete_action=True,
                        done_callback=scenario.done,
                        cam_range=arena_size
                        )
    return env


def make_parallel_envs(args):
    # make parallel environments
    env_args_dict_list = [({'args': args, 'proc_index': i},) for i in range(GlobalConfig.num_threads)]

    if GlobalConfig.num_threads > 1:
        envs = gym_vecenv.SubprocVecEnv(env_args_dict_list)
    else:
        envs = gym_vecenv.SubprocVecEnv(env_args_dict_list)

    envs = gym_vecenv.MultiAgentVecNormalize(envs, ob=False, ret=False)
    return envs


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def copy_clone(x):
    if x is None: return None
    return x.clone() if hasattr(x, 'clone') else x.copy() if hasattr(x, 'copy') else copy.deepcopy(x)


def my_view(x, shape):
    for i, dim in enumerate(shape):
        if dim == 0: shape[i] = x.shape[i]
    if isinstance(x, np.ndarray): return x.reshape(*shape)
    return x.view(*shape)



# def mat3d_normalize_each_line(mat):
#     return mat / torch.unsqueeze(torch.norm(mat, dim=2) + 1e-16, -1)

def add_onehot_id_at_last_dim(x):
    _hot_dim = x.shape[-2]
    _identity = np.tile(np.eye(_hot_dim), (*x.shape[:-2], 1, 1))
    return np.concatenate((x, _identity), -1)
    
def np_one_hot(x, n):
    return np.eye(n)[x]

@jit(forceobj=True)
def add_obs_container_subject(container_emb, subject_emb, div):
    # for subject, add one-hot embedding of its group
    n_container = container_emb.shape[1]
    subject_belonging_info = np_one_hot(div, n_container)
    subject_out_emb = np.concatenate((subject_emb, subject_belonging_info), -1)
    # for container, add add multi-hot embedding of its subjects
    container_multihot = np.concatenate(
        [np.expand_dims((div == nth_container).astype(np.long), 1) for nth_container in range(n_container)], 1)
    container_out_emb = np.concatenate((container_emb, container_multihot), -1)
    return container_out_emb, subject_out_emb



def args_to_np(f):
    def to_cpu_numpy(x):
        return None if x is None else x.cpu().numpy() if isinstance(x, torch.Tensor) \
               else x
    @wraps(f)
    def decorated(*args, **kwargs):
        for key in kwargs: kwargs[key] = to_cpu_numpy(kwargs[key])
        return f(*(to_cpu_numpy(arg) for arg in args), **kwargs)

    return decorated


def Args2tensor(f):
    def _2tensor(x):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        elif isinstance(x, np.ndarray):
            if x.dtype == np.float64: x = x.astype(np.float32)
            return torch.from_numpy(x).to(device)
        elif isinstance(x, dict):
            y = {}
            for key in x: y[key] = _2tensor(x[key])
            return y
        else:
            return x

    @wraps(f)
    def decorated(*args, **kwargs):
        for key in kwargs: kwargs[key] = _2tensor(kwargs[key])
        return f(*(_2tensor(arg) for arg in args), **kwargs)

    return decorated


def Args2tensor_Return2numpy(f):
    def _2tensor(x):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        elif isinstance(x, np.ndarray):
            if x.dtype == np.float64: x = x.astype(np.float32)
            return torch.from_numpy(x).to(device)
        elif isinstance(x, dict):
            y = {}
            for key in x: y[key] = _2tensor(x[key])
            return y
        else:
            return x

    def _2cpu2numpy(x):
        return None if x is None else x.cpu().numpy() if isinstance(x, torch.Tensor) \
            else x

    @wraps(f)
    def decorated(*args, **kwargs):
        for key in kwargs: kwargs[key] = _2tensor(kwargs[key])
        ret_tuple = f(*(_2tensor(arg) for arg in args), **kwargs)
        return (_2cpu2numpy(ret) for ret in ret_tuple)

    return decorated


def _2tensor(x):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, np.ndarray):
        if x.dtype == np.float64: x = x.astype(np.float32)
        return torch.from_numpy(x).to(device)
    elif isinstance(x, dict):
        y = {}
        for key in x: y[key] = _2tensor(x[key])
        return y
    else:
        return x


class Policy_shift_observer(object):
    def __init__(self, act_range, act_num):
        self.act_range = act_range # 15
        self.act_num = act_num # 3
        self.act_cnt_array = np.zeros(shape=(act_num,act_range))
        self.rate = None
        self.rate_history = None

    def new_sample(self, act):
        act_rec = act.shape[0]
        for act_index in range(self.act_num):
            for act_nth in range(self.act_range):
                self.act_cnt_array[act_index,act_nth] = torch.sum((act[:,act_index] == act_nth).long())
        self.rate = self.act_cnt_array/act_rec
        if self.rate_history is None: self.rate_history = self.rate
        else: self.rate_history = self.rate_history*0.9 + self.rate*0.1
        print('rate', self.rate)
        # conclusion: the action distribution is not reinforced because the rewards signal is too weak.