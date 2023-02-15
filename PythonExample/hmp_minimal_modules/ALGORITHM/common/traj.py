# cython: language_level=3
import numpy as np
from UTIL.colorful import *
from UTIL.tensor_ops import __hash__

class TRAJ_BASE():
    key_data_type = {}
    key_data_shape = {}
    max_mem_length = -1

    def __init__(self, traj_limit, env_id):
        self.traj_limit = traj_limit
        self.env_id = env_id
        self.readonly_lock = False
        self.key_dict = []
        self.time_pointer = 0
        self.need_reward_bootstrap = False
        self.deprecated_flag = False

    # remember something in a time step, add it to trajectory
    def remember(self, key, content):
        assert not self.readonly_lock
        if not (key in self.key_dict) and (content is not None):
            self.init_track(key=key, first_content=content)
            getattr(self, key)[self.time_pointer] = content
        elif not (key in self.key_dict) and (content is None):
            self.init_track_none(key=key)
        elif (key in self.key_dict) and (content is not None):
            getattr(self, key)[self.time_pointer] = content
        else:
            pass
    
    # duplicate/rename a trajectory
    def copy_track(self, origin_key, new_key):
        if hasattr(self, origin_key):
            origin_handle = getattr(self, origin_key)
            setattr(self, new_key, origin_handle.copy())
            new_handle = getattr(self, new_key)
            self.key_dict.append(new_key)
            #return origin_handle, new_handle
        else:
            real_key_list = [real_key for real_key in self.__dict__ if (origin_key+'>' in real_key)]
            assert len(real_key_list)>0, ('this key does not exist (yet), check:', origin_key)
            for real_key in real_key_list:
                mainkey, subkey = real_key.split('>')
                self.copy_track(real_key, (new_key+'>'+subkey))
            #return

    # make sure dtype is ok
    def check_type_shape(self, key, first_content=None):
        if first_content is not None:
            content_type = first_content.dtype
            content_shape = first_content.shape
            if key in TRAJ_BASE.key_data_type: 
                assert TRAJ_BASE.key_data_type[key] == content_type
            else: 
                TRAJ_BASE.key_data_type[key] = content_type
                TRAJ_BASE.key_data_shape[key] = content_shape
            return content_type, content_shape
        assert key in TRAJ_BASE.key_data_type
        return TRAJ_BASE.key_data_type[key], TRAJ_BASE.key_data_shape[key]

    # create track, executed used when a key showing up for the first time in 'self.remember'
    def init_track(self, key, first_content):
        content = first_content
        self.check_type_shape(key, first_content)
        assert isinstance(content, np.ndarray) or isinstance(content, float), (key, content.__class__)
        tensor_size = ((self.traj_limit,) + tuple(content.shape))
        set_item = np.zeros(shape=tensor_size, dtype=content.dtype)
        set_item[:] = np.nan  if np.issubdtype(content.dtype, np.floating) else 0
        setattr(self, key, set_item)
        self.key_dict.append(key)

    # key pop up yet content is None, 
    # read dtype from history dtype dictionary to fill the hole
    def init_track_none(self, key):
        content_dtype, content_shape = self.check_type_shape(key)
        tensor_size = ((self.traj_limit,) + tuple(content_shape))
        set_item = np.zeros(shape=tensor_size, dtype=content_dtype)
        set_item[:] = np.nan  if np.issubdtype(content_dtype, np.floating) else 0
        setattr(self, key, set_item)
        self.key_dict.append(key)

    # push the time pointer forward, before you call 'self.remember' again to fill t+1 data
    def time_shift(self):
        assert self.time_pointer < self.traj_limit
        self.time_pointer += 1

    # cut trajectory tail, when the number of episode time step < traj_limit
    def cut_tail(self): 
        TJ = lambda key: getattr(self, key)
        self.readonly_lock = True
        n_frame = self.time_pointer
        # check is buffer size too big
        if n_frame > TRAJ_BASE.max_mem_length: 
            TRAJ_BASE.max_mem_length = n_frame
            print('max_mem_length：%d, traj_limit:%d'%(TRAJ_BASE.max_mem_length, self.traj_limit))
        # clip tail
        for key in self.key_dict: setattr(self, key, TJ(key)[:n_frame])