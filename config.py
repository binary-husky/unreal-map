import time
import numpy as np
from UTILS.colorful import *

class ChainVar(object):
    def __init__(self, chain_func, chained_with):
        self.chain_func = chain_func
        self.chained_with = chained_with

class GlobalConfig(object): # ADD_TO_CONF_SYSTEM //DO NOT remove this comment//

    '''__  config may changed when launched with json __'''

    # which environment, see ./MISSIONS/env_router.py
    env_name = 'sr_tasks->cargo' 
    env_path = 'MISSIONS.sr_tasks.multiagent.cargo' 
    draw_mode = 'OFF'  # 'Web','Native'
    activate_logger = True
    scale_creward = 1.0
    # seed
    seed = np.random.randint(0, 100000) 
    test_logger = 'test_only_profile.txt'

    # in case you forget the purpose of this trainning session, write a note
    note = 'more_testing'
    logdir = './ZHECKPOINT/%s/'%note
    logdir_cv = ChainVar(lambda note: './ZHECKPOINT/%s/'%note, chained_with=['note']) 

    test_logger = 'test_only_profile.txt'
    # use GPU or not
    device = 'cuda'
    # GPU memory is precious, 
    # assign multiple training process to a 'party', they will share GPU memory
    gpu_party = 'off'   
    manual_gpu_ctl = False  # auto variable, do not alter


    # run N parallel envs, a 'env' is refered to as a 'thread'
    num_threads = 64
    # A 'linux process' can handle multiple envs ('thread')
    # run N parallel envs, on N//fold processes, 
    fold = 1


    # Number of frames to run (in each frame, all parallel-envs step once)
    n_parallel_frame = int(5e6)
    max_n_episode = int(20e4) # 15 万episode


    # In align mode, all 'thread's begin new episode at the same time, 
    # a thread that ends early has to 'wait' for other threads before starting new
    # If 'False', threads will not 'wait' for others, 'reset' immediately on 'done'
    align_episode = True
    use_float64 = False

    # Matlab Plot Bridge switch on/off
    show_curlve = False

    interested_agent_uid = range(0,50)
    train_time_testing = True
    test_only = False

    # reporting interval
    report_reward_interval = num_threads
    report_reward_interval_cv = ChainVar(lambda num_threads:num_threads, chained_with=['num_threads'])
    # test interval
    test_interval = 32*num_threads
    test_interval_cv = ChainVar(lambda num_threads:4*num_threads, chained_with=['num_threads'])
    # test epoch
    test_epoch = 32 if num_threads <= 32 else num_threads
    test_epoch_cv = ChainVar(lambda num_threads: 32 if num_threads <= 32 else num_threads, chained_with=['num_threads'])


    # # # >>> ↓↓↓ (AUTO) define the location of ScenarioConfig
    scenario_config = 'scenario_config will be automatically linked to task configuration later in ./MISSIONS/env_router.py'


    backup_files = []
    matlab_logger = None
    recall_previous_session = False