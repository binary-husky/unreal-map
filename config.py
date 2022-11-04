import time
import numpy as np
from UTIL.colorful import *
from UTIL.config_args import ChainVar



'''
    GlobalConfig: This config class will be 'injected' with new settings from JSONC.
    (E.g., override configs with ```python main.py --cfg example.jsonc```)
    (As the name indicated, ChainVars will change WITH vars it 'chained_with' during config injection)
    (please see UTIL.config_args to find out how this advanced trick works out.)

    * Explaining a very important setting option: 
        - align_episode (True/False):
            In align mode, all threads begin new episode synchronously,
            which means a env thread that ends early has to 'wait' for other threads before restarting
            If set to 'False', threads will not wait for each others, and will 'reset' immediately on 'done'

        - note (str):
            Name you experiment carefully with note setting.
            The note defines where the results of a single experiment will go. 
            for example, if note='conc', everything produced in the experiment will be save in ZHECKPOINT/conc/*,
            including images, saved pytorch model, 
        
        - env_name:
            Which mission/environment/task to use, 
            See ./MISSION/env_router.py for the dictionary of available envs.

        - env_path:
            The path of selected mission. In fact, hmp do not need this setting at all,
            it exists here only to Double Check that you have chosen the correct mission env.

    * Why the Algorithm selection is not here ?!
        It is the missions (envs) that choose the algorithm(s)! Please go to mission configuration!
        Important to remember:
            - hmp selects a mission, 
            - mission selects algorithm(s).
        In fact, if you have two teams in env, 
        you can choose two different algorithms to fight each other in the same env!
            - Please goto ./MISSION/env_router.py to find out where the ScenarioConfig of your env is written,
            - Please set ```TEAM_NAMES``` to include the path of your favored algorithm(s)
'''


class GlobalConfig(object): # ADD_TO_CONF_SYSTEM //DO NOT remove this comment//
    align_episode = True                                # ! please try to understand this with TOP priority

    env_name = 'sr_tasks->cargo'                        # which environment, see ./MISSION/env_router.py
    env_path = 'MISSION.sr_tasks.multiagent.cargo'     # path of environment
    draw_mode = 'OFF'                                   # 'Web','Native','Img','Threejs' 
    activate_logger = True                              # activate data plotting (Tensorboard is not used because I do not like it)
    data_logger = 'auto load, do not change this var!'  # activate data plotting (Tensorboard is not used because I do not like it)
    resume_mod = False                                  # resume unfinished

    seed = np.random.randint(0, 100000)                 # seed for numpy and pytorch

    # ! warning, the note also determine where the experiment log is stored, typically at ./ZHECKPOINT/$note/*
    note = 'more_testing'                               # in case you forget the purpose of this trainning session, write a note
    logdir = './ZHECKPOINT/%s/'%note
    logdir_cv = ChainVar(lambda note: './ZHECKPOINT/%s/'%note, chained_with=['note']) 
    recall_previous_session = False                     # continue previously interrupted training session

    test_only = False                                   # only testing and no training, it controlls a flag sending to Alg side
    test_logger = 'test_only_profile.txt'               # logger path, experimental, writing win rate in a file

    device = 'cuda'                                     # choose from 'cpu' (no GPU), 'cuda' (auto select GPU), 'cuda:3' (manual select GPU)
    gpu_party = 'off'                                   # GPU memory is precious! assign multiple training process to a 'party', they will share GPU memory
    manual_gpu_ctl = False                              # auto variable, do not alter! do not change!
    gpu_fraction = 1.0

    num_threads = 64                                    # run N parallel envs, a 'env' is refered to as a 'thread'
    fold = 1                                            # A 'linux process' can handle multiple envs ('thread'), run N parallel envs, on (N//fold) processes
                                                        # this 'folding' is designed for IPC efficiency, you can thank python GIL for such a strange design... 
    
    n_parallel_frame = int(5e6)                         # Number of frames to run (in each frame, all parallel-envs step once)
    max_n_episode = int(2e5)                            # max number of episodes

    use_float64 = False                                 # force float64 when converting numpy->tensor

    interested_team = 0                                 # the interested agents, used in reward recording
    interested_agent_num = 50                           # the interested agents, used in reward recording
    interested_agent_uid = range(0,50)                  # the indices of interested agents, used in reward recording
    interested_agent_uid_cv = ChainVar(lambda interested_agent_num:range(0,interested_agent_num), chained_with=['interested_agent_num']) 

    report_reward_interval = num_threads                # reporting interval
    report_reward_interval_cv = ChainVar(lambda num_threads:num_threads, chained_with=['num_threads'])

    train_time_testing = True                           # allow hmp to test algorithm policies every test_interval episodes
    test_interval = 32*num_threads                      # test interval
    test_interval_cv = ChainVar(lambda num_threads:4*num_threads, chained_with=['num_threads'])

    test_epoch = 32 if num_threads <= 32 else num_threads    # test epoch
    test_epoch_cv = ChainVar(lambda num_threads: 32 if num_threads <= 32 else num_threads, chained_with=['num_threads'])

    ScenarioConfig = 'This ScenarioConfig var will be automatically linked to task configuration later in ./MISSION/env_router.py'


    backup_files = []                                   # a list of files that needs to be backed up at each run
    matlab_logger = None                                # this is just a global logger, although we do not use matlab anymore...
    heartbeat_on = True                                 # some fancy commandline visual effect to show that envirenment is running
    
    cfg_ready = False                                   # DO NOT change! automatically set to True when Json configuration is all locked-and-loaded

    # ! uploading "./ZHECKPOINT/$note" to a data storage server 
    allow_res_upload = True                             # upload results to a data storage server when exiting
    upload_after_test = False                           # upload results to a data storage server when completing a test run
    machine_info = 'auto load, do not change this var!'
    remote_server_ops = ""
    # KEY = {"addr": None, "usr":None, "pwd":None}        # data storage server ip addr, username and password