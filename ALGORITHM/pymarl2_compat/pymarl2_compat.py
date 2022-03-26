from scipy.optimize import linear_sum_assignment
import numpy as np
import copy
import time
import random
import redis, pickle
import subprocess
import json
import os
# from subprocess import DEVNULL
from UTILS.colorful import print亮紫
from UTILS.hidden_print import HiddenPrints
from config import GlobalConfig

class AlgorithmConfig():
    load_checkpoint = False
    episode_limit = 400 # int(100e3)
    batch_size = 2 # Number of episodes to train on
    pymarl_config_injection = {}

def 加密字符串(s):
    k = ''.join(['@']*1000)
    encry_str = ""
    for i,j in zip(s,k):
        # i为字符，j为秘钥字符
        temp = str(ord(i)+ord(j))+'_'
        encry_str = encry_str + temp
    return encry_str

def 解密字符串(p):
    k = ''.join(['@']*1000)
    dec_str = ""
    for i,j in zip(p.split("_")[:-1],k):
        # i 为加密字符，j为秘钥字符
        temp = chr(int(i) - ord(j))
        dec_str = dec_str+temp
    return dec_str


class PymarlFoundation():
    def init_pymarl(self):
        fp = open('%s/pymarl.log'%GlobalConfig.logdir, 'w+')
        import uuid, atexit
        self.remote_uuid = uuid.uuid1().hex   # use uuid to identify threads
        # If code fails here, please install redis-server on ubuntu host (outside the docker container)
        self.redis = redis.Redis(host='127.0.0.1', port=6379)
        # self.redis.delete()
        # add basic
        AlgorithmConfig.pymarl_config_injection['config.py->GlobalConfig'] = {
            'HmpRoot': os.getcwd(),
            'ExpNote': GlobalConfig.note,
            'draw_mode': GlobalConfig.draw_mode,
            'logdir': GlobalConfig.logdir,
            'seed': GlobalConfig.seed,
            'activate_logger': GlobalConfig.activate_logger,
            'train_time_testing': GlobalConfig.train_time_testing,
            'test_interval': GlobalConfig.test_interval,
            'test_only': GlobalConfig.test_only,
            'test_epoch': GlobalConfig.test_epoch,
        }

        subprocess.Popen(["python", 
            "/home/fuqingxu/pymarl2/pymarl2src/main.py", 
            "--force", 
            "--config=qmix", 
            "--env-config=HMP_compat",
            "with",
            "pymarl_config_injection=%s"%加密字符串(json.dumps(AlgorithmConfig.pymarl_config_injection)),  
            "batch_size_run=%d"%self.n_thread,
            "batch_size=%d"%AlgorithmConfig.batch_size,
            "env_args.env_uuid=%s"%self.remote_uuid], stdout=fp, stderr=fp)
            # "env_args.env_uuid=%s"%self.remote_uuid]) #, stdout=fp, stderr=fp)
        
        from UTILS.network import UnixUdpServer
        unix_path = 'RECYCLE/Sockets/unix/%s'%self.remote_uuid
        self.remote_link_server = UnixUdpServer(unix_path, obj='pickle')
        atexit.register(lambda: self.__del__()) # avoid redis leaking
        time.sleep(5)

    def __del__(self):
        print('PymarlFoundation end, cleaning redis')
        # self.shared_memory.close()
        self.redis.delete('>>hmp%s'%self.remote_uuid)
        for uuid, which_env in self.uuid2threads.items():
            self.redis.delete('<<hmp%s'%uuid)

    def __init__(self, n_agent, n_thread, space, mcv):
        self.n_thread = n_thread
        self.n_agent = n_agent
        self.handler = [None for _  in range(self.n_thread)]
        self.remote_pymarl_start_cmd = ""
        self.remote_pymarl_interface = ""
        self.team_intel = None
        self.space = space
        self.uuid2threads = {'thread_cnt':0}
        self.current_actions = [None for _ in range(self.n_thread)]
        self.previous_action = None
        self.previous_ENV_PAUSE = None
        self.register_step_call = [False for _ in range(self.n_thread)]
        self.scenario_config = GlobalConfig.scenario_config

        self.init_pymarl()

        # missing :{'battle_won': False, 'dead_allies': 6, 'dead_enemies': 0}

    def basic_io(self):
        _, buf = self.redis.brpop('>>hmp%s'%self.remote_uuid)
        cmd_arg = pickle.loads(buf)
        cmd, args, uuid = cmd_arg
        self.current_uuid = uuid
        res = getattr(self, cmd)(*args)
        if cmd=='step_of': # only step function need a delay
            pass
        elif cmd=='close':
            raise ReferenceError
        else:
            self.redis.lpush('<<hmp%s'%uuid, pickle.dumps(res))
    
    def step_callback_pymarl(self):
        for uuid, which_env in self.uuid2threads.items():
            if uuid == 'thread_cnt': continue
            if not self.register_step_call[which_env]: continue
            self.register_step_call[which_env] = False

            reward = self.team_intel['Latest-Reward'][which_env]
            terminated = self.team_intel['Env-Suffered-Reset'][which_env]
            env_info = self.team_intel['Latest-Team-Info'][which_env].copy()
            for key in ['obs-echo','state-echo','state','avail-act-echo','avail-act']:
                if key in env_info: env_info.pop(key)
            env_info['testing'] = self.team_intel['Test-Flag']
            res = (reward, terminated, env_info)
            self.redis.lpush('<<hmp%s'%uuid, pickle.dumps(res))

    def get_current_mode(self):
        return 'Testing' if self.team_intel['Test-Flag'] else 'Training'

    # @basic_io_call
    def step_of(self, act):
        which_env = self.get_env_with_currentuuid()
        self.current_actions[which_env] = act
        self.register_step_call[which_env] = True

    # @basic_io_call
    def get_state_size(self):
        try:
            return self.space['obs_space']['state_shape']
        except:
            info = self.team_intel['Latest-Team-Info'][0]   # the info of environment 0
            if 'state' not in info:
                return 0
            else:
                return info['state'].shape[-1]


    # @basic_io_call
    def get_obs_size(self):
        try:
            return self.space['obs_space']['obs_shape']
        except:
            obs = self.team_intel['Latest-Obs'][0]   # the info of environment 0
            return obs.shape[-1]

    # @basic_io_call
    def get_n_agents(self):
        try:
            self.n_agent = self.space['act_space']['n_agents']
            return self.n_agent
        except:
            return self.n_agent

    # @basic_io_call
    def get_episode_limit(self):
        return AlgorithmConfig.episode_limit

    # @basic_io_call
    def get_total_actions(self):
        try:
            self.n_actions = self.space['act_space']['n_actions']
            return self.space['act_space']['n_actions']
        except:
            assert self.scenario_config.use_simple_action_space
            self.n_actions = self.scenario_config.n_actions
            return self.scenario_config.n_actions # self.scenario_config.str_action_description

    # @basic_io_call
    def confirm_reset(self):
        # reset 函数在 parallel_runner.py中调用
        return True

    # @basic_io_call
    def close(self):
        # reset 函数在 parallel_runner.py中调用
        return

    # @basic_io_call
    def get_stats_of(self):
        which_env = self.get_env_with_currentuuid()
        env_info = self.team_intel['Latest-Team-Info'][which_env].copy()
        for key in ['obs-echo','state-echo','state','avail-act-echo','avail-act']:
            if key in env_info: env_info.pop(key)
        return env_info

    # @basic_io_call
    def get_state_of(self):
        which_env = self.get_env_with_currentuuid()
        if self.team_intel['Env-Suffered-Reset'][which_env]: 
            if (not self.team_intel['ENV-PAUSE'][which_env]):   # not paused, it is being unfrozen, or everything just init
                return self.team_intel['Latest-Team-Info'][which_env]['state']  # return newest state
            else:   # If the env is being frozen, return old state
                return self.team_intel['Latest-Team-Info'][which_env]['state-echo'] # return state echo
        # otherwise, normal situations
        return self.team_intel['Latest-Team-Info'][which_env]['state'] # return newest state
        # which_env = self.get_env_with_currentuuid()
        # return self.team_intel['Latest-Team-Info'][which_env]['state']

    # @basic_io_call
    def get_avail_actions_of(self):
        which_env = self.get_env_with_currentuuid()
        if 'avail-act' in self.team_intel['Latest-Team-Info'][which_env]:
            return self.team_intel['Latest-Team-Info'][which_env]['avail-act']
        else:
            return np.ones((self.n_agent, self.n_actions))

    # @basic_io_call
    def get_obs_of(self):
        which_env = self.get_env_with_currentuuid()
        if self.team_intel['Env-Suffered-Reset'][which_env]: 
            if (not self.team_intel['ENV-PAUSE'][which_env]):   # not paused, it is being unfrozen, or everything just init
                return self.team_intel['Latest-Obs'][which_env]
            else:   # If the env is being frozen, return old state
                return self.team_intel['Terminal-Obs-Echo'][which_env]
        # otherwise, normal situations
        return self.team_intel['Latest-Obs'][which_env]

    def deal_with_pymarl(self):
        # print('deal_with_pymarl')
        while any([act is None for act in self.current_actions]):
            self.basic_io()
            # print('basic_io fin')

    def clear_actions(self):
        self.current_actions = [None for i in range(self.n_thread)]
        for ith, paused in enumerate(self.team_intel['ENV-PAUSE']):
            if paused: 
                assert self.previous_action[ith] is not None
                self.current_actions[ith] = self.previous_action[ith]+np.nan


    def interact_with_env(self, team_intel):
        self.team_intel = team_intel
        # print亮紫(self.team_intel['ENV-PAUSE'])

        # finish previous step call
        self.step_callback_pymarl() 
        # check step_call register
        assert not any(self.register_step_call)

        # clear all actions, set 'NaN' action for Paused threads, note that 'NaN' differs from 'None'!
        self.clear_actions()
        self.deal_with_pymarl()

        # info = team_intel['Latest-Team-Info']
        # done = team_intel['Env-Suffered-Reset']
        # step_cnt = team_intel['Current-Obs-Step']
        self.previous_action = np.array(self.current_actions)
        self.previous_ENV_PAUSE = copy.deepcopy(team_intel['ENV-PAUSE'])
        ret_action_list = np.swapaxes(np.array(self.current_actions),0,1)
        # action_list = np.zeros(shape=(self.n_agent, self.n_thread, 1))
        return ret_action_list, team_intel



    def get_env_with_currentuuid(self):
        # mapping uuid to which_env
        uuid = self.current_uuid
        if uuid not in self.uuid2threads:
            self.uuid2threads[uuid] = self.uuid2threads['thread_cnt']
            self.uuid2threads['thread_cnt'] += 1
            assert self.uuid2threads['thread_cnt'] <= self.n_thread, ('accessed by too many threads, check setting!')
        return self.uuid2threads[uuid]