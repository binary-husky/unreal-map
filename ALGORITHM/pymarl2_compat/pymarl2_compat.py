import numpy as np
import copy
import time
import random
import pickle
import subprocess
import json
import os
from UTIL.hidden_print import HiddenPrints
from config import GlobalConfig

class AlgorithmConfig():
    use_shell = ''
    state_compat = 'pad'   # 'pad', 'obs_mean', 'obs_cat'
    pymarl_config_injection = {}

def 加密字符串(s):  # encrpt string
    k = ''.join(['@']*1000)
    encry_str = ""
    for i,j in zip(s,k):
        temp = str(ord(i)+ord(j))+'_'
        encry_str = encry_str + temp
    return encry_str

def 解密字符串(p): # decrpt string
    k = ''.join(['@']*1000)
    dec_str = ""
    for i,j in zip(p.split("_")[:-1],k):
        temp = chr(int(i) - ord(j))
        dec_str = dec_str+temp
    return dec_str

class PymarlFoundation():
    def init_pymarl(self):
        fp = open('%s/pymarl.log'%GlobalConfig.logdir, 'w+')
        import uuid, atexit
        self.remote_uuid = uuid.uuid1().hex   # use uuid to identify threads
        # add basic
        if 'config.py->GlobalConfig' not in AlgorithmConfig.pymarl_config_injection:
            AlgorithmConfig.pymarl_config_injection['config.py->GlobalConfig'] = {}
            
        AlgorithmConfig.pymarl_config_injection['config.py->GlobalConfig'].update({
            'HmpRoot': os.getcwd(),
            'ExpNote': GlobalConfig.note,
            'draw_mode': GlobalConfig.draw_mode,
            'logdir': GlobalConfig.logdir,
            'n_thread': GlobalConfig.num_threads,
            'seed': GlobalConfig.seed,
            'activate_logger': GlobalConfig.activate_logger,
            'train_time_testing': GlobalConfig.train_time_testing,
            'test_interval': GlobalConfig.test_interval,
            'test_only': GlobalConfig.test_only,
            'test_epoch': GlobalConfig.test_epoch,
        })

        subprocess.Popen(["python", 
            "/home/hmp/pymarl2/pymarl2src/main.py", 
            "--force", 
            "--config=qmix", 
            "--env-config=HMP_compat",
            "with",
            "pymarl_config_injection=%s"%加密字符串(json.dumps(AlgorithmConfig.pymarl_config_injection)),  
            # "batch_size_run=%d"%self.n_thread,
            # "batch_size=%d"%AlgorithmConfig.batch_size,
            "env_args.env_uuid=%s"%self.remote_uuid], stdout=fp, stderr=fp)
        
        from UTIL.network import UnixTcpServerP2P
        unix_path = 'TEMP/Sockets/unix/%s'%self.remote_uuid
        self.remote_link_server = UnixTcpServerP2P(unix_path, obj='pickle')
        atexit.register(lambda: self.__del__()) 
        time.sleep(5)

    def deal_with_pymarl(self):
        while any([act is None for act in self.current_actions]):
            dgram = self.remote_link_server.wait_next_dgram()
            cmd = dgram[0]
            args = dgram[1:]
            assert isinstance(cmd, str)
            res = getattr(self, cmd)(*args)
            self.remote_link_server.reply_last_client(res)



    def interact_with_env(self, team_intel):
        self.team_intel = team_intel
        self.current_actions = [None]*self.n_thread 

        self.deal_with_pymarl()

        self.previous_action = np.array(self.current_actions)
        self.previous_ENV_PAUSE = copy.deepcopy(team_intel['ENV-PAUSE'])
        ret_action_list = np.swapaxes(np.array(self.current_actions), 0, 1)
        return ret_action_list, team_intel


    def reset_confirm_all(self):
        assert self.team_intel['Env-Suffered-Reset'].all()
        res = [{
                    "state": self.get_state_of(i),
                    "avail_actions": self.get_avail_actions_of(i),
                    "obs": self.get_obs_of(i)
                } for i in range(self.n_thread)]
        return res

    def step_all(self, action, remote_active_flag):
        assert (remote_active_flag==~self.team_intel['ENV-PAUSE']).all()
        for i in range(self.n_thread):
            if self.team_intel['ENV-PAUSE'][i]: 
                assert action[i] is None
                self.current_actions[i] = np.array([np.nan]*self.n_agent)
            else:
                assert action[i] is not None
                self.current_actions[i] = action[i]

        if self.team_intel['Test-Flag']: self.remote_active_flag_T = remote_active_flag
        else: self.remote_active_flag = remote_active_flag
        return None # cannot reply here

    def get_step_future(self):
        res = []
        if self.team_intel['Test-Flag']: remote_active_flag = self.remote_active_flag_T
        else: remote_active_flag = self.remote_active_flag

        if remote_active_flag is None: 
            return 'entering_test_phase'
 
        for i, acive in enumerate(remote_active_flag):
            if not acive: continue
            env_info = self.team_intel['Latest-Team-Info'][i].copy()
            for key in ['obs-echo','state-echo','state','avail-act-echo','avail-act']:
                if key in env_info: env_info.pop(key)
            env_info['testing'] = self.team_intel['Test-Flag']

            res.append({
                "state": self.get_state_of(i), 
                "avail_actions": self.get_avail_actions_of(i),
                "obs": self.get_obs_of(i),
                "reward": self.team_intel['Latest-Reward'][i],
                "terminated": self.team_intel['Env-Suffered-Reset'][i],
                "info": env_info
            })

        if self.team_intel['Test-Flag']: self.remote_active_flag_T = None
        else: self.remote_active_flag_T = None
        return res

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.get_n_agents(), #self.get_n_agents(), #self.n_agents,
                    "episode_limit": self.get_episode_limit()}
        return env_info

    def __del__(self):
        print('PymarlFoundation end, cleaning')
        self.remote_link_server.__del__()

    def __init__(self, n_agent, n_thread, space, mcv, team=None):
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
        self.ScenarioConfig = GlobalConfig.ScenarioConfig
        self.init_pymarl()
        if AlgorithmConfig.use_shell != '':
            if AlgorithmConfig.use_shell == 'mini_shell_uhmap':
                from .mini_shell_uhmap import ShellEnv
                self.shell = ShellEnv(self, n_agent, n_thread, space, mcv, team)
            else:
                assert False, "unknown shell env"
            
    def get_current_mode(self):
        return 'Testing' if self.team_intel['Test-Flag'] else 'Training'

    # @basic_io_call
    def get_state_size(self):
        if AlgorithmConfig.state_compat == 'native':
            try:
                return self.space['obs_space']['state_shape']
            except:
                info = self.team_intel['Latest-Team-Info'][0]   # the info of environment 0
                if 'state' not in info:
                    return 0
                else:
                    return info['state'].shape[-1]
        else:
            return self.space['obs_space']['state_shape']
            

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
        return int(self.ScenarioConfig.MaxEpisodeStep*1.5) # AlgorithmConfig.episode_limit

    # @basic_io_call
    def get_total_actions(self):
        try:
            self.n_actions = self.space['act_space']['n_actions']
            return self.n_actions
        except:
            assert self.ScenarioConfig.use_simple_action_space
            self.n_actions = self.ScenarioConfig.n_actions
            return self.ScenarioConfig.n_actions # self.ScenarioConfig.str_action_description

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
    def get_state_of(self, which_env):
        if self.team_intel['Env-Suffered-Reset'][which_env]: 
            if (not self.team_intel['ENV-PAUSE'][which_env]):   # not paused, it is being unfrozen, or everything just init
                return self.team_intel['Latest-Team-Info'][which_env]['state']  # return newest state
            else:   # If the env is being frozen, return old state
                return self.team_intel['Latest-Team-Info'][which_env]['state-echo'] # return state echo
        # otherwise, normal situations
        return self.team_intel['Latest-Team-Info'][which_env]['state'] # return newest state

    # @basic_io_call
    def get_avail_actions_of(self, which_env):
        if 'avail-act' in self.team_intel['Latest-Team-Info'][which_env]:
            return self.team_intel['Latest-Team-Info'][which_env]['avail-act']
        else:
            return np.ones((self.n_agent, self.n_actions))

    # @basic_io_call
    def get_obs_of(self, which_env):
        if self.team_intel['Env-Suffered-Reset'][which_env]: 
            if (not self.team_intel['ENV-PAUSE'][which_env]):   # not paused, it is being unfrozen, or everything just init
                return self.team_intel['Latest-Obs'][which_env]
            else:   # If the env is being frozen, return old state
                return self.team_intel['Terminal-Obs-Echo'][which_env]
        # otherwise, normal situations
        return self.team_intel['Latest-Obs'][which_env]

    def clear_actions(self):
        self.current_actions = [None for i in range(self.n_thread)]
        for ith, paused in enumerate(self.team_intel['ENV-PAUSE']):
            if paused: 
                assert self.previous_action[ith] is not None
                self.current_actions[ith] = self.previous_action[ith]+np.nan

    def get_env_with_currentuuid(self):
        # mapping uuid to which_env
        uuid = self.current_uuid
        if uuid not in self.uuid2threads:
            self.uuid2threads[uuid] = self.uuid2threads['thread_cnt']
            self.uuid2threads['thread_cnt'] += 1
            assert self.uuid2threads['thread_cnt'] <= self.n_thread, ('accessed by too many threads, check setting!')
        return self.uuid2threads[uuid]