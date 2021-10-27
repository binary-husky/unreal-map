import numpy as np
import importlib


class MMPlatform(object):
    # supports centralized training of agents in a team
    def __init__(self, mcv, envs):
        from config import GlobalConfig
        self.n_t =       GlobalConfig.scenario_config.N_TEAM
        n_agents_each_t = GlobalConfig.scenario_config.N_AGENT_EACH_TEAM
        t_member_list =  GlobalConfig.scenario_config.AGENT_ID_EACH_TEAM
        self.t_name =    GlobalConfig.scenario_config.TEAM_NAMES
        self.align_episode = GlobalConfig.align_episode
        # env give reward of each team instead of agent
        self.RewardAsUnity = False
        if hasattr(GlobalConfig.scenario_config, 'RewardAsUnity'):
            self.RewardAsUnity = GlobalConfig.scenario_config.RewardAsUnity 
        self.ActAsUnity = False
        if hasattr(GlobalConfig.scenario_config, 'ActAsUnity'):
            self.ActAsUnity = GlobalConfig.scenario_config.ActAsUnity
        self.ObsAsUnity = False
        if hasattr(GlobalConfig.scenario_config, 'ObsAsUnity'):
            self.ObsAsUnity = GlobalConfig.scenario_config.ObsAsUnity

        self.t_member_list = t_member_list
        self.t_member = t_member_list
        self.n_thread = GlobalConfig.num_threads
        self.algo_foundations = []
        space = envs.get_space()

        for t in range(self.n_t):
            assert len(t_member_list[t]) == n_agents_each_t[t]
            assert '->' in self.t_name[t]
            module_, class_ = self.t_name[t].split('->')
            init_f = getattr(importlib.import_module(module_), class_)
            self.algo_foundations.append(
                init_f(n_agent=n_agents_each_t[t], n_thread=self.n_thread, space=space, mcv=mcv)
            )
        # self.L_RUNNING = None
        pass


    def act(self, runner_info):
        actions_list = []
        for t_name, t_members, algo_fdn, t_index in zip(self.t_name, self.t_member, self.algo_foundations, range(self.n_t)):
            # split info such as reward and observation
            _t_intel_ = self._split_intel(runner_info, t_members, t_name, t_index)
            # each t (controlled by their different algorithm) interacts with env and act
            _act_, _t_intel_ = algo_fdn.interact_with_env(_t_intel_)
            # concat actions of each agent
            assert _act_.shape[0]==len(t_members), ('number of actions differs number of agents!')
            append_op = actions_list.append if self.ActAsUnity else actions_list.extend; append_op(_act_)
            # loop back internal states registered in _t_intel_  (e.g._division_obs_)
            if _t_intel_ is None: continue
            for key in _t_intel_:
                if key.startswith('_') and key.endswith('_'): 
                    self._update_runner(runner_info, runner_info['ENV-PAUSE'], t_name, key, _t_intel_[key])
        pass

        # $n_agent(n_t if ActAsUnity).$n_thread --> $n_thread.$n_agent(n_t if ActAsUnity)
        actions_list = np.swapaxes(np.array(actions_list, dtype=np.double), 0, 1) 

        ENV_PAUSE = runner_info['ENV-PAUSE']
        #debug
        # assert (actions_list[ENV_PAUSE,:] == -1).all()
        if ENV_PAUSE.any() and self.align_episode:  actions_list[ENV_PAUSE,:] = np.nan
        return actions_list, runner_info

    def _update_runner(self, runner_info, ENV_PAUSE, t_name, key, content):
        u_key = t_name+key
        if (u_key in runner_info) and hasattr(content, '__len__') and\
                len(content)==self.n_thread and ENV_PAUSE.any():
            runner_info[u_key][~ENV_PAUSE] = content[~ENV_PAUSE]
            return
        runner_info[u_key] = content
        return


    def _split_intel(self, runner_info, t_members, t_name, t_index):
        RUNNING = ~runner_info['ENV-PAUSE']
        if runner_info['Latest-Team-Info'] is not None:
            assert isinstance(runner_info['Latest-Team-Info'][0], dict)
            Team_Info = runner_info['Latest-Team-Info']
            # if a env just ended ('Env-Suffered-Reset'), the final step obs can be acquired here
            ter_obs_echo = np.array([self.__split_obs_thread(Team_Info[thread_idx]['obs-echo'], t_index)
                            if done and ('obs-echo' in Team_Info[thread_idx]) else None 
                            for thread_idx, done in enumerate(runner_info['Env-Suffered-Reset'])], dtype=object)
        else:
            Team_Info = None
            ter_obs_echo = None

        o = self.__split_obs(runner_info['Latest-Obs'], t_index)
        reward = runner_info['Latest-Reward']
        # summary
        t_intel_basic = {
            'Team_Name':            t_name,
            'Latest-Obs':           o, 
            'Latest-Team-Info':     Team_Info,
            'Env-Suffered-Reset':   runner_info['Env-Suffered-Reset'],
            'Terminal-Obs-Echo':    ter_obs_echo, 
            'ENV-PAUSE':            runner_info['ENV-PAUSE'],
            'Test-Flag':            runner_info['Test-Flag'],
            'Latest-Reward':        reward[:, t_members] if not self.RewardAsUnity else reward[:, t_index],
            'Current-Obs-Step':     runner_info['Current-Obs-Step']
        }



        for key in runner_info:
            if not (t_name in key): continue
            # otherwise t_name in key
            s_key = key.replace(t_name, '')
            t_intel_basic[s_key] = runner_info[key]
            if not ('_hook_' in s_key): continue
            # otherwise deal with _hook_
            self.deal_with_hook(t_intel_basic[s_key], t_intel_basic)
            runner_info[key] = t_intel_basic[s_key] = None

        # t_intel_basic = self.filter_running(t_intel_basic, RUNNING)
        # self.L_RUNNING = RUNNING
        return t_intel_basic

    # def filter_running(self, intel_basic, running):
    #     intel_basic_ = intel_basic.copy()
    #     for key in intel_basic:
    #         if hasattr(intel_basic[key], '__len__') and \
    #                 len(intel_basic[key]) == self.n_thread and key != 'ENV-PAUSE':
    #             intel_basic_[key] = intel_basic[key][running]
    #     return intel_basic_

    def deal_with_hook(self, hook, t_intel_basic):
        # use the hook left by algorithm to callback some function 
        # to deliver reward and reset signals
        # assert self.L_RUNNING is not None
        # t_intel_basic = self.filter_running(t_intel_basic, self.L_RUNNING)
        hook({'reward':t_intel_basic['Latest-Reward'], 
                'done': t_intel_basic['Env-Suffered-Reset'],
                'info': t_intel_basic['Latest-Team-Info'],
                'Latest-Obs':t_intel_basic['Latest-Obs'],
                'Terminal-Obs-Echo': t_intel_basic['Terminal-Obs-Echo'],
                })

    def __split_obs(self, obs, t_index):
        # obs $n_thread.$n_team/n_agent.$coredim
        if obs[0] is None:
            o = None
        elif self.ObsAsUnity:
            o = obs[:, t_index]
        else:   # in most cases
            o = obs[:, self.t_member_list[t_index]]
        return o
    def __split_obs_thread(self, obs, t_index):
        # obs $n_thread.$n_team/n_agent.$coredim
        if self.ObsAsUnity:
            o = obs[t_index]
        else:   # in most cases
            o = obs[self.t_member_list[t_index]]
        return o


    # def _NaN(self, dtype):
    #     if np.issubdtype(dtype, np.floating):
    #         return np.nan
    #     else:
    #         return np.iinfo(dtype).min
