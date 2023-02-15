import numpy as np
import importlib
from UTIL.data_struct import UniqueList

class MMPlatform(object):

    def __init__(self, mcv, envs):
        from config import GlobalConfig
        self.n_t =           GlobalConfig.ScenarioConfig.N_TEAM                # n_t => n_teams
        n_agents_each_t =    GlobalConfig.ScenarioConfig.N_AGENT_EACH_TEAM     # n_agents_each_t => n_agents_each_team
        self.t_member_list = GlobalConfig.ScenarioConfig.AGENT_ID_EACH_TEAM
        self.t_name =        GlobalConfig.ScenarioConfig.TEAM_NAMES
        assert self.n_t == len(self.t_name), 'Team does not match agent id'    # check N_TEAM
        assert self.n_t == len(UniqueList(self.t_name)), 'Team name must not repeat' # please duplicate algorithm if needed
        self.align_episode = GlobalConfig.align_episode
        self.n_thread =      GlobalConfig.num_threads
        self.legacy_act_order = True
        if GlobalConfig.mt_act_order == 'new_method':
            self.legacy_act_order = False
        self.RewardAsUnity = False # env give reward of each team instead of agent
        if hasattr(GlobalConfig.ScenarioConfig, 'RewardAsUnity'):
            self.RewardAsUnity = GlobalConfig.ScenarioConfig.RewardAsUnity 
        self.ActAsUnity = False
        if hasattr(GlobalConfig.ScenarioConfig, 'ActAsUnity'):
            self.ActAsUnity = GlobalConfig.ScenarioConfig.ActAsUnity
        self.ObsAsUnity = False
        if hasattr(GlobalConfig.ScenarioConfig, 'ObsAsUnity'):
            self.ObsAsUnity = GlobalConfig.ScenarioConfig.ObsAsUnity

        space = envs.get_space()    # get observation space and action space

        self.algo_foundations = []  # import and initialize algorithms
        for t in range(self.n_t):
            assert len(self.t_member_list[t]) == n_agents_each_t[t]
            assert '->' in self.t_name[t]
            module_, class_ = self.t_name[t].split('->')
            init_f = getattr(importlib.import_module(module_), class_)
            self.algo_foundations.append(
                init_f(n_agent=n_agents_each_t[t], n_thread=self.n_thread, space=space, mcv=mcv, team=t)
            )
        pass


    def act(self, runner_info):
        actions_list = []
        for t_name, t_members, algo_fdn, t_index in zip(self.t_name, self.t_member_list, self.algo_foundations, range(self.n_t)):
            # split intel such as reward and observation into different teams
            _t_intel_ = self._split_intel(runner_info, t_members, t_name, t_index)
            # each team (controlled by different algorithms) interacts with env and act
            _act_, _t_intel_ = algo_fdn.interact_with_env(_t_intel_)
            # concat actions of each agent ('_act_' --> 'actions_list')
            actions_list = self._append_act_to_list(_act_, actions_list, t_members)
            # loop back internal states registered in _t_intel_  (e.g._division_obs_)
            if _t_intel_ is None: continue
            # process internal states loop back, featured with keys that startswith and endswith '_'
            for key in _t_intel_:
                if key.startswith('_') and key.endswith('_'): 
                    self._update_runner(runner_info, runner_info['ENV-PAUSE'], t_name, key, _t_intel_[key])
        pass

        # swapaxes:  [n_agent(n_teams if ActAsUnity), n_thread] --> [n_thread, $n_agent(n_teams if ActAsUnity)]
        actions_list = np.swapaxes(np.array(actions_list, dtype=np.double), 0, 1) 

        # in align_episode mod, threads that are paused are forced to give NaN action
        ENV_PAUSE = runner_info['ENV-PAUSE']
        if ENV_PAUSE.any() and self.align_episode: actions_list[ENV_PAUSE,:] = np.nan
        return actions_list, runner_info

    def before_terminate(self, runner_info):
        for t_name, t_members, t_index in zip(self.t_name, self.t_member_list, range(self.n_t)):
            # split info such as reward and observation
            self._split_intel(runner_info, t_members, t_name, t_index)


    def _update_runner(self, runner_info, ENV_PAUSE, t_name, key, content):
        u_key = t_name+key
        if (u_key in runner_info) and hasattr(content, '__len__') and \
                len(content)==self.n_thread and ENV_PAUSE.any():
            runner_info[u_key][~ENV_PAUSE] = content[~ENV_PAUSE]
            return
        runner_info[u_key] = content
        return

    # seperate observation between teams
    def _split_intel(self, runner_info, t_members, t_name, t_index):
        # RUNNING = ~runner_info['ENV-PAUSE']
        # Team_Info and ter_obs_echo are None when runner_info['Latest-Team-Info'] is absent
        Team_Info = None
        ter_obs_echo = None
        # load Team_Info and ter_obs_echo
        if runner_info['Latest-Team-Info'] is not None:
            assert isinstance(runner_info['Latest-Team-Info'][0], dict)
            Team_Info = runner_info['Latest-Team-Info']
            # if a env just ended ('Env-Suffered-Reset'), the final step obs can be acquired here
            ter_obs_echo = np.array([self.__split_obs_thread(Team_Info[thread_idx]['obs-echo'], t_index)
                            if done and ('obs-echo' in Team_Info[thread_idx]) else None 
                            for thread_idx, done in enumerate(runner_info['Env-Suffered-Reset'])], dtype=object)

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
            if (s_key != '_hook_'): continue
            # otherwise deal with _hook_
            if t_intel_basic['_hook_'] is not None:
                self.deal_with_hook(t_intel_basic['_hook_'], t_intel_basic)
                runner_info[key] = None
                t_intel_basic['_hook_'] = None
            # remove _hook_ key
            t_intel_basic.pop('_hook_')
            
        # t_intel_basic = self.filter_running(t_intel_basic, RUNNING)
        return t_intel_basic

    def _append_act_to_list(self, _act_, actions_list, t_members):
        if not self.legacy_act_order: _act_ = np.swapaxes(_act_, 0, 1) 
        assert _act_.shape[0]==len(t_members), ('number of actions differs number of agents!')
        append_op = actions_list.append if self.ActAsUnity else actions_list.extend
        append_op(_act_)
        return actions_list

    def deal_with_hook(self, hook, t_intel_basic):
        # use the hook left by algorithm to callback some function 
        # to deliver reward and reset signals
        # assert self.L_RUNNING is not None
        # t_intel_basic = self.filter_running(t_intel_basic, self.L_RUNNING)
        hook({  'reward':t_intel_basic['Latest-Reward'], 
                'done': t_intel_basic['Env-Suffered-Reset'],
                'info': t_intel_basic['Latest-Team-Info'],
                'Latest-Obs':t_intel_basic['Latest-Obs'],
                'Terminal-Obs-Echo': t_intel_basic['Terminal-Obs-Echo'],
                })

    def notify_teams(self, message, **kargs):
        for t_index, algo_fdn in enumerate(self.algo_foundations):
            if (not hasattr(algo_fdn, 'on_notify')) or (not callable(algo_fdn.on_notify)): continue
            team_kargs = {k:v[t_index] for k,v in kargs.items()}
            algo_fdn.on_notify(message, **team_kargs)
            
    def __split_obs(self, obs, t_index):
        # obs [n_thread, n_team/n_agent, coredim]
        if obs[0] is None:
            o = None
        elif self.ObsAsUnity:
            o = obs[:, t_index]
        else:   # in most cases
            o = obs[:, self.t_member_list[t_index]]
        return o

    def __split_obs_thread(self, obs, t_index):
        # obs [n_thread, n_team/n_agent, coredim]
        if self.ObsAsUnity:
            o = obs[t_index]
        else:   # in most cases
            o = obs[self.t_member_list[t_index]]
        return o