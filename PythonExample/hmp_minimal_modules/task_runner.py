"""
    Author: Fu Qingxu,CASIA
    Description: HMP task runner, coordinates environments and algorithms
    Notes before you read code: 
    In general, HMP task runner can operate two ways:
        self.align_episode = False: threads immediately restart at terminal state, threads do not wait each other
        self.align_episode = True: threads pause at terminal state, waiting until all threads terminate, then reset
"""

import time, os
import numpy as np
from UTIL.colorful import *
from UTIL.exp_helper import upload_exp
from config import GlobalConfig as cfg
from MISSION.env_router import make_parallel_envs
class Runner(object):
    def __init__(self, process_pool):
        self.process_pool = process_pool
        self.envs = make_parallel_envs(process_pool)          # parallel environments start
        self.mcv = self.get_a_logger(cfg.note)                # multiagent silent logging bridge active
        if cfg.mt_parallel: from multi_team_parallel import MMPlatform  # parallel the decision process
        else:               from multi_team          import MMPlatform
        self.platform_controller = MMPlatform(self.mcv, self.envs)  # block infomation access between teams
        self.info_runner = {}                                       # dict of realtime obs, reward, reward, info et.al.
        self.n_agent  =  sum(cfg.ScenarioConfig.N_AGENT_EACH_TEAM)
        self.n_team  =   len(cfg.ScenarioConfig.N_AGENT_EACH_TEAM)
        # please specify: env gives reward of each team instead of agent ?
        self.RewardAsUnity = False  
        if hasattr(cfg.ScenarioConfig, 'RewardAsUnity'):
            self.RewardAsUnity = cfg.ScenarioConfig.RewardAsUnity
        # let test env sleep (when not used) to save memory ?
        self.test_env_sleepy = False
        if hasattr(cfg.ScenarioConfig, 'CanTurnOff'):
            self.test_env_sleepy = cfg.ScenarioConfig.CanTurnOff
        self.n_thread = cfg.num_threads
        self.n_frame =  cfg.n_parallel_frame
        self.note =     cfg.note   # experiment note
        self.hb_on =    cfg.heartbeat_on and stdout.isatty()    # show the environment stepping heartbeat
        self.current_n_frame = 0
        self.current_n_episode = 0
        self.max_n_episode = cfg.max_n_episode
        # Reward monitoring for agents of your interest
        self.train_time_testing = cfg.train_time_testing
        self.test_interval = cfg.test_interval
        self.test_only = cfg.test_only
        self.align_episode = cfg.align_episode
        self._exit_early_ = False
        self._init_interested_agent_logging() 

    # -------------------------------------------------------------------------
    # ------------------------------ Major Loop -------------------------------
    # -------------------------------------------------------------------------
    def run(self):
        # all item in self.info_runner: shape =(n_thread, n_agent/n_team, ...)
        self.init_runner()
        # test machine performance
        tic = time.time()
        # start simulation
        for cnt in range(self.n_frame):
            # line 1: get action, block infomation access between teams (LINK to ARGORITHM)
            # (The controller can also handle algorithm internal state loopback by following simple rules)
            actions_list, self.info_runner = self.platform_controller.act(self.info_runner)
            # line 2: multi-thread environment step (LINK to MISSION)
            # (When thread align is needed, NaN actions will be used to make envs freeze for a step)
            obs, reward, done, info = self.envs.step(actions_list)
            # line 3: prepare obs and reward for next round 
            # (If required, a test run will be started at proper time)
            self.info_runner = self.update_runner(done, obs, reward, info)
            toc=time.time(); dt = toc-tic; tic = toc
            if self.hb_on: print('\r [task runner]: FPS %d, episode steping %s       '%(
                self.get_fps(dt), self.heartbeat()), end='', flush=True)
            if self._exit_early_: print('exit_early'); break
        # All task done! Time to shut down
        return

    def init_runner(self):
        self.info_runner['Test-Flag'] = self.test_only  # not testing mode for rl methods
        self.info_runner['Recent-Reward-Sum'] = []
        self.info_runner['Recent-Win'] = []
        self.info_runner['Recent-Team-Ranking'] = []
        obs_info = self.envs.reset() # assumes only the first time reset is manual
        self.info_runner['Latest-Obs'], self.info_runner['Latest-Team-Info'] = obs_info if isinstance(obs_info, tuple) else (obs_info, None)
        self.info_runner['Env-Suffered-Reset'] = np.array([True for _ in range(self.n_thread)])
        self.info_runner['ENV-PAUSE']          = np.array([False for _ in range(self.n_thread)])
        self.info_runner['Current-Obs-Step']   = np.array([0 for _ in range(self.n_thread)])
        self.info_runner['Latest-Reward']      = np.zeros(shape=(self.n_thread, self.n_agent))
        self.info_runner['Latest-Reward-Sum']  = np.zeros(shape=(self.n_thread, self.n_agent))
        self.info_runner['Thread-Episode-Cnt'] = np.array([0 for _ in range(self.n_thread)])
        if self.RewardAsUnity:
            self.info_runner['Latest-Reward']  = np.zeros(shape=(self.n_thread, self.n_team))
            self.info_runner['Latest-Reward-Sum'] = np.zeros(shape=(self.n_thread, self.n_team))
        return

    def update_runner(self, done, obs, reward, info):
        P = self.info_runner['ENV-PAUSE']
        R = ~P
        assert info is not None
        if self.info_runner['Latest-Team-Info'] is None: self.info_runner['Latest-Team-Info'] = info
        self.info_runner['Latest-Obs'][R] = obs[R]
        self.info_runner['Latest-Team-Info'][R] = info[R]
        self.info_runner['Latest-Reward'][R] = reward[R]    # note, reward shape: (thread, n-team\n-agent)
        self.info_runner['Latest-Reward-Sum'][R] += reward[R]
        self.info_runner['Current-Obs-Step'][R] += 1
        for i in range(self.n_thread):
            self.info_runner['Env-Suffered-Reset'][i] = done[i].all()
            # if the environment has not been reset, do nothing
            if P[i] or (not self.info_runner['Env-Suffered-Reset'][i]): continue
            # otherwise, the environment just been reset
            self.current_n_frame += self.info_runner['Current-Obs-Step'][i]
            self.current_n_episode += 1
            self.info_runner['Recent-Reward-Sum'].append(self.info_runner['Latest-Reward-Sum'][i].copy())
            term_info = self.info_runner['Latest-Team-Info'][i]
            # update win/lose (single-team), or team ranking (multi-team)
            win = 1 if 'win' in term_info and term_info['win']==True else 0
            self.info_runner['Recent-Win'].append(win)
            if 'team_ranking' in term_info: 
                self.info_runner['Recent-Team-Ranking'].append(term_info['team_ranking'].copy())
            self.info_runner['Latest-Reward-Sum'][i] = 0
            self.info_runner['Current-Obs-Step'][i] = 0
            self.info_runner['Thread-Episode-Cnt'][i] += 1
            # hault finished threads to wait unfinished ones
            if self.align_episode: self.info_runner['ENV-PAUSE'][i] = True
            # monitoring agents/team of interest
            if self.current_n_episode % self.report_interval == 0: 
                self._checkout_interested_agents(self.info_runner)  # monitor rewards for some specific agents
                self.info_runner['Recent-Reward-Sum'] = []
                self.info_runner['Recent-Win'] = []
                self.info_runner['Recent-Team-Ranking'] = []
            # begin a testing session?
            if self.train_time_testing and (not self.test_only) and (self.current_n_episode % self.test_interval == 0): 
                self.platform_controller.before_terminate(self.info_runner)
                self.start_a_test_run()
        # all threads haulted, finished and Aligned, then restart all thread
        if self.align_episode and self.info_runner['ENV-PAUSE'].all():  self.info_runner['ENV-PAUSE'][:] = False
        # when too many episode is done, Terminate flag on.
        if self.current_n_episode >= self.max_n_episode: self._exit_early_ = True
        return self.info_runner




    # ------------------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ About TEST RUN routine, almost a Mirror of above ----------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------------------------------
    # -- I know these code below might merge with above for simplicity --
    # -- But I decide not, in order to make it easier to read and debug --
    if cfg.train_time_testing:
        def start_a_test_run(self):
            print靛('\r[task runner]: test run started!')
            self.init_test_runner()
            # loop until all env is done
            assert cfg.test_epoch%self.n_thread == 0, ('please set test_epoch as (n_thread * N)!')
            num_runs = cfg.test_epoch // self.n_thread
            print靛('\r[task runner]: test run is going to run %d episode'%cfg.test_epoch)
            while True:
                actions_list, self.test_info_runner = self.platform_controller.act(self.test_info_runner)
                obs, reward, done, info = self.test_envs.step(actions_list)
                self.test_info_runner = self.update_test_runner(done, obs, reward, info)
                if self.hb_on: print('\r [task runner]: testing %s  '%self.heartbeat(
                    style=3, beat=self.test_info_runner['Current-Obs-Step']), end='', flush=True)
                # If the test run reach its end, record the reward and win-rate:
                if (self.test_info_runner['Thread-Episode-Cnt']>=num_runs).all():
                    # get the reward average
                    t_win_rates, t_rewards = self._checkout_interested_agents(self.test_info_runner, testing=True)
                    self.platform_controller.before_terminate(self.test_info_runner)
                    self.platform_controller.notify_teams('test done', win_rate=t_win_rates, mean_reward=t_rewards)
                    # close all
                    if self.test_env_sleepy: self.test_envs.sleep()
                    return
        def init_test_runner(self):
            if not hasattr(self, 'test_envs'):
                self.test_envs = make_parallel_envs(self.process_pool, marker='test') # 平行环境
            self.test_info_runner = {}  # dict of realtime obs, reward, reward, info et.al.
            self.test_info_runner['ENV-PAUSE'] = np.array([False for _ in range(self.n_thread)])
            self.test_info_runner['Test-Flag'] = True
            self.test_info_runner['Recent-Win'] = []
            self.test_info_runner['Recent-Reward-Sum'] = []
            self.test_info_runner['Recent-Team-Ranking'] = []
            test_obs_info = self.test_envs.reset() # assume only the first time reset is manual
            self.test_info_runner['Latest-Obs'], self.test_info_runner['Latest-Team-Info'] = test_obs_info if isinstance(test_obs_info, tuple) else (test_obs_info, None)
            self.test_info_runner['Env-Suffered-Reset'] = np.array([True for _ in range(self.n_thread)])
            self.test_info_runner['Latest-Reward'] = np.zeros(shape=(self.n_thread, self.n_agent))
            self.test_info_runner['Latest-Reward-Sum'] = np.zeros(shape=(self.n_thread, self.n_agent))
            self.test_info_runner['Current-Obs-Step'] = np.array([0 for _ in range(self.n_thread)])
            self.test_info_runner['Thread-Episode-Cnt'] = np.array([0 for _ in range(self.n_thread)])
            if self.RewardAsUnity:
                self.test_info_runner['Latest-Reward'] = np.zeros(shape=(self.n_thread, self.n_team))
                self.test_info_runner['Latest-Reward-Sum'] = np.zeros(shape=(self.n_thread, self.n_team))
            return
        def update_test_runner(self, done, obs, reward, info):
            P = self.test_info_runner['ENV-PAUSE']
            R = ~P
            assert info is not None
            if self.test_info_runner['Latest-Team-Info'] is None: self.test_info_runner['Latest-Team-Info'] = info
            self.test_info_runner['Latest-Obs'][R] = obs[R]
            self.test_info_runner['Latest-Team-Info'][R] = info[R]
            self.test_info_runner['Latest-Reward'][R] = reward[R]
            self.test_info_runner['Latest-Reward-Sum'][R] += reward[R]
            self.test_info_runner['Current-Obs-Step'][R] += 1
            for i in range(self.n_thread):
                self.test_info_runner['Env-Suffered-Reset'][i] = done[i].all()
                # if the environment has not been reset, do nothing
                if P[i] or (not self.test_info_runner['Env-Suffered-Reset'][i]): continue
                # otherwise, the environment just been reset
                self.test_info_runner['Recent-Reward-Sum'].append(self.test_info_runner['Latest-Reward-Sum'][i].copy())
                self.test_info_runner['Latest-Reward-Sum'][i] = 0
                self.test_info_runner['Current-Obs-Step'][i] = 0
                self.test_info_runner['Thread-Episode-Cnt'][i] += 1
                term_info = self.test_info_runner['Latest-Team-Info'][i]
                win = 1 if 'win' in term_info and term_info['win']==True else 0
                self.test_info_runner['Recent-Win'].append(win)
                if 'team_ranking' in term_info: 
                    self.test_info_runner['Recent-Team-Ranking'].append(term_info['team_ranking'].copy())
                if self.align_episode: self.test_info_runner['ENV-PAUSE'][i] = True
            if self.align_episode and self.test_info_runner['ENV-PAUSE'].all(): self.test_info_runner['ENV-PAUSE'][:] = False
            return self.test_info_runner




    # -- If you care much about the agents running your algorthm... --
    # -- you may delete them if monitering is established in ALGORITHM level --
    def _init_interested_agent_logging(self):
        self.report_interval = cfg.report_reward_interval
        self.interested_agents_uid = cfg.interested_agent_uid
        self.interested_team = cfg.interested_team
        self.top_rewards = None
        self.test_top_rewards = None
        return
    def _checkout_interested_agents(self, info_runner, testing=False):
        # (1). record mean reward
        if not testing: self.mcv.rec(self.current_n_episode, 'time')
        prefix = 'test' if testing else ''
        recent_rewards = np.stack(info_runner['Recent-Reward-Sum'])
        mean_reward_each_team = []
        if self.RewardAsUnity:
            for interested_team in range(self.n_team):
                mean_reward_each_team.append(recent_rewards[:, interested_team].mean().copy())
        else:
            for interested_team in range(self.n_team):
                tean_agent_uid = cfg.ScenarioConfig.AGENT_ID_EACH_TEAM[interested_team]
                mean_reward_each_team.append(recent_rewards[:, tean_agent_uid].mean().copy())

        for team in range(self.n_team):
            self.mcv.rec(mean_reward_each_team[team], f'{prefix} reward of=team-{team}')

        # (2).reflesh historical top reward
        if not testing: 
            if self.top_rewards is None: self.top_rewards = mean_reward_each_team
            top_rewards_list_pointer = self.top_rewards
        else:
            if self.test_top_rewards is None: self.test_top_rewards = mean_reward_each_team
            top_rewards_list_pointer = self.test_top_rewards
        for team in range(self.n_team):
            if mean_reward_each_team[team] > top_rewards_list_pointer[team]:
                top_rewards_list_pointer[team] = mean_reward_each_team[team]
            self.mcv.rec(top_rewards_list_pointer[team], f'{prefix} top reward of=team-{team}')

        # (3).record winning rate (single-team) or record winning rate (multi-team)
        # for team in range(self.n_team):
        teams_ranking = info_runner['Recent-Team-Ranking']
        win_rate_each_team = [0]*self.n_team
        if len(teams_ranking)>0:
            for team in range(self.n_team):
                rank_itr_team = np.array(teams_ranking)[:, team]
                win_rate = (rank_itr_team==0).mean()  # 0 means rank first
                win_rate_each_team[team] = win_rate
                self.mcv.rec(win_rate, f'{prefix} top-rank ratio of=team-{team}')
        else:
            team = 0; assert self.n_team == 1, "There is only one team"
            win_rate_each_team[team] = np.array(info_runner['Recent-Win']).mean()
            win_rate = np.array(info_runner['Recent-Win']).mean()
            self.mcv.rec(win_rate, f'{prefix} win rate of=team-{team}')

        # plot the figure
        self.mcv.rec_show()
        if testing: 
            print_info = ['\r[task runner]: Test result at episode %d.'%(self.current_n_episode)]
        else:
            print_info = ['\r[task runner]: (%s) Finished episode %d, frame %d.'%(self.note, self.current_n_episode, self.current_n_frame)]

        for team in range(self.n_team): 
            print_info.append(' | team-%d: win rate: %.3f, recent reward %.3f'%(team, win_rate_each_team[team], mean_reward_each_team[team]))
        print靛(''.join(print_info))
            
        return win_rate_each_team, mean_reward_each_team


    # -- below is nothing of importance --
    # -- you may delete it or replace it with Tensorboard --
    @staticmethod
    def get_a_logger(note):
        from VISUALIZE.mcom import mcom
        logdir = cfg.logdir
        if cfg.activate_logger:
            mcv = mcom( path='%s/logger/'%logdir,
                        digit=16,
                        rapid_flush=True,
                        draw_mode=cfg.draw_mode,
                        tag='[task_runner.py]',
                        resume_mod=cfg.resume_mod)
            cfg.data_logger = mcv
        mcv.rec_init(color='b')
        return mcv

    def heartbeat(self, style=0, beat=None):
        # default ⠁⠈⠐⠠⢀⡀⠄⠂
        width = os.get_terminal_size().columns
        if style==0: sym = ['⠁','⠈','⠐','⠠','⢀','⡀','⠄','⠂',]
        elif style==1: sym = ['◐ ','◓ ','◑ ','◒ ']
        elif style==2: sym = ['▁','▂','▃','▄','▅','▆','▇','█']
        elif style==3: sym = ['💐','🌷','🌸','🌹','🌺','🌻','🌼',]
        if beat is None: beat = self.info_runner['Current-Obs-Step']
        beat = beat % len(sym)
        beat = beat[:int(width*0.2)]
        beat.astype(np.int)
        beat = [sym[t] for t in beat]
        return ''.join(beat)

    
    def get_fps(self, dt):
        new_fps = int(self.n_thread/dt)
        if not hasattr(self, 'fps_smooth'):
            self.fps_smooth = new_fps
        else:
            self.fps_smooth = self.fps_smooth*0.98 + new_fps*0.02
        return int(self.fps_smooth)