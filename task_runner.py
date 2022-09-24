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
from multi_team import MMPlatform
from config import GlobalConfig as cfg
from MISSION.env_router import make_parallel_envs
class Runner(object):
    def __init__(self, process_pool):
        self.process_pool = process_pool
        self.envs = make_parallel_envs(process_pool)          # parallel environments start
        self.mcv = self.get_a_logger(cfg.note)                # MATLAB silent logging bridge active
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
        self.test_epoch=cfg.test_epoch
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
                self.info_runner['Recent-Team-Ranking'].append(term_info['team_ranking'])
            self.info_runner['Latest-Reward-Sum'][i] = 0
            self.info_runner['Current-Obs-Step'][i] = 0
            self.info_runner['Thread-Episode-Cnt'][i] += 1
            # hault finished threads to wait unfinished ones
            if self.align_episode: self.info_runner['ENV-PAUSE'][i] = True
            # monitoring agents/team of interest
            if self.current_n_episode % self.report_interval == 0: 
                self._checkout_interested_agents()  # monitor rewards for some specific agents
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
            assert self.test_epoch%self.n_thread == 0, ('please set test_epoch as (n_thread * N)!')
            num_runs = self.test_epoch // self.n_thread
            print靛('\r[task runner]: test run is going to run %d episode'%self.test_epoch)
            while True:
                actions_list, self.test_info_runner = self.platform_controller.act(self.test_info_runner)
                obs, reward, done, info = self.test_envs.step(actions_list)
                self.test_info_runner = self.update_test_runner(done, obs, reward, info)
                if self.hb_on: print('\r [task runner]: testing %s  '%self.heartbeat(
                    style=3, beat=self.test_info_runner['Current-Obs-Step']), end='', flush=True)
                # If the test run reach its end, record the reward and win-rate:
                if (self.test_info_runner['Thread-Episode-Cnt']>=num_runs).all():
                    # get the reward average
                    reward_of_each_ep = np.stack(self.test_info_runner['Recent-Reward-Sum']) #.squeeze()
                    if self.RewardAsUnity: 
                        reward_avg_itr_agent = reward_of_each_ep[:, self.interested_team].mean()
                    else:
                        reward_avg_itr_agent = reward_of_each_ep[:, self.interested_agents_uid].mean()
                    # get the win rate
                    win_rate = np.array(self.test_info_runner['win']).mean()
                    teams_ranking = self.test_info_runner['Recent-Team-Ranking']
                    if len(teams_ranking)>0:
                        rank_itr_team = np.array(teams_ranking)[:,self.interested_team]
                        win_rate = (rank_itr_team==0).mean()  # 0 means rank first
                        self.mcv.rec(win_rate, 'test top-rank ratio')
                    else:
                        self.mcv.rec(win_rate, 'test-win-rate')
                    self.mcv.rec(reward_avg_itr_agent, 'test-reward')
                    self.mcv.rec_show()
                    print_info = 'average reward: %.2f, win rate: %.2f'%(reward_avg_itr_agent, win_rate)
                    print靛('\r[task runner]: test finished, %s'%print_info )
                    if cfg.upload_after_test: upload_exp(cfg)
                    self.platform_controller.notify_teams(message='test done:%s', 
                        win_rate=win_rate, mean_reward=reward_avg_itr_agent)
                    # close all
                    if self.test_env_sleepy: self.test_envs.sleep()
                    self.platform_controller.before_terminate(self.test_info_runner)
                    return
        def init_test_runner(self):
            if not hasattr(self, 'test_envs'):
                self.test_envs = make_parallel_envs(self.process_pool, marker='test') # 平行环境
            self.test_info_runner = {}  # dict of realtime obs, reward, reward, info et.al.
            self.test_info_runner['ENV-PAUSE'] = np.array([False for _ in range(self.n_thread)])
            self.test_info_runner['Test-Flag'] = True
            self.test_info_runner['win'] = []
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
                self.test_info_runner['win'].append(win)
                if 'team_ranking' in term_info: 
                    self.test_info_runner['Recent-Team-Ranking'].append(term_info['team_ranking'])
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
        return
    def _checkout_interested_agents(self):
        # (1). record mean reward
        recent_rewards = np.stack(self.info_runner['Recent-Reward-Sum'])
        if self.RewardAsUnity:
            mean_reward = recent_rewards[:, self.interested_team].mean()
        else:
            if recent_rewards.shape[-1] != len(self.interested_agents_uid): 
                print('warning! interested_agents_uid:', self.interested_agents_uid)
            mean_reward = recent_rewards[:, self.interested_agents_uid].mean()
        self.mcv.rec(mean_reward, 'reward')
        # (2).reflesh historical top reward
        if self.top_rewards is None: self.top_rewards = mean_reward
        if mean_reward > self.top_rewards: self.top_rewards = mean_reward
        self.mcv.rec(self.top_rewards, 'top reward')
        # (3).record winning rate (single-team) or record winning rate (multi-team)
        win_rate = np.array(self.info_runner['Recent-Win']).mean()
        teams_ranking = self.info_runner['Recent-Team-Ranking']
        if len(teams_ranking)>0:
            rank_itr_team = np.array(teams_ranking)[:,self.interested_team]
            win_rate = (rank_itr_team==0).mean()  # 0 means rank first
            self.mcv.rec(win_rate, 'top-rank ratio')
        else:
            self.mcv.rec(win_rate, 'win rate')
        # plot the figure
        self.mcv.rec_show()
        print靛('\r[task runner]: (%s) finished episode %d, frame %d. | agents of interest: recent reward %.3f, best reward %.3f, win rate %.3f'
                % (self.note, self.current_n_episode, self.current_n_frame, mean_reward, self.top_rewards, win_rate))
        return


    # -- below is nothing of importance --
    # -- you may delete it or replace it with Tensorboard --
    # MATLAB silent logging bridge
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
            self.fps_smooth = int(self.fps_smooth*0.98 + new_fps*0.02)
        return self.fps_smooth
