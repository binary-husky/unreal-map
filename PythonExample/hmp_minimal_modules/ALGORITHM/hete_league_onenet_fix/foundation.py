import os, time, torch, traceback, shutil, pickle, io
import numpy as np
from UTIL.colorful import *
from config import GlobalConfig
from UTIL.tensor_ops import repeat_at, _2tensor
from ALGORITHM.common.rl_alg_base import RLAlgorithmBase
class AlgorithmConfig:
    '''
        AlgorithmConfig: This config class will be 'injected' with new settings from json.
        (E.g., override configs with ```python main.py --cfg example.jsonc```)
        (please see UTIL.config_args to find out how this advanced trick works out.)
    '''
    # configuration, open to jsonc modification
    gamma = 0.99
    tau = 0.95
    train_traj_needed = 512
    hete_n_alive_frontend = 1
    TakeRewardAsUnity = False
    use_normalization = True
    wait_norm_stable = True
    add_prob_loss = False
    n_focus_on = 2
    n_entity_placeholder = 11

    load_checkpoint = False
    load_specific_checkpoint = ''

    # PPO part
    clip_param = 0.2
    ppo_epoch = 16
    n_pieces_batch_division = 1
    value_loss_coef = 0.1
    entropy_coef = 0.05
    max_grad_norm = 0.5
    clip_param = 0.2
    lr = 1e-4

    # prevent GPU OOM
    prevent_batchsize_oom = False
    gamma_in_reward_forwarding = False
    gamma_in_reward_forwarding_value = 0.99

    net_hdim = 24
    
    dual_conc = True

    n_agent = 'auto load, do not change'

    ConfigOnTheFly = True


    hete_n_net_placeholder = 5
    hete_thread_align = False
    hete_same_prob = 0.25
    hete_lasted_n = 100
    
    policy_resonance = False

    use_avail_act = True
    
    debug = False
    ignore_test = False
    
    type_agent_diff_lr = False
    hete_exclude_zero_wr = False
    policy_matrix_testing = False
    test_which_cpk = 1
    type_sel_override = False
    type_sel_override_list = []
    allow_fast_test = True

def str_array_to_num(str_arr):
    out_arr = []
    buffer = {}
    for str in str_arr:
        if str not in buffer:
            buffer[str] = len(buffer)
        out_arr.append(buffer[str])
    return out_arr

def itemgetter(*items):
    # same with operator.itemgetter
    def g(obj): return tuple(obj[item] if item in obj else None for item in items)
    return g
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)
class ReinforceAlgorithmFoundation(RLAlgorithmBase):
    def __init__(self, n_agent, n_thread, space, mcv=None, team=None):
        from .shell_env import ShellEnvWrapper, ActionConvertLegacy
        from .hete_net import HeteNet
        super().__init__(n_agent, n_thread, space, mcv, team)
        AlgorithmConfig.n_agent = n_agent
        self.action_converter = ActionConvertLegacy(
                SELF_TEAM_ASSUME=team, 
                OPP_TEAM_ASSUME=(1-team), 
                OPP_NUM_ASSUME=GlobalConfig.ScenarioConfig.N_AGENT_EACH_TEAM[1-team]
        )
        n_actions = len(self.action_converter.dictionary_args)
        # change obs format, e.g., converting dead agent obs into NaN
        self.shell_env = ShellEnvWrapper(n_agent, n_thread, space, mcv, self, AlgorithmConfig, GlobalConfig.ScenarioConfig, self.team)
        if self.ScenarioConfig.EntityOriented: rawob_dim = self.ScenarioConfig.obs_vec_length
        else: rawob_dim = space['obs_space']['obs_shape']

        # self.StagePlanner, for policy resonance
        from .stage_planner import StagePlanner
        self.stage_planner = StagePlanner(mcv=mcv)

        # heterogeneous agent types
        agent_type_list = [a['type'] for a in GlobalConfig.ScenarioConfig.SubTaskConfig.agent_list]
        self.HeteAgentType = str_array_to_num(agent_type_list)
        hete_type = np.array(self.HeteAgentType)[self.ScenarioConfig.AGENT_ID_EACH_TEAM[team]]
        
        # initialize policy
        self.policy = HeteNet(rawob_dim=rawob_dim, n_action=n_actions, hete_type=hete_type, stage_planner=self.stage_planner)
        self.policy = self.policy.to(self.device)

        # initialize optimizer and trajectory (batch) manager
        from .ppo import PPO
        from .trajectory import BatchTrajManager
        self.trainer = PPO(self.policy, ppo_config=AlgorithmConfig, mcv=mcv)
        self.traj_manager = BatchTrajManager(
            n_env=n_thread, traj_limit=int(GlobalConfig.ScenarioConfig.MaxEpisodeStep),
            trainer_hook=self.trainer.train_on_traj)
        self.stage_planner.trainer = self.trainer

        # confirm that reward method is correct
        self.check_reward_type(AlgorithmConfig)

        # load checkpoints if needed
        self.load_model(AlgorithmConfig)

        # enable config_on_the_fly ability
        if AlgorithmConfig.ConfigOnTheFly:
            self._create_config_fly()

        if AlgorithmConfig.policy_matrix_testing:
            self.threads_test_reward_sum = np.zeros(shape=(n_thread,), dtype=float)
            # self.threads_test_reward = []
            self.recent_test_rewards = []
            self.recent_test_wins = []
            self._unfi_frag_matrix_ = None
            self.recent_test_hete_gp_summary = []
            self.current_hete_gp_summary = None
            from VISUALIZE.mcom import mcom
            self.mcv_matrix = mcom( 
                            path='%s/logger/matrix/'%GlobalConfig.logdir,
                            image_path='%s/matrix.jpg'%GlobalConfig.logdir,
                            draw_mode='Img',
                            tag='[ppo.py]' )
            self.mcv_matrix.rec_init(color='r')


    def action_making(self, StateRecall, test_mode):
        # make sure hook is cleared
        assert ('_hook_' not in StateRecall)
        
        # read obs et.al.
        obs, threads_active_flag, avail_act, hete_pick, hete_type, gp_sel_summary, eprsn = \
            itemgetter('obs', 'threads_active_flag', 'avail_act', '_hete_pick_', '_hete_type_', '_gp_pick_', '_EpRsn_')(StateRecall)
            
        
        # make sure obs shape is correct
        assert obs is not None, ('Make sure obs is ok')
        assert len(obs) == sum(threads_active_flag), ('check batch size')
        
        # make sure avail_act is correct
        if AlgorithmConfig.use_avail_act: assert avail_act is not None
        
        # policy resonance flag reshape
        eprsn = repeat_at(eprsn, -1, self.n_agent)
        thread_index = np.arange(self.n_thread)[threads_active_flag]
        
        # make decision
        with torch.no_grad():
            action, value, action_log_prob = self.policy.act(obs=obs,
                                                             test_mode=test_mode,
                                                             avail_act=avail_act,
                                                             hete_pick=hete_pick,
                                                             hete_type=hete_type,
                                                             gp_sel_summary=gp_sel_summary,
                                                             thread_index=thread_index,
                                                             eprsn=eprsn,
                                                             )

        # commit obs to buffer, vars named like _x_ are aligned, others are not!
        traj_framefrag = {
            "_SKIP_":        ~threads_active_flag,
            "value":         value,
            "hete_pick":     hete_pick,
            "hete_type":     hete_type,
            "gp_sel_summary": gp_sel_summary,
            "avail_act":     avail_act,
            "actionLogProb": action_log_prob,
            "obs":           obs,
            "action":        action,
        }
        if avail_act is not None: traj_framefrag.update({'avail_act':  avail_act})
        
        # deal with rollout later when the reward is ready, leave a hook as a callback here
        if not test_mode: 
            StateRecall['_hook_'] = self.commit_traj_frag(traj_framefrag, req_hook = True)
        else:
            if test_mode and AlgorithmConfig.policy_matrix_testing: 
                StateRecall['_hook_'] = self.matrix_callback_special(traj_framefrag)
        return action.copy(), StateRecall



    ''' 
        function to be called when reward is received
    '''
    def matrix_callback_special(self, framefrag):
        assert self._unfi_frag_matrix_ is None
        self._unfi_frag_matrix_ = framefrag
        return self.matrix_callback_special_callback

    def matrix_callback_special_callback(self, new_frag):
        
        fi_frag = self._unfi_frag_matrix_
        self._unfi_frag_matrix_ = None

        reward = new_frag['reward'].copy()
        done = new_frag['done'].copy()
        # self.threads_test_reward.append(reward)
        self.threads_test_reward_sum += reward * ~fi_frag['_SKIP_']
        
        if not any(fi_frag['_SKIP_']):
            self.current_hete_gp_summary = fi_frag["gp_sel_summary"]

        if done.all():
            self.recent_test_rewards.extend(self.threads_test_reward_sum)
            self.recent_test_wins.extend([q['team_ranking'][self.team]==0 for q in new_frag['info']]) # 0 means rank first
            self.recent_test_hete_gp_summary.extend(self.current_hete_gp_summary)
            self.threads_test_reward_sum *= 0
            self.current_hete_gp_summary = None

        return None


    def interact_with_env(self, StateRecall):
        '''
            Interfacing with marl, standard method that you must implement
            (redirect to shell_env to help with history rolling)
        '''
        return self.shell_env.interact_with_env(StateRecall)


    def interact_with_env_genuine(self, StateRecall):
        '''
            When shell_env finish the preparation, interact_with_env_genuine is called
            (Determine whether or not to do a training routinue)
        '''
        # if not StateRecall['Test-Flag']: self.train()  # when needed, train!
        return self.action_making(StateRecall, StateRecall['Test-Flag'])

    def train(self):
        '''
            Get event from hmp task runner, save model now!
        '''
        if self.traj_manager.can_exec_training():
            if self.stage_planner.can_exec_trainning():
                self.traj_manager.train_and_clear_traj_pool()
            else:
                self.traj_manager.clear_traj_pool()
                
            # read configuration
            if AlgorithmConfig.ConfigOnTheFly: self._config_on_fly()
            
            # 
            self.stage_planner.update_plan()

    # override parent function
    def on_notify(self, message, **kargs):
        win_rate = kargs['win_rate']
        mean_reward = kargs['mean_reward']
        
        path = self.save_model(
            update_cnt=self.traj_manager.update_cnt,
            info=str(kargs)
        )
        # print('[random win rate] ! ! ! ! !')
        # win_rate = np.random.rand()
        self.policy.register_ckp(win_rate, path, mean_reward)
        
        if AlgorithmConfig.policy_matrix_testing:
            from UTIL.data_struct import UniqueList
            # self.recent_test_hete_gp_summary_str = self.recent_test_hete_gp_summary # [str(q.tolist()) for q in self.recent_test_hete_gp_summary]
            recent_test_hete_gp_summary_ls = [q.tolist() for q in self.recent_test_hete_gp_summary]
            ulist = UniqueList(recent_test_hete_gp_summary_ls)
            for u in ulist:
                feature = self.policy.ph_to_feature[u].squeeze().cpu().numpy().tolist()
                feature = "[%.2f,%.2f,%.2f]"%tuple(feature)
                mask = [u==uu  for uu in recent_test_hete_gp_summary_ls]
                r = np.array(self.recent_test_rewards)[mask].mean()
                wr = np.array(self.recent_test_wins)[mask].mean()

                self.mcv_matrix.rec(self.policy.ckpg_input_cnt, 'time')
                self.mcv_matrix.rec(r,  'r of=%s'%feature)
                self.mcv_matrix.rec(wr, 'w of=%s'%feature)
                self.mcv_matrix.rec(sum(mask), 'n of=%s'%feature)

            self.mcv_matrix.rec_show()
            
            self.recent_test_rewards = []
            self.recent_test_wins = []
            self.recent_test_hete_gp_summary = []




    def save_model(self, update_cnt, info=None):
        '''
            save model now!
            save if triggered when:
            1. Update_cnt = 50, 100, ...
            2. Given info, indicating a hmp command
            3. A flag file is detected, indicating a save command from human
        '''
        if not os.path.exists('%s/history_cpt/' % GlobalConfig.logdir): 
            os.makedirs('%s/history_cpt/' % GlobalConfig.logdir)

        # dir 1
        pt_path = '%s/model.pt' % GlobalConfig.logdir
        print绿('saving model to %s' % pt_path)
        torch.save({
            'policy': self.policy.state_dict(),
            'optimizer': self.trainer.optimizer.state_dict(),
        }, pt_path)

        # dir 2
        info = str(update_cnt) if info is None else ''.join([str(update_cnt), '_', info])
        pt_path2 = '%s/history_cpt/model_%s.pt' % (GlobalConfig.logdir, info)
        shutil.copyfile(pt_path, pt_path2)

        # save ckpg_info
        with open('%s/history_cpt/ckpg_info.pkl'%GlobalConfig.logdir, 'wb') as f:pickle.dump((self.policy.ckpg_info, self.policy.ckpg_input_cnt, [(n.feature, n.static, n.ready_to_go) for n in self.policy._nets_flat_placeholder_]),f)
        print绿('save_model fin')
        return pt_path2

    def find_ckp(self, feature):
        import glob
        list_ckp = glob.glob('%s/history_cpt/*.pt'%GlobalConfig.logdir)
        ckp_dir = [ckp for ckp in list_ckp if str(feature[0]) in ckp][0]
        cuda_n = 'cpu' if 'cpu' in self.device else self.device
        cpt = torch.load(ckp_dir, map_location=cuda_n)
        # get previous frontier network
        return {k.replace('_nets_flat_placeholder_.0.',''):v for k, v in cpt['policy'].items() if '_nets_flat_placeholder_.0.' in k}


    def load_model(self, AlgorithmConfig):
        '''
            load model now
        '''

        if AlgorithmConfig.load_checkpoint:
            manual_dir = AlgorithmConfig.load_specific_checkpoint
            ckpt_dir = '%s/model.pt' % GlobalConfig.logdir if manual_dir == '' else '%s/%s' % (GlobalConfig.logdir, manual_dir)
            cuda_n = 'cpu' if 'cpu' in self.device else self.device
            strict = True
            if not platform.system()=="Linux": assert ':' not in ckpt_dir, ('Windows OS does not allow : in file name')
            cpt = torch.load(ckpt_dir, map_location=cuda_n)
            self.policy.load_state_dict(cpt['policy'], strict=strict)
            # https://github.com/pytorch/pytorch/issues/3852
            self.trainer.optimizer.load_state_dict(cpt['optimizer'])

            print黄('loaded checkpoint:', ckpt_dir)
            if os.path.exists('%s/history_cpt/ckpg_info.pkl'%GlobalConfig.logdir):
                with open('%s/history_cpt/ckpg_info.pkl'%GlobalConfig.logdir, 'rb') as f:
                    self.policy.ckpg_info, self.policy.ckpg_input_cnt, n_flags = CPU_Unpickler(f).load()
                for (n, flags) in zip(self.policy._nets_flat_placeholder_, n_flags):
                    n.feature = flags[0]
                    n.static = flags[1]
                    n.ready_to_go = flags[2]
                    if n.feature!=1: 
                        n.load_state_dict(self.find_ckp(n.feature), strict=True)
                self.policy.ph_to_feature = _2tensor(np.array([n.feature for n in self.policy._nets_flat_placeholder_]))
                print黄('loaded ckpg_info')
            else:
                print('Warning, past policy missing !!')


    def process_framedata(self, traj_framedata):
        ''' 
            hook is called when reward and next moment observation is ready,
            now feed them into trajectory manager.
            Rollout Processor | 准备提交Rollout, 以下划线开头和结尾的键值需要对齐(self.n_thread, ...)
            note that keys starting with _ must have shape (self.n_thread, ...), details see fn:mask_paused_env()
        '''
        # strip info, since it is not array
        items_to_pop = ['info', 'Latest-Obs']
        for k in items_to_pop:
            if k in traj_framedata:
                traj_framedata.pop(k)
        # the agent-wise reward is supposed to be the same, so averge them
        if self.ScenarioConfig.RewardAsUnity:
            traj_framedata['reward'] = repeat_at(traj_framedata['reward'], insert_dim=-1, n_times=self.n_agent)
        # change the name of done to be recognised (by trajectory manager)
        traj_framedata['_DONE_'] = traj_framedata.pop('done')
        traj_framedata['_TOBS_'] = traj_framedata.pop(
            'Terminal-Obs-Echo') if 'Terminal-Obs-Echo' in traj_framedata else None
        # mask out pause thread
        traj_framedata = self.mask_paused_env(traj_framedata)
        # put the frag into memory
        self.traj_manager.feed_traj_framedata(traj_framedata)

    def mask_paused_env(self, frag):
        running = ~frag['_SKIP_']
        if running.all():
            return frag
        for key in frag:
            if not key.startswith('_') and hasattr(frag[key], '__len__') and len(frag[key]) == self.n_thread:
                frag[key] = frag[key][running]
        return frag


    def _create_config_fly(self):
        logdir = GlobalConfig.logdir
        self.input_file_dir = '%s/cmd_io.txt' % logdir
        if not os.path.exists(self.input_file_dir):
            with open(self.input_file_dir, 'w+', encoding='utf8') as f: f.writelines(["# Write cmd at next line: ", ""])

    def _config_on_fly(self):
        if not os.path.exists(self.input_file_dir): return

        with open(self.input_file_dir, 'r', encoding='utf8') as f:
            cmdlines = f.readlines()

        cmdlines_writeback = []
        any_change = False

        for cmdline in cmdlines:
            if cmdline.startswith('#') or cmdline=="\n" or cmdline==" \n":
                cmdlines_writeback.append(cmdline)
            else:
                any_change = True
                try:
                    print亮绿('[foundation.py] ------- executing: %s ------'%cmdline)
                    exec(cmdline)
                    cmdlines_writeback.append('# [execute successfully]\t'+cmdline)
                except:
                    print红(traceback.format_exc())
                    cmdlines_writeback.append('# [execute failed]\t'+cmdline)

        if any_change:
            with open(self.input_file_dir, 'w+', encoding='utf8') as f:
                f.writelines(cmdlines_writeback)
