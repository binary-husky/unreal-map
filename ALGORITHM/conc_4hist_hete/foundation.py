import os, time, torch, traceback, shutil
import numpy as np
from UTIL.colorful import *
from config import GlobalConfig
from UTIL.tensor_ops import repeat_at
from ..commom.rl_alg_base import RLAlgorithmBase
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
    TakeRewardAsUnity = False
    use_normalization = True
    add_prob_loss = False
    n_focus_on = 2
    n_entity_placeholder = 24

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

    # sometimes the episode length gets longer,
    # resulting in more samples and causing GPU OOM,
    # prevent this by fixing the number of samples to initial
    # by randomly sampling and droping
    prevent_batchsize_oom = False
    gamma_in_reward_forwarding = False
    gamma_in_reward_forwarding_value = 0.99

    # net
    net_hdim = 24
    dual_conc = True

    n_agent = 'auto load, do not change'
    # yita = 0.
    # div_tree_init_level = 0 # set to -1 means max level
    # yita_min_prob = 0.15  #  should be >= (1/n_action)
    ConfigOnTheFly = True
    # UseDivTree = False
    hete_type_trainable = [False, False, True]

    # personality reinforcement dynamic
    # 0 means activiting PR at beginning, -1 means never activate PR, >0 means activiting PR after some updates
    # personality_reinforcement_start_at_update = -1
    # div_tree_level_inc_per_update = 0.0 # (30 updates per inc)
    # yita_max = 0.75
    # yita_inc_per_update = 0.75/100 # (increase to 0.75 in 500 updates)

    # PR_ACTIVATE = False # please always init to False
    
    #####
    n_policy_groups = 5
    #####

    entity_distinct = 'auto load, do not change'


class ReinforceAlgorithmFoundation(RLAlgorithmBase):
    def __init__(self, n_agent, n_thread, space, mcv=None, team=None):
        from .shell_env import ShellEnvWrapper, ActionConvertLegacy
        from .net import HeteNet
        super().__init__(n_agent, n_thread, space, mcv, team)
        AlgorithmConfig.n_agent = n_agent
        n_actions = len(ActionConvertLegacy.dictionary_args)

        self.shell_env = ShellEnvWrapper(n_agent, n_thread, space, mcv, self, AlgorithmConfig, GlobalConfig.ScenarioConfig, self.team)
        if self.ScenarioConfig.EntityOriented :
            rawob_dim = self.ScenarioConfig.obs_vec_length
        else:
            rawob_dim = space['obs_space']['obs_shape']
            
            
        # hete agent policy
        assert self.ScenarioConfig.HeteAgents
        self.HeteAgentType = self.ScenarioConfig.HeteAgentType
        hete_type = np.array(self.HeteAgentType)[self.ScenarioConfig.AGENT_ID_EACH_TEAM[team]]
        self.policy = HeteNet(rawob_dim=rawob_dim, n_action=n_actions, hete_type=hete_type)
        self.policy = self.policy.to(self.device)

        self.AvgRewardAgentWise = AlgorithmConfig.TakeRewardAsUnity
        # initialize policy network and traj memory manager
        from .ppo import PPO
        from .trajectory import BatchTrajManager
        self.trainer = PPO(self.policy, ppo_config=AlgorithmConfig, mcv=mcv)
        self.traj_manager = BatchTrajManager(
            n_env=n_thread, traj_limit=int(GlobalConfig.ScenarioConfig.MaxEpisodeStep),
            trainer_hook=self.trainer.train_on_traj)

        # confirm that reward method is correct
        self.check_reward_type(AlgorithmConfig)

        # load checkpoints if needed
        self.load_model(AlgorithmConfig)

        # activate config on the fly ability
        if AlgorithmConfig.ConfigOnTheFly:
            self._create_config_fly()

        # assert AlgorithmConfig.personality_reinforcement_start_at_update>=0, "?"


    def action_making(self, StateRecall, test_mode):
        assert StateRecall['obs'] is not None, ('Make sure obs is ok')
        assert ('_hook_' not in StateRecall)

        obs, threads_active_flag = StateRecall['obs'], StateRecall['threads_active_flag']
        assert len(obs) == sum(threads_active_flag), ('Make sure the right batch of obs!')
        avail_act = StateRecall['avail_act'] if 'avail_act' in StateRecall else None
        hete_pick = StateRecall['_Type_']
        with torch.no_grad():
            # if AlgorithmConfig.PR_ACTIVATE:  self.policy.ccategorical.register_fixmax(StateRecall['_FixMax_'])
            action, value, action_log_prob = self.policy.act(obs=obs, test_mode=test_mode, avail_act=avail_act, hete_pick=hete_pick)

        # Warning! vars named like _x_ are aligned, others are not!
        traj_framefrag = {
            "_SKIP_":        ~threads_active_flag,
            "value":         value,
            "hete_pick":     hete_pick,
            "actionLogProb": action_log_prob,
            "obs":           obs,
            "action":        action,
        }
        if avail_act is not None: traj_framefrag.update({'avail_act':  avail_act})
        
        # deal with rollout later when the reward is ready, leave a hook as a callback here
        if not test_mode: StateRecall['_hook_'] = self.commit_traj_frag(traj_framefrag, req_hook = True)
        return action.copy(), StateRecall


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
        if not StateRecall['Test-Flag']: self.train()  # when needed, train!
        return self.action_making(StateRecall, StateRecall['Test-Flag'])


    # def activate_pr(self):
    #     AlgorithmConfig.PR_ACTIVATE = True
    #     AlgorithmConfig.only_train_div_tree_and_ct = True
    #     self.trainer.fn_only_train_div_tree_and_ct()

    # def when_pr_inactive(self):
    #     assert not AlgorithmConfig.PR_ACTIVATE
    #     if AlgorithmConfig.personality_reinforcement_start_at_update >= 0:
    #         # mean need to activate pr later
    #         if self.traj_manager.update_cnt > AlgorithmConfig.personality_reinforcement_start_at_update:
    #             # time is up, activate pr
    #             self.activate_pr()

    #     # log
    #     PR_ACTIVATE = 1 if AlgorithmConfig.PR_ACTIVATE else 0
    #     self.mcv.rec(PR_ACTIVATE, 'PR_ACTIVATE')
    #     self.mcv.rec(self.policy.AT_div_tree.current_level, 'personality level')
    #     self.mcv.rec(AlgorithmConfig.yita, 'yita')

    # def when_pr_active(self):
    #     assert AlgorithmConfig.PR_ACTIVATE
    #     self._update_yita()
    #     self._update_personality_division()

    #     # log
    #     PR_ACTIVATE = 1 if AlgorithmConfig.PR_ACTIVATE else 0
    #     self.mcv.rec(PR_ACTIVATE, 'PR_ACTIVATE')
    #     self.mcv.rec(self.policy.AT_div_tree.current_level, 'personality level')
    #     self.mcv.rec(AlgorithmConfig.yita, 'yita')

    def train(self):
        '''
            Get event from hmp task runner, save model now!
        '''
        if self.traj_manager.can_exec_training():
            # time to start a training routine
            self.traj_manager.train_and_clear_traj_pool()
            # read configuration
            if AlgorithmConfig.ConfigOnTheFly:
                self._config_on_fly()
            # if AlgorithmConfig.PR_ACTIVATE:
            #     self.when_pr_active()
            # elif not AlgorithmConfig.PR_ACTIVATE:
            #     self.when_pr_inactive()


    # def _update_personality_division(self):
    #     '''
    #         increase personality tree level @div_tree_level_inc_per_update per fn call, 
    #         when floating break int threshold, the tree enters next level
    #     '''
    #     personality_tree = self.policy.AT_div_tree
    #     personality_tree.current_level_floating += AlgorithmConfig.div_tree_level_inc_per_update
    #     if personality_tree.current_level_floating > personality_tree.max_level:
    #         personality_tree.current_level_floating = personality_tree.max_level

    #     expected_level = int(personality_tree.current_level_floating)
    #     if expected_level == personality_tree.current_level: return
    #     personality_tree.change_div_tree_level(expected_level, auto_transfer=True)
    #     print('[div_tree]: change_div_tree_level, ', personality_tree.current_level)


    # def _update_yita(self):
    #     '''
    #         increase yita by @yita_inc_per_update per function call
    #     '''
    #     AlgorithmConfig.yita += AlgorithmConfig.yita_inc_per_update
    #     if AlgorithmConfig.yita > AlgorithmConfig.yita_max:
    #         AlgorithmConfig.yita = AlgorithmConfig.yita_max
    #     print亮绿('AlgorithmConfig.yita update:', AlgorithmConfig.yita)







    def save_model(self, update_cnt, info=None):
        '''
            save model now!
            save if triggered when:
            1. Update_cnt = 50, 100, ...
            2. Given info, indicating a hmp command
            3. A flag file is detected, indicating a save command from human
        '''
        if not os.path.exists('%s/history_cpt/' % GlobalConfig.logdir): os.makedirs('%s/history_cpt/' % GlobalConfig.logdir)

        # dir 1
        pt_path = '%s/model.pt' % GlobalConfig.logdir
        print绿('saving model to %s' % pt_path)
        torch.save({
            'policy': self.policy.state_dict(),
            'at_optimizer': self.trainer.at_optimizer.state_dict(),
            'ct_optimizer': self.trainer.ct_optimizer.state_dict(),
        }, pt_path)

        # dir 2
        info = str(update_cnt) if info is None else ''.join([str(update_cnt), '_', info])
        pt_path2 = '%s/history_cpt/model_%s.pt' % (GlobalConfig.logdir, info)
        shutil.copyfile(pt_path, pt_path2)

        print绿('save_model fin')



    def load_model(self, AlgorithmConfig):
        '''
            load model now
        '''

        if AlgorithmConfig.load_checkpoint:
            manual_dir = AlgorithmConfig.load_specific_checkpoint
            ckpt_dir = '%s/model.pt' % GlobalConfig.logdir if manual_dir == '' else '%s/%s' % (GlobalConfig.logdir, manual_dir)
            cuda_n = 'cpu' if 'cpu' in self.device else self.device
            strict = True
            
            cpt = torch.load(ckpt_dir, map_location=cuda_n)
            self.policy.load_state_dict(cpt['policy'], strict=strict)
            # https://github.com/pytorch/pytorch/issues/3852
            self.trainer.at_optimizer.load_state_dict(cpt['at_optimizer'])
            self.trainer.ct_optimizer.load_state_dict(cpt['ct_optimizer'])

            print黄('loaded checkpoint:', ckpt_dir)



    def process_framedata(self, traj_framedata):
        ''' 
            hook is called when reward and next moment observation is ready,
            now feed them into trajectory manager.
            Rollout Processor 准备提交Rollout, 以下划线开头和结尾的键值需要对齐(self.n_thread, ...)
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
