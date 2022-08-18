from .foundation import AlgorithmConfig
from UTIL.colorful import *

class PolicyRsnConfig:
    yita = 0.
    div_tree_init_level = 0 # set to -1 means max level
    yita_min_prob = 0.15  #  should be >= (1/n_action)
    
    personality_reinforcement_start_at_update = -1
    div_tree_level_inc_per_update = 0.0 # (30 updates per inc)
    yita_max = 0.75
    
    yita_inc_per_update = 0.75/100 # (increase to 0.75 in 500 updates)

class StagePlanner:
    def __init__(self) -> None:
        self.resonance_active = False
        
        if AlgorithmConfig.wait_norm_stable:
            self.wait_norm_stable_cnt = 5
        else:
            self.wait_norm_stable_cnt = 0

    def can_exec_trainning(self):
        if self.wait_norm_stable_cnt > 0:
            print亮绿('waiting initial normalization stable, skip training!')
            self.traj_manager.clear_traj_pool()
            self.wait_norm_stable_cnt -= 1
            return False
        else:
            return True

    def is_resonance_active(self,):
        return self.resonance_active
    
    def update_plan(self):
        
        if self.resonance_active:
            self.when_pr_active()
        elif not self.resonance_active:
            self.when_pr_inactive()
        return
    
    def activate_pr(self):
        self.resonance_active = True
        AlgorithmConfig.only_train_div_tree_and_ct = True
        self.trainer.fn_only_train_div_tree_and_ct()



    def when_pr_inactive(self):
        assert not self.resonance_active
        if AlgorithmConfig.personality_reinforcement_start_at_update >= 0:
            # mean need to activate pr later
            if self.traj_manager.update_cnt > AlgorithmConfig.personality_reinforcement_start_at_update:
                # time is up, activate pr
                self.activate_pr()

        # log
        pr = 1 if self.resonance_active else 0
        self.mcv.rec(pr, 'PR_ACTIVATE')
        self.mcv.rec(self.policy.AT_div_tree.current_level, 'personality level')
        self.mcv.rec(AlgorithmConfig.yita, 'yita')

    def when_pr_active(self):
        assert self.resonance_active
        self._update_yita()
        self._update_personality_division()

        # log
        pr = 1 if self.resonance_active else 0
        self.mcv.rec(pr, 'PR_ACTIVATE')
        self.mcv.rec(self.policy.AT_div_tree.current_level, 'personality level')
        self.mcv.rec(AlgorithmConfig.yita, 'yita')


    def _update_yita(self):
        '''
            increase yita by @yita_inc_per_update per function call
        '''
        AlgorithmConfig.yita += AlgorithmConfig.yita_inc_per_update
        if AlgorithmConfig.yita > AlgorithmConfig.yita_max:
            AlgorithmConfig.yita = AlgorithmConfig.yita_max
        print亮绿('AlgorithmConfig.yita update:', AlgorithmConfig.yita)


    def _update_personality_division(self):
        '''
            increase personality tree level @div_tree_level_inc_per_update per fn call, 
            when floating break int threshold, the tree enters next level
        '''
        personality_tree = self.policy.AT_div_tree
        personality_tree.current_level_floating += AlgorithmConfig.div_tree_level_inc_per_update
        if personality_tree.current_level_floating > personality_tree.max_level:
            personality_tree.current_level_floating = personality_tree.max_level

        expected_level = int(personality_tree.current_level_floating)
        if expected_level == personality_tree.current_level: return
        personality_tree.change_div_tree_level(expected_level, auto_transfer=True)
        print('[div_tree]: change_div_tree_level, ', personality_tree.current_level)

