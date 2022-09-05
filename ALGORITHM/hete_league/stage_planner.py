from .foundation import AlgorithmConfig
from UTIL.colorful import *

class PolicyRsnConfig:
    resonance_start_at_update = 10
    yita_min_prob = 0.15  #  should be >= (1/n_action)
    yita_max = 0.75
    yita_inc_per_update = 0.0075 # (increase to 0.75 in 500 updates)

class StagePlanner:
    def __init__(self, mcv) -> None:
        if AlgorithmConfig.policy_resonance:
            self.resonance_active = False
            self.yita = 0
            self.yita_min_prob = PolicyRsnConfig.yita_min_prob
        self.freeze_body = False
        self.update_cnt = 0
        self.mcv = mcv
        self.trainer = None
        if AlgorithmConfig.wait_norm_stable:
            self.wait_norm_stable_cnt = 2
        else:
            self.wait_norm_stable_cnt = 0
        return
    
    def is_resonance_active(self,):
        return self.resonance_active
    
    def is_body_freeze(self,):
        return self.freeze_body
    
    def get_yita(self):
        return self.yita
    
    def get_yita_min_prob(self):
        return PolicyRsnConfig.yita_min_prob
    
    def can_exec_trainning(self):
        if self.wait_norm_stable_cnt > 0:
            print亮绿('waiting initial normalization stable, skip training!')
            self.wait_norm_stable_cnt -= 1
            return False
        else:
            return True

    def update_plan(self):
        self.update_cnt += 1
        if AlgorithmConfig.policy_resonance:
            if self.resonance_active:
                self.when_pr_active()
            elif not self.resonance_active:
                self.when_pr_inactive()
        return
    
    def activate_pr(self):
        self.resonance_active = True
        self.freeze_body = True
        self.trainer.freeze_body()

    def when_pr_inactive(self):
        assert not self.resonance_active
        if PolicyRsnConfig.resonance_start_at_update >= 0:
            # mean need to activate pr later
            if self.update_cnt > PolicyRsnConfig.resonance_start_at_update:
                # time is up, activate pr
                self.activate_pr()
        # log
        pr = 1 if self.resonance_active else 0
        self.mcv.rec(pr, 'resonance')
        self.mcv.rec(self.yita, 'self.yita')

    def when_pr_active(self):
        assert self.resonance_active
        self._update_yita()
        # log
        pr = 1 if self.resonance_active else 0
        self.mcv.rec(pr, 'resonance')
        self.mcv.rec(self.yita, 'self.yita')

    def _update_yita(self):
        '''
            increase self.yita by @yita_inc_per_update per function call
        '''
        self.yita += PolicyRsnConfig.yita_inc_per_update
        if self.yita > PolicyRsnConfig.yita_max:
            self.yita = PolicyRsnConfig.yita_max
        print亮绿('yita update:', self.yita)
