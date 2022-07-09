import copy
import numpy as np
from UTIL.tensor_ops import copy_clone
from MISSION.uhmap.actset_lookup import encode_action_as_digits
from config import GlobalConfig

class DummyAlgConfig():
    reserve = ""

class DummyAlgorithmBase():
    def __init__(self, n_agent, n_thread, space, mcv=None, team=None):
        self.n_agent = n_agent
        self.n_thread = n_thread
        self.team = team
        self.scenario_config = GlobalConfig.scenario_config
        self.attack_order = {}
        self.team_agent_uid = GlobalConfig.scenario_config.AGENT_ID_EACH_TEAM[team]

    def forward(self, inp, state, mask=None):
        raise NotImplementedError

    def to(self, device):
        return self
    
    def interact_with_env(self, State_Recall):
        assert State_Recall['Latest-Obs'] is not None, ('make sure obs is ok')
        
        ENV_PAUSE = State_Recall['ENV-PAUSE']
        ENV_ACTIVE = ~ENV_PAUSE

        assert self.n_thread == len(ENV_ACTIVE), ('the number of thread is wrong?')
        
        n_active_thread = sum(ENV_ACTIVE)

        # assert len(State_Recall['Latest-Obs']) == n_active_thread, ('make sure we have the right batch of obs')

        actions = np.zeros(shape=(self.n_thread, self.n_agent, 8))

        # set actions of in-active threads to NaN (will be done again in multi_team.py, this line is not necessary)
        actions[ENV_PAUSE] = np.nan
        
        # swap (self.n_thread, self.n_agent) -> (self.n_agent, self.n_thread) 
        actions = np.swapaxes(actions, 0, 1)
        return actions, {}


class DummyAlgorithmSeqFire(DummyAlgorithmBase):
    def interact_with_env(self, State_Recall):
        assert State_Recall['Latest-Obs'] is not None, ('make sure obs is ok')
        
        ENV_PAUSE = State_Recall['ENV-PAUSE']
        ENV_ACTIVE = ~ENV_PAUSE

        assert self.n_thread == len(ENV_ACTIVE), ('the number of thread is wrong?')
        
        n_active_thread = sum(ENV_ACTIVE)

        # assert len(State_Recall['Latest-Obs']) == n_active_thread, ('make sure we have the right batch of obs')

        actions = np.zeros(shape=(self.n_thread, self.n_agent, 8 ))

        for thread in range(self.n_thread):
            if ENV_PAUSE[thread]: 
                # 如果,该线程停止，不做任何处理
                continue

            # 如果,该线程没有停止
            if State_Recall['Env-Suffered-Reset'][thread]:
                # 如果该线程刚刚reset
                opp_uid_range = GlobalConfig.scenario_config.AGENT_ID_EACH_TEAM[1-self.team]
                opp_uid_range = list(copy.deepcopy(opp_uid_range))
                np.random.shuffle(opp_uid_range)
                self.attack_order[thread] = opp_uid_range
            # 当前的Episode步数
            step_cnt = State_Recall['Current-Obs-Step'][thread]
            # 当前的info
            info = State_Recall['Latest-Team-Info']
            raw_info = State_Recall['Latest-Team-Info'][thread]['dataArr']
            
            # 判断agent是否存活
            def uid_alive(uid):
                return raw_info[uid]['agentAlive']

            for uid in self.attack_order[thread]:
                if uid_alive(uid):
                    # 如果该敌方存活，则集火攻击（:）
                    actions[thread, :] = encode_action_as_digits('SpecificAttacking', 'N/A', x=None, y=None, z=None, UID=uid, T=None, T_index=None)



        # set actions of in-active threads to NaN (will be done again in multi_team.py, this line is not necessary)
        actions[ENV_PAUSE] = np.nan
        # swap (self.n_thread, self.n_agent) -> (self.n_agent, self.n_thread) 
        actions = np.swapaxes(actions, 0, 1)
        return actions, {}




class DummyAlgorithmIdle(DummyAlgorithmBase):
    def interact_with_env(self, State_Recall):
        assert State_Recall['Latest-Obs'] is not None, ('make sure obs is ok')

        ENV_PAUSE = State_Recall['ENV-PAUSE']
        ENV_ACTIVE = ~ENV_PAUSE

        assert self.n_thread == len(ENV_ACTIVE), ('the number of thread is wrong?')

        n_active_thread = sum(ENV_ACTIVE)
        AirCarrierUID = 2

        # assert len(State_Recall['Latest-Obs']) == n_active_thread, ('make sure we have the right batch of obs')

        actions = np.zeros(shape=(self.n_thread, self.n_agent, 8 ))

        for thread in range(self.n_thread):
            if ENV_PAUSE[thread]: 
                # 如果,该线程停止，不做任何处理
                continue

            if State_Recall['Env-Suffered-Reset'][thread]:
                actions[thread, :] = encode_action_as_digits('Idle', 'AggressivePersue', x=None, y=None, z=None, UID=None, T=None, T_index=None)
            else:
                actions[thread, :] = encode_action_as_digits('N/A', 'N/A', x=None, y=None, z=None, UID=None, T=None, T_index=None)



        # set actions of in-active threads to NaN (will be done again in multi_team.py, this line is not necessary)
        actions[ENV_PAUSE] = np.nan
        # swap (self.n_thread, self.n_agent) -> (self.n_agent, self.n_thread) 
        actions = np.swapaxes(actions, 0, 1)
        return actions, {}



class DummyAlgorithmMarch(DummyAlgorithmBase):
    def interact_with_env(self, State_Recall):
        assert State_Recall['Latest-Obs'] is not None, ('make sure obs is ok')

        ENV_PAUSE = State_Recall['ENV-PAUSE']
        ENV_ACTIVE = ~ENV_PAUSE

        assert self.n_thread == len(ENV_ACTIVE), ('the number of thread is wrong?')

        n_active_thread = sum(ENV_ACTIVE)
        AirCarrierUID = 2

        # assert len(State_Recall['Latest-Obs']) == n_active_thread, ('make sure we have the right batch of obs')

        actions = np.zeros(shape=(self.n_thread, self.n_agent, 8 ))


        if not hasattr(self, 'march_direction'):
            self.march_direction = '+Y'

        for thread in range(self.n_thread):
            if ENV_PAUSE[thread]: 
                # 如果,该线程停止，不做任何处理
                continue

            if State_Recall['Env-Suffered-Reset'][thread]:
                a_agent_uid = self.team_agent_uid[0]
                self.march_direction = '+Y' if State_Recall['Latest-Team-Info'][thread]['dataArr'][a_agent_uid]['agentLocation']['y'] <0 else '-Y'
                actions[thread, :] = encode_action_as_digits('Idle', 'AggressivePersue', x=None, y=None, z=None, UID=None, T=None, T_index=None)

            else:

                if self.march_direction == '+Y':
                    actions[thread, :] = encode_action_as_digits('PatrolMoving', 'Dir+Y', x=None, y=None, z=None, UID=None, T=None, T_index=None)
                else:
                    actions[thread, :] = encode_action_as_digits('PatrolMoving', 'Dir-Y', x=None, y=None, z=None, UID=None, T=None, T_index=None)



        # set actions of in-active threads to NaN (will be done again in multi_team.py, this line is not necessary)
        actions[ENV_PAUSE] = np.nan
        # swap (self.n_thread, self.n_agent) -> (self.n_agent, self.n_thread) 
        actions = np.swapaxes(actions, 0, 1)
        return actions, {}