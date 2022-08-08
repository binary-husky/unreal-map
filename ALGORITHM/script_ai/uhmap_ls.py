import copy
import numpy as np
from UTIL.tensor_ops import distance_mat_between
from scipy.optimize import linear_sum_assignment
from MISSION.uhmap.actset_lookup import encode_action_as_digits
from config import GlobalConfig

class DummyAlgConfig():
    reserve = ""

class DummyAlgorithmBase():
    def __init__(self, n_agent, n_thread, space, mcv=None, team=None):
        self.n_agent = n_agent
        self.n_thread = n_thread
        self.team = team
        self.ScenarioConfig = GlobalConfig.ScenarioConfig
        self.attack_order = {}
        self.team_agent_uid = GlobalConfig.ScenarioConfig.AGENT_ID_EACH_TEAM[team]

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
                opp_uid_range = GlobalConfig.ScenarioConfig.AGENT_ID_EACH_TEAM[1-self.team]
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








def assign_opponent(opp_pos_arr, opp_id_arr, leader_pos_arr, leader_id_arr):
    result = {}
    dis_mat = distance_mat_between(leader_pos_arr, opp_pos_arr)
    dis_mat[dis_mat == np.inf] = 1e10
    indices, assignments = linear_sum_assignment(dis_mat)
    for i, j, a in zip(range(len(indices)), indices, assignments):
        assert i == j
        result[leader_id_arr[i]] = opp_id_arr[a]
    return result

class DummyAlgorithmLinedAttack(DummyAlgorithmBase):
    def interact_with_env(self, State_Recall):
        assert State_Recall['Latest-Obs'] is not None, ('make sure obs is ok')
        
        ENV_PAUSE = State_Recall['ENV-PAUSE']
        ENV_ACTIVE = ~ENV_PAUSE

        assert self.n_thread == len(ENV_ACTIVE), ('the number of thread is wrong?')
        n_active_thread = sum(ENV_ACTIVE)

        actions = np.zeros(shape=(self.n_thread, self.n_agent, 8 ))

        for thread in range(self.n_thread):
            if ENV_PAUSE[thread]: 
                # 如果,该线程停止，不做任何处理
                continue

            actions[thread] = self.decide_each_thread(
                thread = thread,
                step_cnt = State_Recall['Current-Obs-Step'][thread],
                raw_info = State_Recall['Latest-Team-Info'][thread]['dataArr'],
                Env_Suffered_Reset = State_Recall['Env-Suffered-Reset'][thread]
            )

        # set actions of in-active threads to NaN (will be done again in multi_team.py, this line is not necessary)
        actions[ENV_PAUSE] = np.nan
        # swap (self.n_thread, self.n_agent) -> (self.n_agent, self.n_thread) 
        actions = np.swapaxes(actions, 0, 1)
        return actions, {}

    # 判断agent是否存活
    def uid_alive(raw_info, uid):
        return raw_info[uid]['agentAlive']

    def decide_each_thread(self, **kwargs):
        act_each_agent = np.zeros(shape=( self.n_agent, 8 ))
        self_uid_range = GlobalConfig.ScenarioConfig.AGENT_ID_EACH_TEAM[self.team]

        Env_Suffered_Reset = kwargs['Env_Suffered_Reset']
        thread = kwargs['thread']
        # 当前的Episode步数
        step_cnt = kwargs['step_cnt']
        raw_info = kwargs['raw_info']

        # # 如果,该线程没有停止
        # if Env_Suffered_Reset:
        #     # 如果该线程刚刚reset
        #     opp_uid_range = GlobalConfig.ScenarioConfig.AGENT_ID_EACH_TEAM[1-self.team]
        #     opp_uid_range = list(copy.deepcopy(opp_uid_range))
        #     np.random.shuffle(opp_uid_range)
        #     self.attack_order[thread] = opp_uid_range

        opp_uid_range = GlobalConfig.ScenarioConfig.AGENT_ID_EACH_TEAM[1-self.team]
        pos_arr_2d = np.array([_info['agentLocationArr'][:2] for _info in raw_info])
        opp_pos_arr = pos_arr_2d[opp_uid_range]

        self_air_uid_range = [info['uId'] for info in raw_info if info['agentAlive'] and  info['agentTeam'] == self.team and info['type']=='RLA_UAV_Support']
        N_leader = len(self_air_uid_range)
        self_ground_uid_range = [info['uId'] for info in raw_info if info['agentAlive'] and info['agentTeam'] == self.team and info['type']!='RLA_UAV_Support']
        if N_leader > 0:
            self_air_pos_arr = pos_arr_2d[self_air_uid_range]
            assignments = assign_opponent(
                opp_pos_arr=opp_pos_arr, 
                opp_id_arr=opp_uid_range, 
                leader_pos_arr = self_air_pos_arr, 
                leader_id_arr=self_air_uid_range
            )



            for group in range(N_leader):
                attack_uid = assignments[self_air_uid_range[group]]
                group_member_uids = [uid for uid in self_ground_uid_range if uid%N_leader==group]

                for group_member_uid in group_member_uids:
                    agent_team_index = raw_info[group_member_uid]['indexInTeam']
                    act_each_agent[agent_team_index] = encode_action_as_digits('SpecificAttacking', 'N/A', x=None, y=None, z=None, UID=attack_uid, T=None, T_index=None)

                leader_uid = self_air_uid_range[group]
                agent_team_index = raw_info[leader_uid]['indexInTeam']
                z_leader = raw_info[leader_uid]['agentLocation']['z']
                if len(group_member_uids) > 0:
                    team_center_pos = pos_arr_2d[group_member_uid]
                    act_each_agent[agent_team_index] = encode_action_as_digits('PatrolMoving', 'N/A', x=team_center_pos[0], y=team_center_pos[1], z=z_leader, UID=None, T=None, T_index=None)
                else:
                    act_each_agent[agent_team_index] = encode_action_as_digits('SpecificAttacking', 'N/A', x=None, y=None, z=None, UID=attack_uid, T=None, T_index=None)

            return act_each_agent
        else:
            center_pos_kd = pos_arr_2d[self_ground_uid_range].mean(0, keepdims=True)
            dis = distance_mat_between(center_pos_kd, opp_pos_arr)
            target_index = np.argmin(dis.squeeze())
            attack_uid = opp_uid_range[target_index]

            group_member_uids = self_ground_uid_range

            for group_member_uid in group_member_uids:
                agent_team_index = raw_info[group_member_uid]['indexInTeam']
                act_each_agent[agent_team_index] = encode_action_as_digits('SpecificAttacking', 'N/A', 
                    x=None, y=None, z=None, UID=attack_uid, T=None, T_index=None)

            return act_each_agent

def vector_shift_towards(pos, toward_pos, offset):
    delta = toward_pos - pos 
    delta = delta / (np.linalg.norm(delta) + 1e-10)
    return pos + delta * offset
