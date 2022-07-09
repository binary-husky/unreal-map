import numpy as np
from UTIL.tensor_ops import copy_clone
from MISSION.uhmap.actset_lookup import encode_action_as_digits
from ALGORITHM.script_ai.decision import decision
from ALGORITHM.script_ai.assignment import *
from ALGORITHM.script_ai.global_params import *

attact_states={'211':['idle'],'221':['idle']}
drone_state={'231':['idle']}
defend_states={'311':['idle'],'321':['idle']}
print("===========================")
print(attact_states)
print(drone_state)
print(defend_states)
print("===========================")
class DummyAlgConfig():
    reserve = ""


class DummyAlgorithmBase():
    def __init__(self, n_agent, n_thread, space, mcv=None, team=None):
        from config import GlobalConfig
        self.n_agent = n_agent
        self.n_thread = n_thread
        self.scenario_config = GlobalConfig.scenario_config

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

        assert len(State_Recall['Latest-Obs']) == n_active_thread, ('make sure we have the right batch of obs')

        actions = np.zeros(shape=(self.n_thread, self.n_agent))

        # set actions of in-active threads to NaN (will be done again in multi_team.py, this line is not necessary)
        actions[ENV_PAUSE] = np.nan

        # swap (self.n_thread, self.n_agent) -> (self.n_agent, self.n_thread)
        actions = np.swapaxes(actions, 0, 1)
        return actions, {}


# 进攻方决策
class DummyAlgorithmT1(DummyAlgorithmBase):
    def interact_with_env(self, State_Recall):
        try:
            res = self.interact_with_env_(State_Recall)
        except:
            actions = np.zeros(shape=(self.n_thread, self.n_agent, 8))
            actions[:] = encode_action_as_digits("N/A", "N/A", x=None, y=None, z=None, UID=None, T=None, T_index=None)
            actions = np.swapaxes(actions, 0, 1)
            res = (actions, None)
        return res


    def interact_with_env_(self, State_Recall):
        assert State_Recall['Latest-Obs'] is not None, ('make sure obs is ok')

        ENV_PAUSE = State_Recall['ENV-PAUSE']
        ENV_ACTIVE = ~ENV_PAUSE

        assert self.n_thread == len(ENV_ACTIVE), ('the number of thread is wrong?')

        n_active_thread = sum(ENV_ACTIVE)

        assert len(State_Recall['Latest-Obs']) == n_active_thread, ('make sure we have the right batch of obs')

        actions = np.zeros(shape=(self.n_thread, self.n_agent, 8))

        env0_step = State_Recall['Current-Obs-Step']

        obs = State_Recall['Latest-Team-Info']
        thread = 0
        global attact_states
        global drone_state
        global defend_states
        if State_Recall['Env-Suffered-Reset']==[True]:
             attact_states = {'211': ['idle'], '221': ['idle']}
             drone_state = {'231': ['idle']}
             defend_states = {'311': ['idle'], '321': ['idle']}
        # 防守方red小车的信息
        red_agents_data = {'311': {'ammo':100, 'velocity':0, 'X': obs[0]['dataArr'][3]['agentLocation']['x'],
                                   'Y': obs[0]['dataArr'][3]['agentLocation']['y'],
                                   'Z': obs[0]['dataArr'][3]['agentLocation']['z'],
                                   'Yaw': 0,
                                   'blood': obs[0]['dataArr'][3]['agentHp'], 'state':defend_states['311']},
                           '321': {'ammo':100, 'velocity':0, 'X': obs[0]['dataArr'][4]['agentLocation']['x'],
                                   'Y': obs[0]['dataArr'][4]['agentLocation']['y'],
                                   'Z': obs[0]['dataArr'][4]['agentLocation']['z'],
                                   'Yaw': 0,
                                   'blood': obs[0]['dataArr'][4]['agentHp'], 'state':defend_states['321']}}
        # 进攻方blue小车和无人机信息，其中231是无人机
        blue_agents_data = {'211': {'ammo':100, 'velocity':0, 'X': obs[0]['dataArr'][0]['agentLocation']['x'],
                                   'Y': obs[0]['dataArr'][0]['agentLocation']['y'],
                                   'Z': obs[0]['dataArr'][0]['agentLocation']['z'],
                                   'Yaw': 0,
                                   'blood': obs[0]['dataArr'][0]['agentHp'], 'state':attact_states['211']},
                            '221': {'ammo':100, 'velocity':0, 'X': obs[0]['dataArr'][1]['agentLocation']['x'],
                                   'Y': obs[0]['dataArr'][1]['agentLocation']['y'],
                                   'Z': obs[0]['dataArr'][1]['agentLocation']['z'],
                                   'Yaw': 0,
                                   'blood': obs[0]['dataArr'][1]['agentHp'], 'state':attact_states['221']}}
        drone_data={'ammo':100, 'velocity':0, 'X': obs[0]['dataArr'][2]['agentLocation']['x'],
                                   'Y': obs[0]['dataArr'][2]['agentLocation']['y'],
                                   'Z': obs[0]['dataArr'][2]['agentLocation']['z'],
                                   'Yaw': 0,
                                   'blood': obs[0]['dataArr'][2]['agentHp'], 'state':drone_state['231']}

        blue_alive = [obs[0]['dataArr'][0]['agentAlive'], obs[0]['dataArr'][1]['agentAlive']]
        red_alive = [obs[0]['dataArr'][3]['agentAlive'], obs[0]['dataArr'][4]['agentAlive']]
        # 夺控点信息 在global——params.py修改
       # key_points = [[700, -3300, 500], [-3000, 700, 500]]
        DecisionMake = decision(blue_agents_data,drone_data,red_agents_data)
        attackers = DecisionMake.attackers
        defenders = DecisionMake.defenders
        drone = DecisionMake.drone
        #decision module test
        attack_actions = DecisionMake.act(type='attackers')
        defend_actions = DecisionMake.act(type='defenders')

        att_states = []
        def_states = []
        for k, v in attackers.items():
            att_states.append({k: v['state']})
        for k, v in defenders.items():
            def_states.append({k: v['state']})
        # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        # print(drone['state'])
        # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

        attact_states['211']=att_states[0]['211']
        attact_states['221'] = att_states[1]['221']
        defend_states['311'] = def_states[0]['311']
        defend_states['321'] = def_states[1]['321']
        drone_state['231'] = drone['state']

        print('+++++++++++++++++++++++++ info +++++++++++++++++++++++++++++++++')
        print('211 state: ', attact_states['211'])
        print('221 state: ', attact_states['221'])
        print('311 state: ', defend_states['311'])
        print('321 state: ', defend_states['321'])
        print('drone state: ', drone['state'])
        print('attack actions: ', attack_actions)
        print('defend actions: ', defend_actions)
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        # 小车211决策
        if attact_states['211'][0] == 'attack':
            if attact_states['211'][1] == '311':
                # actions[thread, 0] = encode_action_as_digits("SpecificAttacking", "N/A", x=None, y=None, z=None,UID=3,T=None, T_index=None)
                actions[thread, 0] = encode_action_as_digits("SpecificMoving", "N/A", x=attack_actions['211'][0], y=attack_actions['221'][1], z=500,UID=None,T=None, T_index=None)
            else:
                # actions[thread, 0] = encode_action_as_digits("SpecificAttacking", "N/A", x=None, y=None, z=None, UID=4,T=None, T_index=None)
                actions[thread, 0] = encode_action_as_digits("SpecificMoving", "N/A", x=attack_actions['211'][0], y=attack_actions['211'][1],z=500,UID=None,T=None, T_index=None)
        else:
            actions[thread, 0] = encode_action_as_digits("SpecificMoving", "N/A", x=attack_actions['211'][0], y=attack_actions['211'][1],z=500,UID=None,T=None, T_index=None)

        # 小车221决策
        if attact_states['221'][0] == 'attack':
            if attact_states['221'][1] == '311':
                # actions[thread, 1] = encode_action_as_digits("SpecificAttacking", "N/A", x=None, y=None, z=None, UID=3,T=None, T_index=None)
                actions[thread, 1] = encode_action_as_digits("SpecificMoving", "N/A", x=attack_actions['221'][0], y=attack_actions['221'][1],z=500,UID=None,T=None, T_index=None)
            else:
                # actions[thread, 1] = encode_action_as_digits("SpecificAttacking", "N/A", x=None, y=None, z=None, UID=4,T=None, T_index=None)
                actions[thread, 1] = encode_action_as_digits("SpecificMoving", "N/A", x=attack_actions['221'][0], y=attack_actions['221'][1],z=500,UID=None,T=None, T_index=None)
        else:
            actions[thread, 1] = encode_action_as_digits("SpecificMoving", "N/A", x=attack_actions['221'][0], y=attack_actions['221'][1],z=500,UID=None,T=None, T_index=None)
        # 无人机231决策
        actions[thread, 2] = encode_action_as_digits("SpecificMoving", "N/A", x=attack_actions['drone'][0],
                                                     y=attack_actions['drone'][1],
                                                     z=500,
                                                     UID=None,
                                                     T=None, T_index=None)

        # if env0_step < 2:
        # actions[thread, :] = self.act2digit_dictionary['ActionSet2::Idle;DynamicGuard']
        # actions[thread, 0] = encode_action_as_digits("SpecificAttacking", "N/A", x=None, y=None, z=None, UID=4, T=None, T_index=None)
        # actions[thread, 0] = encode_action_as_digits("PatrolMoving", "N/A", x=0, y=0, z=379, UID=None, T=None, T_index=None)
        # actions[thread, 1] = encode_action_as_digits("SpecificAttacking", "N/A", x=None, y=None, z=None, UID=3, T=None, T_index=None)
        # actions[thread, 2] = encode_action_as_digits("SpecificMoving", "N/A", x=-3000, y=700, z=500, UID=None,T=None, T_index=None)
        # actions[thread, 2] = encode_action_as_digits("Idle", "DynamicGuard", x=700, y=-3300, z=500, UID=None, T=None, T_index=None)
        """
        if env0_step%4 == 0:
            actions[thread, 2] = encode_action_as_digits("SpecificMoving", "Dir+X+Y", x=700, y=-3300, z=500, UID=None, T=None, T_index=None)
        if env0_step%4 == 1:
            actions[thread, 2] = encode_action_as_digits("SpecificMoving", "Dir+X-Y", x=700, y=-3300, z=500, UID=None, T=None, T_index=None)
        if env0_step%4 == 2:
            actions[thread, 2] = encode_action_as_digits("SpecificMoving", "Dir-X-Y", x=700, y=-3300, z=500, UID=None, T=None, T_index=None)
        if env0_step%4 == 3:
            actions[thread, 2] = encode_action_as_digits("SpecificMoving", "Dir-X+Y", x=700, y=-3300, z=500, UID=None, T=None, T_index=None)
        """
        # set actions of in-active threads to NaN (will be done again in multi_team.py, this line is not necessary)
        actions[ENV_PAUSE] = np.nan

        # swap (self.n_thread, self.n_agent) -> (self.n_agent, self.n_thread)
        actions = np.swapaxes(actions, 0, 1)
        return actions, {}


# 防守方决策
class DummyAlgorithmT2(DummyAlgorithmBase):
    def interact_with_env(self, State_Recall):
        try:
            res = self.interact_with_env_(State_Recall)
        except:
            actions = np.zeros(shape=(self.n_thread, self.n_agent, 8))
            actions[:] = encode_action_as_digits("N/A", "N/A", x=None, y=None, z=None, UID=None, T=None, T_index=None)
            actions = np.swapaxes(actions, 0, 1)
            res = (actions, None)
        return res


    def interact_with_env_(self, State_Recall):
        assert State_Recall['Latest-Obs'] is not None, ('make sure obs is ok')

        ENV_PAUSE = State_Recall['ENV-PAUSE']
        ENV_ACTIVE = ~ENV_PAUSE

        assert self.n_thread == len(ENV_ACTIVE), ('the number of thread is wrong?')

        n_active_thread = sum(ENV_ACTIVE)

        assert len(State_Recall['Latest-Obs']) == n_active_thread, ('make sure we have the right batch of obs')

        actions = np.zeros(shape=(self.n_thread, self.n_agent, 8))

        env0_step = State_Recall['Current-Obs-Step']

        obs = State_Recall['Latest-Team-Info']
        thread = 0


        # 防守方red小车的信息
        red_agents_data = {'311': {'ammo': 100, 'velocity': 0, 'X': obs[0]['dataArr'][3]['agentLocation']['x'],
                                   'Y': obs[0]['dataArr'][3]['agentLocation']['y'],
                                   'Z': obs[0]['dataArr'][3]['agentLocation']['z'],
                                   'Yaw': 0,
                                   'blood': obs[0]['dataArr'][3]['agentHp'], 'state': defend_states['311']},
                           '321': {'ammo': 100, 'velocity': 0, 'X': obs[0]['dataArr'][4]['agentLocation']['x'],
                                   'Y': obs[0]['dataArr'][4]['agentLocation']['y'],
                                   'Z': obs[0]['dataArr'][4]['agentLocation']['z'],
                                   'Yaw': 0,
                                   'blood': obs[0]['dataArr'][4]['agentHp'], 'state': defend_states['321']}}
        # 进攻方blue小车和无人机信息，其中231是无人机
        blue_agents_data = {'211': {'ammo': 100, 'velocity': 0, 'X': obs[0]['dataArr'][0]['agentLocation']['x'],
                                    'Y': obs[0]['dataArr'][0]['agentLocation']['y'],
                                    'Z': obs[0]['dataArr'][0]['agentLocation']['z'],
                                    'Yaw': 0,
                                    'blood': obs[0]['dataArr'][0]['agentHp'], 'state': attact_states['211']},
                            '221': {'ammo': 100, 'velocity': 0, 'X': obs[0]['dataArr'][1]['agentLocation']['x'],
                                    'Y': obs[0]['dataArr'][1]['agentLocation']['y'],
                                    'Z': obs[0]['dataArr'][1]['agentLocation']['z'],
                                    'Yaw': 0,
                                    'blood': obs[0]['dataArr'][1]['agentHp'], 'state': attact_states['221']}}
        drone_data =  {'ammo': 100, 'velocity': 0, 'X': obs[0]['dataArr'][2]['agentLocation']['x'],
                              'Y': obs[0]['dataArr'][2]['agentLocation']['y'],
                              'Z': obs[0]['dataArr'][2]['agentLocation']['z'],
                              'Yaw': 0,
                              'blood': obs[0]['dataArr'][2]['agentHp'], 'state': drone_state['231']}

        blue_alive = [obs[0]['dataArr'][0]['agentAlive'], obs[0]['dataArr'][1]['agentAlive']]
        red_alive = [obs[0]['dataArr'][3]['agentAlive'], obs[0]['dataArr'][4]['agentAlive']]
        # 夺控点信息 在global——params.py修改
        # key_points = [[700, -3300, 500], [-3000, 700, 500]]
        DecisionMake = decision(blue_agents_data, drone_data, red_agents_data)
        attackers = DecisionMake.attackers
        defenders = DecisionMake.defenders
        drone = DecisionMake.drone
        # decision module test
        attack_actions = DecisionMake.act(type='attackers')
        defend_actions = DecisionMake.act(type='defenders')

        att_states = []
        def_states = []
        for k, v in attackers.items():
            att_states.append({k: v['state']})
        for k, v in defenders.items():
            def_states.append({k: v['state']})
        attact_states['211'] = att_states[0]['211']
        attact_states['221'] = att_states[1]['221']
        defend_states['311'] = def_states[0]['311']
        defend_states['321'] = def_states[1]['321']
        drone_state['231'] = drone['state']

        # print("==============智能体速度位置信息======================")
        # print(red_agents_data)
        # print(blue_agents_data)
        # print(drone_data)
        # print("====================================")
        print("==============智能体state信息======================")
        print(attact_states)
        print(defend_states)
        # print(drone_state)
        print("====================================")

        # 小车311决策
        if defend_states['311'][0] == 'attack':
            if defend_states['311'][1] == '211':
                # actions[thread, 0] = encode_action_as_digits("SpecificAttacking", "N/A", x=None, y=None, z=None, UID=0,T=None, T_index=None)
                actions[thread, 0] = encode_action_as_digits("SpecificMoving", "N/A", x=defend_actions['311'][0], y=defend_actions['311'][1],z=500,UID=None,T=None, T_index=None)
            else:
                # actions[thread, 0] = encode_action_as_digits("SpecificAttacking", "N/A", x=None, y=None, z=None, UID=1,T=None, T_index=None)
                actions[thread, 1] = encode_action_as_digits("SpecificMoving", "N/A", x=defend_actions['311'][0], y=defend_actions['311'][1],z=500,UID=None,T=None, T_index=None)
        else:
            actions[thread, 0] = encode_action_as_digits("SpecificMoving", "N/A", x=defend_actions['311'][0], y=defend_actions['311'][1],z=500,UID=None,T=None, T_index=None)


        # 小车321决策
        if defend_states['321'][0] == 'attack':
            if defend_states['321'][1] == '211':
                # actions[thread, 1] = encode_action_as_digits("SpecificAttacking", "N/A", x=None, y=None, z=None, UID=0,T=None, T_index=None)
                actions[thread, 1] = encode_action_as_digits("SpecificMoving", "N/A", x=defend_actions['321'][0], y=defend_actions['321'][1],z=500,UID=None,T=None, T_index=None)
            else:
                # actions[thread, 1] = encode_action_as_digits("SpecificAttacking", "N/A", x=None, y=None, z=None, UID=1,T=None, T_index=None)
                actions[thread, 1] = encode_action_as_digits("SpecificMoving", "N/A", x=defend_actions['321'][0], y=defend_actions['321'][1],z=500,UID=None,T=None, T_index=None)
        else:
            actions[thread, 1] = encode_action_as_digits("SpecificMoving", "N/A", x=defend_actions['321'][0], y=defend_actions['321'][1],z=500,UID=None,T=None, T_index=None)


        # actions[thread, 0] = encode_action_as_digits("SpecificAttacking", "N/A", x=None, y=None, z=None, UID=1, T=None, T_index=None)
        # actions[thread, 0] = encode_action_as_digits("Idle", "AggressivePersue", x=10000, y=-10000, z=379, UID=None, T=None, T_index=None)
        # actions[thread, 0] = encode_action_as_digits("SpecificMoving", "N/A", x=10000, y=-10000, z=379, UID=None, T=None, T_index=None)
        # actions[thread, 1] = encode_action_as_digits("PatrolMoving", "N/A", x=444*5, y=444*5, z=379, UID=None, T=None, T_index=None)

        # set actions of in-active threads to NaN (will be done again in multi_team.py, this line is not necessary)
        actions[ENV_PAUSE] = np.nan

        # swap (self.n_thread, self.n_agent) -> (self.n_agent, self.n_thread)
        actions = np.swapaxes(actions, 0, 1)
        return actions, {}


'''
        if env0_step < 5:
            actions[thread, :] = self.act2digit_dictionary['ActionSet2::Idle;DynamicGuard']
        elif env0_step < 15:
            actions[thread, :] = self.act2digit_dictionary['ActionSet2::SpecificAttacking;UID-3']
        elif env0_step < 25:
            actions[thread, :] = self.act2digit_dictionary['ActionSet2::SpecificAttacking;UID-4']
        elif env0_step < 35:
            actions[thread, :] = self.act2digit_dictionary['ActionSet2::SpecificAttacking;UID-5']
        elif env0_step < 45:
            actions[thread, :] = self.act2digit_dictionary['ActionSet2::SpecificAttacking;UID-6']
        elif env0_step < 55:
            actions[thread, :] = self.act2digit_dictionary['ActionSet2::SpecificAttacking;UID-7']
'''

'''
        if env0_step < 5:
            actions[thread, 0] = self.act2digit_dictionary['ActionSet2::Idle;DynamicGuard']
        else:
            actions[thread, 0] = self.act2digit_dictionary['ActionSet2::SpecificAttacking;UID-1']


'''

'''
    if env0_step < 5:
        if env0_step%4 == 0:
            actions[thread, 2] = self.act2digit_dictionary['ActionSet2::PatrolMoving;Dir+X+Y']
        if env0_step%4 == 1:
            actions[thread, 2] = self.act2digit_dictionary['ActionSet2::PatrolMoving;Dir+X-Y']
        if env0_step%4 == 2:
            actions[thread, 2] = self.act2digit_dictionary['ActionSet2::PatrolMoving;Dir-X-Y']
        if env0_step%4 == 3:
            actions[thread, 2] = self.act2digit_dictionary['ActionSet2::PatrolMoving;Dir-X+Y']
    elif env0_step < 10:
        actions[thread, 2] = self.act2digit_dictionary['ActionSet2::Idle;DynamicGuard']
    elif env0_step < 15:
        if env0_step%4 == 0:
            actions[thread, 2] = self.act2digit_dictionary['ActionSet2::SpecificMoving;Dir+X']
        if env0_step%4 == 1:
            actions[thread, 2] = self.act2digit_dictionary['ActionSet2::SpecificMoving;Dir+Y']
        if env0_step%4 == 2:
            actions[thread, 2] = self.act2digit_dictionary['ActionSet2::SpecificMoving;Dir-X']
        if env0_step%4 == 3:
            actions[thread, 2] = self.act2digit_dictionary['ActionSet2::SpecificMoving;Dir-Y']
    elif env0_step < 20:
        actions[thread, 0] = self.act2digit_dictionary['ActionSet2::Idle;StaticAlert']
    elif env0_step < 30:
        if env0_step%4 == 0:
            actions[thread, 2] = self.act2digit_dictionary['ActionSet2::PatrolMoving;Dir+X+Y']
        if env0_step%4 == 1:
            actions[thread, 2] = self.act2digit_dictionary['ActionSet2::PatrolMoving;Dir+X-Y']
        if env0_step%4 == 2:
            actions[thread, 2] = self.act2digit_dictionary['ActionSet2::PatrolMoving;Dir-X-Y']
        if env0_step%4 == 3:
            actions[thread, 2] = self.act2digit_dictionary['ActionSet2::PatrolMoving;Dir-X+Y']
    else:
        actions[thread, 0] = self.act2digit_dictionary['ActionSet2::Idle;StaticAlert']

'''

"""
        thread = 0
        if env0_step%4 == 0:
            actions[thread, 0] = self.act2digit_dictionary['ActionSet2::PatrolMoving;Dir+X+Y']
        if env0_step%4 == 1:
            actions[thread, 0] = self.act2digit_dictionary['ActionSet2::PatrolMoving;Dir+X-Y']
        if env0_step%4 == 2:
            actions[thread, 0] = self.act2digit_dictionary['ActionSet2::PatrolMoving;Dir-X-Y']
        if env0_step%4 == 3:
            actions[thread, 0] = self.act2digit_dictionary['ActionSet2::PatrolMoving;Dir-X+Y']
"""

"""
        thread = 0
        if env0_step%4 == 0:
            actions[thread, 0] = self.act2digit_dictionary['ActionSet2::SpecificMoving;Dir+X+Y']
        if env0_step%4 == 1:
            actions[thread, 0] = self.act2digit_dictionary['ActionSet2::SpecificMoving;Dir+X-Y']
        if env0_step%4 == 2:
            actions[thread, 0] = self.act2digit_dictionary['ActionSet2::SpecificMoving;Dir-X-Y']
        if env0_step%4 == 3:
            actions[thread, 0] = self.act2digit_dictionary['ActionSet2::SpecificMoving;Dir-X+Y']
"""