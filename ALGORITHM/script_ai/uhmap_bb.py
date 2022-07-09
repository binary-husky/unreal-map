import copy
from math import sqrt
import numpy as np
from MISSION.uhmap.actset_lookup import encode_action_as_digits
from config import GlobalConfig

class DummyAlgConfig():
    reserve = ""

class DummyAlgorithmBase():
    def __init__(self, n_agent, n_thread, space, mcv=None, team=None):
        self.n_agent = n_agent
        self.n_thread = n_thread
        self.scenario_config = GlobalConfig.scenario_config
        self.attack_order = {}

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


class DummyAlgorithmT2(DummyAlgorithmBase):
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


            AirCarrier = State_Recall['Latest-Team-Info'][thread]['dataArr'][AirCarrierUID]
            if AirCarrier['agentAlive']:
                assert 'RLA_UAV' in AirCarrier['type'] 
                landmarks = State_Recall['Latest-Team-Info'][thread]['dataGlobal']['keyObjArr']

                squredis = lambda a,b: sqrt(
                    (a['agentLocation']['x']-b['location']['x'])**2 + 
                    (a['agentLocation']['y']-b['location']['y'])**2 + 
                    (a['agentLocation']['z']-b['location']['z'])**2 )
                AirCarrirSquareDisToEachLandmark = [squredis(AirCarrier, landmark) for landmark in landmarks]
                nearLandmark = np.argmin(AirCarrirSquareDisToEachLandmark)

                pos_lm = np.array([
                    landmarks[nearLandmark]['location']['x'],
                    landmarks[nearLandmark]['location']['y'],
                    landmarks[nearLandmark]['location']['z'],
                ])

                pos_ac_proj = np.array([
                    AirCarrier['agentLocation']['x'],
                    AirCarrier['agentLocation']['y'],
                    landmarks[nearLandmark]['location']['z'],
                ])

                unit_2ac_prj = (pos_ac_proj - pos_lm) / np.linalg.norm(pos_ac_proj - pos_lm)
                p = unit_2ac_prj*400 + pos_lm

      
                actions[thread, :] = encode_action_as_digits('PatrolMoving', 'N/A', x=p[0], y=p[1], z=p[2], UID=None, T=None, T_index=None)
            else:
                actions[thread, :] = encode_action_as_digits('N/A', 'N/A', x=None, y=None, z=None, UID=None, T=None, T_index=None)
            

        # set actions of in-active threads to NaN (will be done again in multi_team.py, this line is not necessary)
        actions[ENV_PAUSE] = np.nan
        # swap (self.n_thread, self.n_agent) -> (self.n_agent, self.n_thread) 
        actions = np.swapaxes(actions, 0, 1)
        return actions, {}

class DummyAlgorithmT1(DummyAlgorithmBase):
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
            landmarks = State_Recall['Latest-Team-Info'][thread]['dataGlobal']['keyObjArr']
            px = landmarks[0]['location']['x']
            py = landmarks[0]['location']['y']
            for a in range(self.n_agent):
                if not State_Recall['Latest-Team-Info'][thread]['dataArr'][a]['agentAlive']: continue
                pz = State_Recall['Latest-Team-Info'][thread]['dataArr'][a]['agentLocation']['z']
                
                actions[thread, a] = encode_action_as_digits('SpecificMoving', 'N/A', x=px, y=py, z=pz, UID=None, T=None, T_index=None)



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


            # AirCarrier = State_Recall['Latest-Team-Info'][thread]['dataArr'][AirCarrierUID]
            # if AirCarrier['agentAlive']:
            #     assert 'RLA_UAV' in AirCarrier['type'] 
            #     landmarks = State_Recall['Latest-Team-Info'][thread]['dataGlobal']['keyObjArr']

            #     squredis = lambda a,b: sqrt(
            #         (a['agentLocation']['x']-b['location']['x'])**2 + 
            #         (a['agentLocation']['y']-b['location']['y'])**2 + 
            #         (a['agentLocation']['z']-b['location']['z'])**2 )
            #     AirCarrirSquareDisToEachLandmark = [squredis(AirCarrier, landmark) for landmark in landmarks]
            #     nearLandmark = np.argmin(AirCarrirSquareDisToEachLandmark)

            #     px = landmarks[nearLandmark]['location']['x']
            #     py = landmarks[nearLandmark]['location']['y']
            #     pz = landmarks[nearLandmark]['location']['z']
            #     actions[thread, :] = encode_action_as_digits('PatrolMoving', 'N/A', x=px, y=py, z=pz, UID=None, T=None, T_index=None)
            # else:
            #     actions[thread, :] = encode_action_as_digits('N/A', 'N/A', x=None, y=None, z=None, UID=None, T=None, T_index=None)
            

            if State_Recall['Env-Suffered-Reset'][thread]:
                actions[thread, :] = encode_action_as_digits('N/A', 'N/A', x=None, y=None, z=None, UID=None, T=None, T_index=None)
            else:
                actions[thread, :] = encode_action_as_digits('Idle', 'StaticAlert', x=None, y=None, z=None, UID=None, T=None, T_index=None)



        # set actions of in-active threads to NaN (will be done again in multi_team.py, this line is not necessary)
        actions[ENV_PAUSE] = np.nan
        # swap (self.n_thread, self.n_agent) -> (self.n_agent, self.n_thread) 
        actions = np.swapaxes(actions, 0, 1)
        return actions, {}