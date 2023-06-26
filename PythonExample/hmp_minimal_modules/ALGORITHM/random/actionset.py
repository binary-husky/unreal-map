# ====================================================================
# random moving 
# ====================================================================
import numpy as np
from MISSION.uhmap.actionset_v3 import strActionToDigits, ActDigitLen
from MISSION.uhmap.actset_lookup import encode_action_as_digits

class ActionConvertMovingV4():
    def __init__(self, SELF_TEAM_ASSUME, OPP_TEAM_ASSUME, OPP_NUM_ASSUME) -> None:
        self.dictionary_args = [
            'ActionSet4::MoveToDirection;X=1.0 Y=0.0 Z=0.0',
            'ActionSet4::MoveToDirection;X=1.0 Y=1.0 Z=0.0',
            'ActionSet4::MoveToDirection;X=0.0 Y=1.0 Z=0.0',
            'ActionSet4::MoveToDirection;X=-1.0 Y=1.0 Z=0.0',
            'ActionSet4::MoveToDirection;X=-1.0 Y=0.0 Z=0.0',
            'ActionSet4::MoveToDirection;X=-1.0 Y=-1.0 Z=0.0',
            'ActionSet4::MoveToDirection;X=0.0 Y=-1.0 Z=0.0',
            'ActionSet4::MoveToDirection;X=1.0 Y=-1.0 Z=0.0',
        ] 
        self.n_act = len(self.dictionary_args)
        self.ActDigitLen = ActDigitLen

    def convert_act_arr(self, type, a):
        return strActionToDigits(self.dictionary_args[a])

    def get_tp_avail_act(self, type):
        DISABLE = 0; ENABLE = 1
        ret = np.zeros(self.n_act) + ENABLE  # enable all
        return ret

class ActionConvertV2():
    def __init__(self, SELF_TEAM_ASSUME, OPP_TEAM_ASSUME, OPP_NUM_ASSUME) -> None:
        self.SELF_TEAM_ASSUME = SELF_TEAM_ASSUME
        self.OPP_TEAM_ASSUME = OPP_TEAM_ASSUME
        self.OPP_NUM_ASSUME = OPP_NUM_ASSUME
        # (main_cmd, sub_cmd, x=None, y=None, z=None, UID=None, T=None, T_index=None)
        self.dictionary_args = [
            'ActionSet2::N/A;N/A'                        ,
            'ActionSet2::Idle;DynamicGuard'              ,
            'ActionSet2::Idle;StaticAlert'               ,
            'ActionSet2::Idle;AggressivePersue'          ,
            'ActionSet2::Idle;AsFarAsPossible'           ,
            'ActionSet2::Idle;StayWhenTargetInRange'     ,
            'ActionSet2::Idle;StayWhenTargetInHalfRange' ,
            'ActionSet2::SpecificMoving;Dir+X'           ,
            'ActionSet2::SpecificMoving;Dir+Y'           ,
            'ActionSet2::SpecificMoving;Dir-X'           ,
            'ActionSet2::SpecificMoving;Dir-Y'           ,
            'ActionSet2::PatrolMoving;Dir+X'             ,
            'ActionSet2::PatrolMoving;Dir+Y'             ,
            'ActionSet2::PatrolMoving;Dir-X'             ,
            'ActionSet2::PatrolMoving;Dir-Y'             ,
        ]
        for i in range(self.OPP_NUM_ASSUME):
            self.dictionary_args.append( f'ActionSet2::SpecificAttacking;T{OPP_TEAM_ASSUME}-{i}')
        self.ActDigitLen = ActDigitLen
        self.n_act = len(self.dictionary_args)

    def convert_act_arr(self, type, a):
        return strActionToDigits(self.dictionary_args[a])

    def get_tp_avail_act(self, type):
        DISABLE = 0
        ENABLE = 1
        n_act = len(self.dictionary_args)
        ret = np.zeros(n_act) + ENABLE
        for i in range(n_act):
            args = self.dictionary_args[i]
            
            # for all kind of agents
            if args[0] == 'PatrolMoving':       ret[i] = DISABLE
            
            if type == 'RLA_UAV_Support':
                if args[0] == 'PatrolMoving':       ret[i] = DISABLE
                if args[0] == 'SpecificAttacking':  ret[i] = DISABLE
                if args[0] == 'Idle':               ret[i] = DISABLE
                if args[1] == 'StaticAlert':        ret[i] = ENABLE
        return ret

    def confirm_parameters_are_correct(self, team, agent_num, opp_agent_num):
        assert team == self.SELF_TEAM_ASSUME
        assert self.SELF_TEAM_ASSUME + self.OPP_TEAM_ASSUME == 1
        assert self.SELF_TEAM_ASSUME + self.OPP_TEAM_ASSUME == 1
        assert opp_agent_num == self.OPP_NUM_ASSUME
    


