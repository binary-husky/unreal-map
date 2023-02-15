import numpy as np
from config import GlobalConfig
from UTIL.colorful import *
from UTIL.tensor_ops import my_view, __hash__, repeat_at, gather_righthand
from MISSION.uhmap.actset_lookup import encode_action_as_digits
from MISSION.uhmap.actionset_v3 import strActionToDigits, ActDigitLen
from .foundation import AlgorithmConfig
from .cython_func import roll_hisory

class ShellEnvConfig:
    add_avail_act = False

class ActionConvertPredatorPrey():
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

    def convert_act_arr(self, type, a):
        return strActionToDigits(self.dictionary_args[a])

    def get_tp_avail_act(self, type):
        DISABLE = 0
        ENABLE = 1
        n_act = len(self.dictionary_args)
        ret = np.zeros(n_act) + ENABLE
        return ret

    def confirm_parameters_are_correct(self, team, agent_num, opp_agent_num):
        pass

class ActionConvertLegacy():
    def __init__(self, SELF_TEAM_ASSUME, OPP_TEAM_ASSUME, OPP_NUM_ASSUME) -> None:
        self.SELF_TEAM_ASSUME = SELF_TEAM_ASSUME
        self.OPP_TEAM_ASSUME = OPP_TEAM_ASSUME
        self.OPP_NUM_ASSUME = OPP_NUM_ASSUME
        # (main_cmd, sub_cmd, x=None, y=None, z=None, UID=None, T=None, T_index=None)
        self.dictionary_args = [
            ('N/A',         'N/A',              None, None, None, None, None, None),   # 0
            ('Idle',        'DynamicGuard',     None, None, None, None, None, None),   # 1
            ('Idle',        'StaticAlert',      None, None, None, None, None, None),   # 2
            ('Idle',        'AsFarAsPossible',              None, None, None, None, None, None),   # 4
            ('Idle',        'StayWhenTargetInRange',        None, None, None, None, None, None),   # 5
            ('SpecificMoving',      'Dir+X',    None, None, None, None, None, None),   # 7
            ('SpecificMoving',      'Dir+Y',    None, None, None, None, None, None),   # 8
            ('SpecificMoving',      'Dir-X',    None, None, None, None, None, None),   # 9
            ('SpecificMoving',      'Dir-Y',    None, None, None, None, None, None),   # 10
        ] 
        for i in range(self.OPP_NUM_ASSUME):
            self.dictionary_args.append( ('SpecificAttacking',   'N/A',      None, None, None, None, OPP_TEAM_ASSUME, i) )
    
    

    def convert_act_arr(self, type, a):
        if type == 'RLA_UAV_Support':
            args = self.dictionary_args[a]
            # override wrong actions
            if args[0] == 'SpecificAttacking':
                return encode_action_as_digits('N/A',         'N/A',              None, None, None, None, None, None)
            # override incorrect actions
            if args[0] == 'Idle':
                return encode_action_as_digits('Idle',        'StaticAlert',      None, None, None, None, None, None)
            return encode_action_as_digits(*args)
        else:
            return encode_action_as_digits(*self.dictionary_args[a])

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
    
def count_list_type(x):
    type_cnt = {}
    for xx in x:
        if xx not in type_cnt: type_cnt[xx] = 0
        type_cnt[xx] += 1
    return len(type_cnt)

 

class ShellEnvWrapper(object):
    def __init__(self, n_agent, n_thread, space, mcv, rl_functional, alg_config, ScenarioConfig, team):
        self.n_agent = n_agent
        self.n_thread = n_thread
        self.team = team
        self.space = space
        self.mcv = mcv
        self.rl_functional = rl_functional
        if GlobalConfig.ScenarioConfig.EntityOriented:
            self.core_dim = GlobalConfig.ScenarioConfig.obs_vec_length
        else:
            self.core_dim = space['obs_space']['obs_shape']
        self.n_entity_placeholder = alg_config.n_entity_placeholder

        # whether to use avail_act to block forbiden actions
        self.AvailActProvided = False
        if hasattr(ScenarioConfig, 'AvailActProvided'):
            self.AvailActProvided = ScenarioConfig.AvailActProvided 

        if GlobalConfig.ScenarioConfig.SubTaskSelection in ['UhmapLargeScale', 'UhmapHuge', 'UhmapBreakingBad']:
            ActionToDiscreteConverter = ActionConvertLegacy
        else:
            ActionToDiscreteConverter = ActionConvertPredatorPrey

        self.action_converter = ActionToDiscreteConverter(
                SELF_TEAM_ASSUME=team, 
                OPP_TEAM_ASSUME=(1-team), 
                OPP_NUM_ASSUME=GlobalConfig.ScenarioConfig.N_AGENT_EACH_TEAM[1-team]
        )
        
        # check parameters
        self.patience = 2000

    def interact_with_env(self, StateRecall):
        if not hasattr(self, 'agent_type'):
            self.agent_uid = GlobalConfig.ScenarioConfig.AGENT_ID_EACH_TEAM[self.team]
            self.agent_type = [agent_meta['type'] 
                               for agent_meta in StateRecall['Latest-Team-Info'][0]['dataArr']
                               if agent_meta['uId'] in self.agent_uid]
            if ShellEnvConfig.add_avail_act:
                self.avail_act = np.stack(tuple(self.action_converter.get_tp_avail_act(tp) for tp in self.agent_type))
                self.avail_act = repeat_at(self.avail_act, insert_dim=0, n_times=self.n_thread)

        act = np.zeros(shape=(self.n_thread, self.n_agent), dtype=np.int) - 1 # 初始化全部为 -1
        
        # read internal coop graph info
        obs = StateRecall['Latest-Obs']
        obs = my_view(obs,[0, 0, -1, self.core_dim])
        obs[(obs==0).all(-1)] = np.nan

        n_entity_raw = obs.shape[-2]
        AlgorithmConfig.entity_distinct = [list(range(1)), list(range(1,n_entity_raw)), list(range(n_entity_raw,2*n_entity_raw))]

        P  =  StateRecall['ENV-PAUSE']
        R  = ~P
        RST = StateRecall['Env-Suffered-Reset']
        
        # when needed, train!
        if not StateRecall['Test-Flag']: self.rl_functional.train()
        
        if RST.all(): 
            # just experienced full reset on all episode, this is the first step of all env threads
            # randomly pick threads 
            eprsn_yita = self.rl_functional.stage_planner.yita if AlgorithmConfig.policy_resonance else 0
            EpRsn = np.random.rand(self.n_thread) < eprsn_yita
            StateRecall['_EpRsn_'] = EpRsn

        # prepare observation for the real RL algorithm
        obs_feed = obs[R]
        I_StateRecall = {
            'obs':obs_feed, 
            'avail_act':self.avail_act[R],
            'Test-Flag':StateRecall['Test-Flag'], 
            '_EpRsn_':StateRecall['_EpRsn_'][R],
            'threads_active_flag':R, 
            'Latest-Team-Info':StateRecall['Latest-Team-Info'][R],
        }
        # load available act to limit action space if possible
        if self.AvailActProvided:
            avail_act = np.array([info['avail-act'] for info in np.array(StateRecall['Latest-Team-Info'][R], dtype=object)])
            I_StateRecall.update({'avail_act':avail_act})

        # the real RL algorithm ! !
        act_active, internal_recall = self.rl_functional.interact_with_env_genuine(I_StateRecall)

        # get decision results
        act[R] = act_active
        
        # confirm actions are valid (satisfy 'avail-act')
        if ShellEnvConfig.add_avail_act and self.patience>0:
            self.patience -= 1
            assert (gather_righthand(self.avail_act, repeat_at(act, -1, 1), check=False)[R]==1).all()
            
        # translate action into ue4 tuple action
        act_converted = np.array([[ self.action_converter.convert_act_arr(self.agent_type[agentid], act) for agentid, act in enumerate(th) ] for th in act])
        
        # swap thread(batch) axis and agent axis
        actions_list = np.swapaxes(act_converted, 0, 1)

        # register callback hook
        if not StateRecall['Test-Flag']:
            StateRecall['_hook_'] = internal_recall['_hook_']
            assert StateRecall['_hook_'] is not None
        return actions_list, StateRecall 
