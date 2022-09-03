import numpy as np
from config import GlobalConfig
from UTIL.colorful import *
from UTIL.tensor_ops import my_view, __hash__, repeat_at, gather_righthand
from MISSION.uhmap.actset_lookup import encode_action_as_digits
from .pymarl2_compat import AlgorithmConfig

class ShellEnv(object):
    def __init__(self, rl_foundation, n_agent, n_thread, space, mcv, team):
        self.n_agent = n_agent
        self.n_thread = n_thread
        self.team = team
        self.rl_foundation = rl_foundation
        self.n_entity = GlobalConfig.ScenarioConfig.obs_n_entity
        self.obssize = GlobalConfig.ScenarioConfig.obs_vec_length

        if AlgorithmConfig.state_compat == 'pad':
            self.state_size = 4
        elif AlgorithmConfig.state_compat == 'obs_cat':
            self.state_size = self.n_entity * self.obssize * self.n_agent
        elif AlgorithmConfig.state_compat == 'obs_mean':
            self.state_size = self.n_entity * self.obssize
        else:
            assert False, 'compat method error'

        self.rl_foundation.space = {
            'act_space':{
                'n_actions': len(ActionConvertLegacy.dictionary_args),
            },
            'obs_space':{
                'obs_shape': self.n_entity * self.obssize,
                'state_shape': self.state_size,
            }
        }
        self.rl_foundation.interact_with_env_real = self.rl_foundation.interact_with_env
        self.rl_foundation.interact_with_env = self.interact_with_env
        self.patience = 2000
        
        opp_type_list = [a['type'] for a in GlobalConfig.ScenarioConfig.SubTaskConfig.agent_list if a['team']!=self.team]
        ActionConvertLegacy.confirm_parameters_are_correct(team, self.n_agent, len(opp_type_list))
        
    def interact_with_env(self, StateRecall):
        if not hasattr(self, 'agent_type'):
            self.agent_uid = GlobalConfig.ScenarioConfig.AGENT_ID_EACH_TEAM[self.team]
            self.agent_type = [agent_meta['type'] for agent_meta in StateRecall['Latest-Team-Info'][0]['dataArr']
                if agent_meta['uId'] in self.agent_uid]
            self.avail_act = np.stack(tuple(ActionConvertLegacy.get_tp_avail_act(tp) for tp in self.agent_type))
            self.avail_act = repeat_at(self.avail_act, insert_dim=0, n_times=self.n_thread)
        StateRecall['Latest-Obs'] = np.nan_to_num(my_view(StateRecall['Latest-Obs'], [0,0,-1]), 0)  
        StateRecall['Terminal-Obs-Echo'] = [np.nan_to_num(my_view(t, [0,-1]), 0)   if t is not None else None for t in StateRecall['Terminal-Obs-Echo']]
        for i, d in enumerate(StateRecall['Latest-Team-Info']):
            if AlgorithmConfig.state_compat == 'pad':
                d['state']      = np.zeros(self.state_size)
                d['state-echo'] = np.zeros(self.state_size)
            elif AlgorithmConfig.state_compat == 'obs_cat':
                d['state']      = StateRecall['Latest-Obs'][i].flatten()
                d['state-echo'] = np.zeros_like(d['state'])
                if StateRecall['Terminal-Obs-Echo'][i] is not None: 
                    d['state-echo'] = StateRecall['Terminal-Obs-Echo'][i].flatten()
            elif AlgorithmConfig.state_compat == 'obs_mean':
                d['state']      = StateRecall['Latest-Obs'][i].mean(0)
                d['state-echo'] = np.zeros_like(d['state'])
                if StateRecall['Terminal-Obs-Echo'][i] is not None: 
                    d['state-echo'] = StateRecall['Terminal-Obs-Echo'][i].mean(0)
            else:
                assert False, 'compat method error'
            d['avail-act'] = self.avail_act[i]
            
        # for t in StateRecall['Terminal-Obs-Echo']:
        #     if t is not None:
        #         print('???')
            
        ret_action_list, team_intel = self.rl_foundation.interact_with_env_real(StateRecall)
        ret_action_list = np.swapaxes(ret_action_list, 0, 1)
        # mask = np.isnan(ret_action_list)
        # ret_action_list = np.nan_to_num(ret_action_list, 0)
        
        R  = ~StateRecall['ENV-PAUSE']
        
        # if self.patience>0:
        #     self.patience -= 1
        #     assert (gather_righthand(self.avail_act, repeat_at(ret_action_list, -1, 1), check=False)[R]==1).all()
            
        act_converted = np.array([
            [ 
                ActionConvertLegacy.convert_act_arr(self.agent_type[agentid], int(act)) 
                    if not np.isnan(act) else 
                    ActionConvertLegacy.convert_act_arr(self.agent_type[agentid], 0) + np.nan
                for agentid, act in enumerate(th) ] 
            for th in ret_action_list])
        
        # act_converted[mask] = np.nan
        act_converted = np.swapaxes(act_converted, 0, 1)
        return act_converted, team_intel
    
    
    
    
    
    
    
class ActionConvertLegacy():
    SELF_TEAM_ASSUME = 0
    OPP_TEAM_ASSUME = 1
    OPP_NUM_ASSUME = 10
    # (main_cmd, sub_cmd, x=None, y=None, z=None, UID=None, T=None, T_index=None)
    dictionary_args = [
        ('N/A',         'N/A',              None, None, None, None, None, None),   # 0
        ('Idle',        'DynamicGuard',     None, None, None, None, None, None),   # 1
        ('Idle',        'StaticAlert',      None, None, None, None, None, None),   # 2
        ('Idle',        'AsFarAsPossible',              None, None, None, None, None, None),   # 4
        ('Idle',        'StayWhenTargetInRange',        None, None, None, None, None, None),   # 5
        ('SpecificMoving',      'Dir+X',    None, None, None, None, None, None),   # 7
        ('SpecificMoving',      'Dir+Y',    None, None, None, None, None, None),   # 8
        ('SpecificMoving',      'Dir-X',    None, None, None, None, None, None),   # 9
        ('SpecificMoving',      'Dir-Y',    None, None, None, None, None, None),   # 10
        ('SpecificAttacking',   'N/A',      None, None, None, None, OPP_TEAM_ASSUME,    0),      # 11
        ('SpecificAttacking',   'N/A',      None, None, None, None, OPP_TEAM_ASSUME,    1),      # 12
        ('SpecificAttacking',   'N/A',      None, None, None, None, OPP_TEAM_ASSUME,    2),      # 13
        ('SpecificAttacking',   'N/A',      None, None, None, None, OPP_TEAM_ASSUME,    3),      # 14
        ('SpecificAttacking',   'N/A',      None, None, None, None, OPP_TEAM_ASSUME,    4),      # 15
        ('SpecificAttacking',   'N/A',      None, None, None, None, OPP_TEAM_ASSUME,    5),      # 16
        ('SpecificAttacking',   'N/A',      None, None, None, None, OPP_TEAM_ASSUME,    6),      # 17        
        ('SpecificAttacking',   'N/A',      None, None, None, None, OPP_TEAM_ASSUME,    7),      # 
        ('SpecificAttacking',   'N/A',      None, None, None, None, OPP_TEAM_ASSUME,    8),      # 
        ('SpecificAttacking',   'N/A',      None, None, None, None, OPP_TEAM_ASSUME,    9),      # 
    ]


    @staticmethod
    def convert_act_arr(type, a):
        if type == 'RLA_UAV_Support':
            args = ActionConvertLegacy.dictionary_args[a]
            # override wrong actions
            if args[0] == 'SpecificAttacking':
                return encode_action_as_digits('N/A',         'N/A',              None, None, None, None, None, None)
            # override incorrect actions
            if args[0] == 'Idle':
                return encode_action_as_digits('Idle',        'StaticAlert',      None, None, None, None, None, None)
            return encode_action_as_digits(*args)
        else:
            return encode_action_as_digits(*ActionConvertLegacy.dictionary_args[a])

    @staticmethod
    def get_tp_avail_act(type):
        DISABLE = 0
        ENABLE = 1
        n_act = len(ActionConvertLegacy.dictionary_args)
        ret = np.zeros(n_act) + ENABLE
        for i in range(n_act):
            args = ActionConvertLegacy.dictionary_args[i]
            
            # for all kind of agents
            if args[0] == 'PatrolMoving':       ret[i] = DISABLE
            
            if type == 'RLA_UAV_Support':
                if args[0] == 'PatrolMoving':       ret[i] = DISABLE
                if args[0] == 'SpecificAttacking':  ret[i] = DISABLE
                if args[0] == 'Idle':               ret[i] = DISABLE
                if args[1] == 'StaticAlert':        ret[i] = ENABLE
        return ret
    
    @staticmethod
    def confirm_parameters_are_correct(team, agent_num, opp_agent_num):
        assert team == ActionConvertLegacy.SELF_TEAM_ASSUME
        assert ActionConvertLegacy.SELF_TEAM_ASSUME + ActionConvertLegacy.OPP_TEAM_ASSUME == 1
        assert ActionConvertLegacy.SELF_TEAM_ASSUME + ActionConvertLegacy.OPP_TEAM_ASSUME == 1
        assert opp_agent_num == ActionConvertLegacy.OPP_NUM_ASSUME
