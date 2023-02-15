import numpy as np
from config import GlobalConfig
from UTIL.colorful import *
from UTIL.tensor_ops import my_view, __hash__, repeat_at, gather_righthand
from MISSION.uhmap.actset_lookup import encode_action_as_digits
from .foundation import AlgorithmConfig
from .cython_func import roll_hisory
from .hete_assignment import select_nets_for_shellenv

class ShellEnvConfig:
    add_avail_act = False
    
    
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
        self.action_converter = ActionConvertLegacy(
                SELF_TEAM_ASSUME=team, 
                OPP_TEAM_ASSUME=(1-team), 
                OPP_NUM_ASSUME=GlobalConfig.ScenarioConfig.N_AGENT_EACH_TEAM[1-team]
        )
        # heterogeneous agent types
        agent_type_list = [a['type'] for a in GlobalConfig.ScenarioConfig.SubTaskConfig.agent_list]
        opp_type_list = [a['type'] for a in GlobalConfig.ScenarioConfig.SubTaskConfig.agent_list if a['team']!=self.team]
        self_type_list = [a['type'] for a in GlobalConfig.ScenarioConfig.SubTaskConfig.agent_list if a['team']==self.team]
        def str_array_to_num(str_arr):
            out_arr = []
            buffer = {}
            for str in str_arr:
                if str not in buffer:
                    buffer[str] = len(buffer)
                out_arr.append(buffer[str])
            return out_arr  
        
        self.HeteAgentType = str_array_to_num(agent_type_list)
        self.hete_type = np.array(self.HeteAgentType)[GlobalConfig.ScenarioConfig.AGENT_ID_EACH_TEAM[team]]
        self.n_hete_types = count_list_type(self.hete_type)
        
        # check parameters
        assert self.n_agent == len(self_type_list)
        self.action_converter.confirm_parameters_are_correct(team, self.n_agent, len(opp_type_list))
        self.patience = 2000
        self.epsiode_cnt = 0

    def cold_start_warmup(self, StateRecall):
        self.agent_uid = GlobalConfig.ScenarioConfig.AGENT_ID_EACH_TEAM[self.team]
        self.agent_type = [agent_meta['type'] 
            for agent_meta in StateRecall['Latest-Team-Info'][0]['dataArr']
            if agent_meta['uId'] in self.agent_uid]
        if ShellEnvConfig.add_avail_act:
            self.avail_act = np.stack(tuple(self.action_converter.get_tp_avail_act(tp) for tp in self.agent_type))
            self.avail_act = repeat_at(self.avail_act, insert_dim=0, n_times=self.n_thread)


    def interact_with_env(self, StateRecall):
        # warm up at first execution
        if not hasattr(self, 'agent_type'):
            self.cold_start_warmup(StateRecall)
            
        # action init to: -1
        act = np.zeros(shape=(self.n_thread, self.n_agent), dtype=np.int) - 1
        
        # read and reshape observation
        obs = StateRecall['Latest-Obs']
        obs = my_view(obs,[0, 0, -1, self.core_dim])
        # mask out invalid observation with NaN
        obs[(obs==0).all(-1)] = np.nan

        # stopped env mask
        P  =  StateRecall['ENV-PAUSE']
        # running env mask 
        R  = ~P
        # reset env mask
        RST = StateRecall['Env-Suffered-Reset']
        
        # when needed, train!
        if not StateRecall['Test-Flag']: self.rl_functional.train()
        
        # if true: just experienced full reset on all episode, this is the first step of all env threads
        if RST.all(): 
            if AlgorithmConfig.allow_fast_test and GlobalConfig.test_only and (self.epsiode_cnt > GlobalConfig.report_reward_interval):
                import sys
                sys.exit(0)
            self.epsiode_cnt += self.n_thread
            # policy resonance
            eprsn_yita = self.rl_functional.stage_planner.yita if AlgorithmConfig.policy_resonance else 0
            EpRsn = np.random.rand(self.n_thread) < eprsn_yita
            StateRecall['_EpRsn_'] = EpRsn
            # heterogeneous agent identification
            StateRecall['_hete_type_'] = repeat_at(self.hete_type, 0, self.n_thread)
            # select static/frontier actor network
            StateRecall['_hete_pick_'], StateRecall['_gp_pick_'] = select_nets_for_shellenv(
                                        n_types=self.n_hete_types, 
                                        policy=self.rl_functional.policy,
                                        hete_type_list=self.hete_type,
                                        n_thread = self.n_thread,
                                        n_gp=AlgorithmConfig.hete_n_net_placeholder,
                                        testing=StateRecall['Test-Flag']
                                    )
            print([(t['win_rate'], t['ckpg_cnt']) for t in self.rl_functional.policy.ckpg_info])

        # prepare observation for the real RL algorithm
        I_StateRecall = {
            'obs':obs[R], 
            'avail_act':self.avail_act[R],
            'Test-Flag':StateRecall['Test-Flag'], 
            '_EpRsn_':StateRecall['_EpRsn_'][R],
            '_hete_pick_':StateRecall['_hete_pick_'][R], 
            '_hete_type_':StateRecall['_hete_type_'][R],
            '_gp_pick_':StateRecall['_gp_pick_'][R], 
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
        else:
            if AlgorithmConfig.policy_matrix_testing:
                StateRecall['_hook_'] = internal_recall['_hook_']
                assert StateRecall['_hook_'] is not None
                

        # all done
        return actions_list, StateRecall 
