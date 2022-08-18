import numpy as np
from config import GlobalConfig
from UTIL.colorful import *
from UTIL.tensor_ops import my_view, __hash__, repeat_at, gather_righthand
from MISSION.uhmap.actset_lookup import encode_action_as_digits
from .foundation import AlgorithmConfig
from .cython_func import roll_hisory


class ShellEnvConfig:
    add_avail_act = False
    
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
    
    
def count_list_type(x):
    type_cnt = {}
    for xx in x:
        if xx not in type_cnt: type_cnt[xx] = 0
        type_cnt[xx] += 1
    return len(type_cnt)

 

class ShellEnvWrapper(object):
    def __init__(self, n_agent, n_thread, space, mcv, RL_functional, alg_config, ScenarioConfig, team):
        self.n_agent = n_agent
        self.n_thread = n_thread
        self.team = team
        self.space = space
        self.mcv = mcv
        self.RL_functional = RL_functional
        if GlobalConfig.ScenarioConfig.EntityOriented:
            self.core_dim = GlobalConfig.ScenarioConfig.obs_vec_length
        else:
            self.core_dim = space['obs_space']['obs_shape']
        self.n_entity_placeholder = alg_config.n_entity_placeholder
        assert self.n_entity_placeholder >= 4

        # whether to use avail_act to block forbiden actions
        self.AvailActProvided = False
        if hasattr(ScenarioConfig, 'AvailActProvided'):
            self.AvailActProvided = ScenarioConfig.AvailActProvided 

        # heterogeneous agent types
        assert GlobalConfig.ScenarioConfig.HeteAgents
        self.HeteAgentType = np.array(GlobalConfig.ScenarioConfig.HeteAgentType)
        self.hete_type = np.array(self.HeteAgentType)[GlobalConfig.ScenarioConfig.AGENT_ID_EACH_TEAM[team]]
        self.n_hete_types = count_list_type(self.hete_type)
        
        # check parameters
        assert self.n_agent == GlobalConfig.ScenarioConfig.n_team1agent
        ActionConvertLegacy.confirm_parameters_are_correct(team, self.n_agent, GlobalConfig.ScenarioConfig.n_team2agent)
        self.patience = 2000
        
    @staticmethod
    def get_binary_array(n, n_bits, dtype=np.float32):
        arr = np.zeros(n_bits, dtype=dtype)
        pointer = 0
        while True:
            arr[pointer] = int(n%2==1)
            n = n >> 1
            pointer += 1
            if n == 0: break
        return arr

    def interact_with_env(self, StateRecall):
        if not hasattr(self, 'agent_type'):
            self.agent_uid = GlobalConfig.ScenarioConfig.AGENT_ID_EACH_TEAM[self.team]
            self.agent_type = [agent_meta['type'] 
                for agent_meta in StateRecall['Latest-Team-Info'][0]['dataArr']
                if agent_meta['uId'] in self.agent_uid]
            if ShellEnvConfig.add_avail_act:
                self.avail_act = np.stack(tuple(ActionConvertLegacy.get_tp_avail_act(tp) for tp in self.agent_type))
                self.avail_act = repeat_at(self.avail_act, insert_dim=0, n_times=self.n_thread)

        act = np.zeros(shape=(self.n_thread, self.n_agent), dtype=np.int) - 1 # 初始化全部为 -1
        # read internal coop graph info
        obs = StateRecall['Latest-Obs']
        if not GlobalConfig.ScenarioConfig.EntityOriented:    
            # 如果环境观测非EntityOriented，可以额外创生一个维度，具体细节需要斟酌
            obs = repeat_at(obs, insert_dim=-2, n_times=self.n_entity_placeholder//2, copy_mem=True)
            obs[:,:,2:] = np.nan    # 0 is self; 1 is repeated self; 2,3,... is NaN
            
        n_entity_raw = obs.shape[-2]
        AlgorithmConfig.entity_distinct = [list(range(1)), list(range(1,n_entity_raw)), list(range(n_entity_raw,2*n_entity_raw))]
        
        P  =  StateRecall['ENV-PAUSE']
        R  = ~StateRecall['ENV-PAUSE']
        RST = StateRecall['Env-Suffered-Reset']
        
        if RST.all(): # just experienced full reset on all episode, this is the first step of all env threads
            # randomly pick threads
            EpRsn = np.random.rand(self.n_thread) < AlgorithmConfig.yita
            StateRecall['_EpRsn_'] = EpRsn
                
            n_types = self.n_hete_types
            selected_type = np.random.randint(low=0, high=n_types, size=())
            selected_agent_bool = (self.hete_type==selected_type)
            selected_agent_bool = repeat_at(selected_agent_bool, 0, self.n_thread)
            
            group_sel_arr = np.random.randint(low=0, high=AlgorithmConfig.n_policy_groups, size=(self.n_thread, self.n_agent))
            hete_type_arr = repeat_at(self.hete_type, 0, self.n_thread)
            
            n_tp = n_types
            n_gp = AlgorithmConfig.n_policy_groups
            get_placeholder = lambda type, group: group*n_tp + type
            get_type_group = lambda ph: (ph%n_tp, ph//n_tp)
            
            StateRecall['_Type_'] = get_placeholder(type=hete_type_arr, group=group_sel_arr)
            StateRecall['_Type_'][selected_agent_bool] = (selected_type) # 

        his_pool_obs = StateRecall['_history_pool_obs_'] if '_history_pool_obs_' in StateRecall \
            else my_view(np.zeros_like(obs),[0, 0, -1, self.core_dim])
        his_pool_obs[RST] = 0

        obs_feed = obs[R]
        his_pool_obs_feed = his_pool_obs[R]
        obs_feed_in, his_pool_next = self.solve_duplicate(obs_feed.copy(), his_pool_obs_feed.copy())
        his_pool_obs[R] = his_pool_next
        his_pool_obs[P] = 0

        I_StateRecall = {
            'obs':obs_feed_in, 
            'avail_act':self.avail_act[R],
            'Test-Flag':StateRecall['Test-Flag'], 
            '_Type_':StateRecall['_Type_'][R], 
            'threads_active_flag':R, 
            'Latest-Team-Info':StateRecall['Latest-Team-Info'][R],
        }
        if self.AvailActProvided:
            avail_act = np.array([info['avail-act'] for info in np.array(StateRecall['Latest-Team-Info'][R], dtype=object)])
            I_StateRecall.update({'avail_act':avail_act})

        if AlgorithmConfig.PR_ACTIVATE:
            self.RL_functional.ccategorical.register_rsn(rsn_flag=StateRecall['_EpRsn_'][R])

        act_active, internal_recall = self.RL_functional.interact_with_env_genuine(I_StateRecall)

        act[R] = act_active
        
        if ShellEnvConfig.add_avail_act and self.patience>0:
            self.patience -= 1
            assert (gather_righthand(self.avail_act, repeat_at(act, -1, 1), check=False)[R]==1).all()
            

        # translate action into ue4 tuple action
        act_converted = np.array([[ ActionConvertLegacy.convert_act_arr(self.agent_type[agentid], act) for agentid, act in enumerate(th) ] for th in act])
        # swap thread(batch) axis and agent axis
        actions_list = np.swapaxes(act_converted, 0, 1)


        StateRecall['_history_pool_obs_'] = his_pool_obs
        if not StateRecall['Test-Flag']:
            StateRecall['_hook_'] = internal_recall['_hook_']
            assert StateRecall['_hook_'] is not None
        return actions_list, StateRecall 

    def solve_duplicate(self, obs_feed_new, prev_his_pool):
        #  input might be (n_thread, n_agent, n_entity, basic_dim), or (n_thread, n_agent, n_entity*basic_dim)
        # both can be converted to (n_thread, n_agent, n_entity, basic_dim)
        obs_feed_new = my_view(obs_feed_new,[0, 0, -1, self.core_dim])
        prev_obs_feed = my_view(prev_his_pool,[0, 0, -1, self.core_dim])

        # turn history into more entities
        obs_feed = np.concatenate((obs_feed_new, prev_obs_feed), axis=-2)

        # turning all zero padding to NaN, which is excluded in normalization
        obs_feed[(obs_feed==0).all(-1)] = np.nan
        obs_feed_new[(obs_feed_new==0).all(-1)] = np.nan
        valid_mask = ~np.isnan(obs_feed_new).any(-1)    #

        # set self as not valid to avoid buffering self obs! valid_mask
        valid_mask[:,:,0] = False

        # N valid: how many subjects (entities) needs to be buffered
        N_valid = valid_mask.sum(-1)

        # alloc mem for next_his_pool
        next_his_pool = np.zeros_like(prev_obs_feed) # twice size  ( threads,  agents,  subjects)

        # fill next_his_pool
        next_his_pool = roll_hisory(obs_feed_new, prev_obs_feed, valid_mask, N_valid, next_his_pool)

        # a very important assumption: if an agent observe itself as NaN *When and Only When* it is dead
        alive_mask = ~np.isnan(obs_feed_new[:,:,0]).any(-1) 
        if (~alive_mask).any():
            obs_feed[~alive_mask] = np.nan
        return obs_feed, next_his_pool

