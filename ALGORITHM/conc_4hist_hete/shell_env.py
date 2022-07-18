import numpy as np
from config import GlobalConfig
from UTIL.colorful import *
from UTIL.tensor_ops import my_view, __hash__, repeat_at
from MISSION.uhmap.actset_lookup import encode_action_as_digits
from .foundation import AlgorithmConfig
from .cython_func import roll_hisory

class ActionConvertLegacy():

    # (main_cmd, sub_cmd, x=None, y=None, z=None, UID=None, T=None, T_index=None)
    dictionary_args = [
        ('N/A',         'N/A',              None, None, None, None, None, None),   # 0
        ('Idle',        'DynamicGuard',     None, None, None, None, None, None),   # 1
        ('Idle',        'StaticAlert',      None, None, None, None, None, None),   # 2
        ('Idle',        'AggressivePersue', None, None, None, None, None, None),   # 3
        ('Idle',        'AsFarAsPossible',              None, None, None, None, None, None),   # 1
        ('Idle',        'StayWhenTargetInRange',        None, None, None, None, None, None),   # 2
        ('Idle',        'StayWhenTargetInHalfRange',    None, None, None, None, None, None),   # 3
        ('SpecificMoving',      'Dir+X',    None, None, None, None, None, None),   # 4
        ('SpecificMoving',      'Dir+Y',    None, None, None, None, None, None),   # 5
        ('SpecificMoving',      'Dir-X',    None, None, None, None, None, None),   # 6
        ('SpecificMoving',      'Dir-Y',    None, None, None, None, None, None),   # 7
        ('SpecificAttacking',   'N/A',      None, None, None, None, 1,    0),      # 8
        ('SpecificAttacking',   'N/A',      None, None, None, None, 1,    1),      # 9
        ('SpecificAttacking',   'N/A',      None, None, None, None, 1,    2),      # 10
        ('SpecificAttacking',   'N/A',      None, None, None, None, 1,    3),      # 11
        ('SpecificAttacking',   'N/A',      None, None, None, None, 1,    4),      # 12
        ('SpecificAttacking',   'N/A',      None, None, None, None, 0,    0),      # 13
        ('SpecificAttacking',   'N/A',      None, None, None, None, 0,    1),      # 14
        ('SpecificAttacking',   'N/A',      None, None, None, None, 0,    2),      # 15
        ('SpecificAttacking',   'N/A',      None, None, None, None, 0,    3),      # 16
        ('SpecificAttacking',   'N/A',      None, None, None, None, 0,    4),      # 17
        ('PatrolMoving',        'Dir+X',    None, None, None, None, None, None),   # 19
        ('PatrolMoving',        'Dir+Y',    None, None, None, None, None, None),   # 20
        ('PatrolMoving',        'Dir-X',    None, None, None, None, None, None),   # 21
        ('PatrolMoving',        'Dir-Y',    None, None, None, None, None, None),   # 22
    ]


    @staticmethod
    def convert_act_arr(type, a):
        if 'RLA_UAV' in type:
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

        act = np.zeros(shape=(self.n_thread, self.n_agent), dtype=np.int) - 1 # 初始化全部为 -1
        # read internal coop graph info
        obs = StateRecall['Latest-Obs']
        if not GlobalConfig.ScenarioConfig.EntityOriented:    
            # 如果环境观测非EntityOriented，可以额外创生一个维度，具体细节需要斟酌
            obs = repeat_at(obs, insert_dim=-2, n_times=self.n_entity_placeholder//2, copy_mem=True)
            obs[:,:,2:] = np.nan    # 0 is self; 1 is repeated self; 2,3,... is NaN
        P = StateRecall['ENV-PAUSE']
        RST = StateRecall['Env-Suffered-Reset']
        
        if RST.all(): # just experienced full reset on all episode, this is the first step of all env threads
            # yita = AlgorithmConfig.yita
            ## randomly pick threads
            # FixMax = np.random.rand(self.n_thread) < yita
            # StateRecall['_FixMax_'] = FixMax
            
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
            # print(FixMax)

        his_pool_obs = StateRecall['_history_pool_obs_'] if '_history_pool_obs_' in StateRecall \
            else my_view(np.zeros_like(obs),[0, 0, -1, self.core_dim])
        his_pool_obs[RST] = 0

        obs_feed = obs[~P]
        his_pool_obs_feed = his_pool_obs[~P]
        obs_feed_in, his_pool_next = self.solve_duplicate(obs_feed.copy(), his_pool_obs_feed.copy())
        his_pool_obs[~P] = his_pool_next
        his_pool_obs[P] = 0

        I_StateRecall = {'obs':obs_feed_in, 
            'Test-Flag':StateRecall['Test-Flag'], 
            # '_FixMax_':StateRecall['_FixMax_'][~P], 
            '_Type_':StateRecall['_Type_'][~P], 
            'threads_active_flag':~P, 
            'Latest-Team-Info':StateRecall['Latest-Team-Info'][~P],
            }
        if self.AvailActProvided:
            avail_act = np.array([info['avail-act'] for info in np.array(StateRecall['Latest-Team-Info'][~P], dtype=object)])
            I_StateRecall.update({'avail_act':avail_act})

        act_active, internal_recall = self.RL_functional.interact_with_env_genuine(I_StateRecall)

        act[~P] = act_active

        act_converted = np.array([
            [
                ActionConvertLegacy.convert_act_arr(self.agent_type[agentid], act)  for agentid, act in enumerate(th) 
            ] for th in act])
        actions_list = np.swapaxes(act_converted, 0, 1) # swap thread(batch) axis and agent axis


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

