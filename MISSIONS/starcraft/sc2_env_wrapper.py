import os,sys
import numpy as np
class ChainVar(object):
    def __init__(self, chain_func, chained_with):
        self.chain_func = chain_func
        self.chained_with = chained_with
'''
    Name            Agents  Enemies Limit  
    3m              3       3       60     
    8m              8       8       120    
    25m             25      25      150    
    5m_vs_6m        5       6       70     
    8m_vs_9m        8       9       120    
    10m_vs_11m      10      11      150    
    27m_vs_30m      27      30      180    
    MMM             10      10      150    
    MMM2            10      12      180    
    2s3z            5       5       120    
    3s5z            8       8       150    
    3s5z_vs_3s6z    8       9       170    
    3s_vs_3z        3       3       150    
    3s_vs_4z        3       4       200    
    3s_vs_5z        3       5       250    
    1c3s5z          9       9       180    
    2m_vs_1z        2       1       150    
    corridor        6       24      400    
    6h_vs_8z        6       8       150    
    2s_vs_1sc       2       1       300    
    so_many_baneling 7       32      100    
    bane_vs_bane    24      24      200    
    2c_vs_64zg      2       64      400    
'''

sc2map_info = {
    "3m":               {"n_agents":3    ,  "n_hostiles":   3    , "ep_limit":  60   },
    "8m":               {"n_agents":8    ,  "n_hostiles":   8    , "ep_limit":  120  },
    "25m":              {"n_agents":25   ,  "n_hostiles":   25   , "ep_limit":  150  },
    "5m_vs_6m":         {"n_agents":5    ,  "n_hostiles":   6    , "ep_limit":  70   },
    "8m_vs_9m":         {"n_agents":8    ,  "n_hostiles":   9    , "ep_limit":  120  },
    "10m_vs_11m":       {"n_agents":10   ,  "n_hostiles":   11   , "ep_limit":  150  },
    "27m_vs_30m":       {"n_agents":27   ,  "n_hostiles":   30   , "ep_limit":  180  },
    "MMM":              {"n_agents":10   ,  "n_hostiles":   10   , "ep_limit":  150  },
    "MMM2":             {"n_agents":10   ,  "n_hostiles":   12   , "ep_limit":  180  },
    "2s3z":             {"n_agents":5    ,  "n_hostiles":   5    , "ep_limit":  120  },
    "3s5z":             {"n_agents":8    ,  "n_hostiles":   8    , "ep_limit":  150  },
    "3s5z_vs_3s6z":     {"n_agents":8    ,  "n_hostiles":   9    , "ep_limit":  170  },
    "3s_vs_3z":         {"n_agents":3    ,  "n_hostiles":   3    , "ep_limit":  150  },
    "3s_vs_4z":         {"n_agents":3    ,  "n_hostiles":   4    , "ep_limit":  200  },
    "3s_vs_5z":         {"n_agents":3    ,  "n_hostiles":   5    , "ep_limit":  250  },
    "1c3s5z":           {"n_agents":9    ,  "n_hostiles":   9    , "ep_limit":  180  },
    "2m_vs_1z":         {"n_agents":2    ,  "n_hostiles":   1    , "ep_limit":  150  },
    "corridor":         {"n_agents":6    ,  "n_hostiles":   24   , "ep_limit":  400  },
    "6h_vs_8z":         {"n_agents":6    ,  "n_hostiles":   8    , "ep_limit":  150  },
    "2s_vs_1sc":        {"n_agents":2    ,  "n_hostiles":   1    , "ep_limit":  300  },
    "so_many_baneling": {"n_agents":7    ,  "n_hostiles":   32   , "ep_limit":  100  },
    "bane_vs_bane":     {"n_agents":24   ,  "n_hostiles":   24   , "ep_limit":  200  },
    "2c_vs_64zg":       {"n_agents":2    ,  "n_hostiles":   64   , "ep_limit":  400  },
}

from smac.env.starcraft2.maps import smac_maps
map_param_registry = smac_maps.get_smac_map_registry()

class ScenarioConfig(object): # ADD_TO_CONF_SYSTEM 加入参数搜索路径 do not remove this comment !!!
    map_ = 'corridor'
    step_mul = 8
    difficulty = '7'
    game_version = 'latest'
    replay_dir = ''

    # automatic select episode length limit
    episode_limit = sc2map_info[map_]["ep_limit"]
    episode_limit_cv = ChainVar(
        lambda map_: sc2map_info[map_]["ep_limit"], 
        chained_with=['map_']
    )

    num_entity = "not avail ?"
    uid_dictionary = "not avail ?"
    N_TEAM = 1  
    SINGLE_TEAM_N_AGENT = 6
    

    N_AGENT_EACH_TEAM = [SINGLE_TEAM_N_AGENT,] 
    N_AGENT_EACH_TEAM_cv = ChainVar(
        lambda SINGLE_TEAM_N_AGENT:[SINGLE_TEAM_N_AGENT,], 
        chained_with=['SINGLE_TEAM_N_AGENT']
    )

    AGENT_ID_EACH_TEAM = [range(0,SINGLE_TEAM_N_AGENT),]
    AGENT_ID_EACH_TEAM_cv = ChainVar(
        lambda SINGLE_TEAM_N_AGENT:[range(0,SINGLE_TEAM_N_AGENT),], 
        chained_with=['SINGLE_TEAM_N_AGENT']
    )

    TEAM_NAMES = [  
                    'ALGORITHM.Starcraft.star_foundation->StarFoundation',
                ] 
    ActAsUnity = False

    reward_vec = False
    RewardAsUnity = True if not reward_vec else False   
    RewardAsUnity_cv = ChainVar(
        lambda reward_vec: (not reward_vec), 
        chained_with=['reward_vec']
    )

    state_provided = True
    avail_act_provided = True


    max_steps_episode = sc2map_info[map_]["ep_limit"]
    max_steps_episode_cv = ChainVar(
        lambda map_: sc2map_info[map_]["ep_limit"], 
        chained_with=['map_']
    )

    n_actions = 6 + sc2map_info[map_]["n_hostiles"]
    n_actions_cv = ChainVar(
        lambda map_:6 + sc2map_info[map_]["n_hostiles"], 
        chained_with=['map_']
    )
    obs_vec_length = 6
    return_mat = False
    block_invalid_action = True # sc2 中，需要始终屏蔽掉不可用的动作
    reward_sparse=False
    render = False
    # sc2map_info[map_]["n_agents"]
    # n_action_cv = ChainVar(
    #     lambda map_: sc2map_info[map_]["n_agents"], 
    #     chained_with=['map_']
    # )

def make_sc2_env(env_id, rank):
    return Env_Compat_Wrapper(rank)



class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout




# 一层套一层。。。这层是为了参数对齐
class Env_Compat_Wrapper():
    def __init__(self, rank):
        from smac.env import StarCraft2Env
        self.env = StarCraft2Env(map_name=ScenarioConfig.map_,
                            step_mul=ScenarioConfig.step_mul,
                            difficulty=ScenarioConfig.difficulty,
                            game_version=ScenarioConfig.game_version,
                            reward_sparse=ScenarioConfig.reward_sparse,
                            return_mat=ScenarioConfig.return_mat,
                            reward_vec=ScenarioConfig.reward_vec,
                            render=True if (rank==0 and ScenarioConfig.render) else False,
                            replay_dir=ScenarioConfig.replay_dir)

        env_info = self.env.get_env_info()
        self.observation_space = {'state_shape': env_info["state_shape"], 
                                  'obs_shape':   env_info["obs_shape"]}

        self.action_space =  {'n_actions': env_info["n_actions"],
                              'n_agents':  env_info["n_agents"]}

        assert env_info["n_agents"] == ScenarioConfig.N_AGENT_EACH_TEAM[0], ('Changed a map? Reconfig ScenarioConfig Above!! n_agents:', env_info["n_agents"])
        assert env_info["episode_limit"] == ScenarioConfig.episode_limit, ('Changed a map? Reconfig ScenarioConfig Above!! episode_limit:',env_info["episode_limit"])
        assert env_info["n_actions"] == ScenarioConfig.n_actions, ('Changed a map? Reconfig ScenarioConfig Above!! n_actions:', env_info["n_actions"])

        self.id = rank
        pass

    def step(self, act):
        # with HiddenPrints():
        reward, terminated, info = self.env.step(act)
        if ScenarioConfig.RewardAsUnity: reward = [reward]
        done = terminated
        ob = np.array(self.env.get_obs())
        info['state'] = self.env.get_state()
        info['avail-act'] = self.env.get_avail_actions()
        return (ob, reward, done, info)

    def reset(self):
        # with HiddenPrints():
        self.env.reset()
        ob = np.array(self.env.get_obs())
        info = {}
        info['state'] = self.env.get_state()
        info['avail-act'] = self.env.get_avail_actions()
        return ob, info

    def render(self):
        return

# self.env.close()
# self.env.reset()
# state = self.env.get_state()
# obs = self.env.get_obs()
# reward, terminated, info = self.env.step(actions)
# win_tag = True if terminated and 'battle_won' in info and info['battle_won'] else False