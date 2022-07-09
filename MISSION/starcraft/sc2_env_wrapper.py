import os,sys
import numpy as np
from smac.env.starcraft2.maps import smac_maps
from UTIL.config_args import ChainVar
map_param_registry = smac_maps.get_smac_map_registry()

# sc2map_info = {
#     "3m":               {"n_agents":3    ,  "n_hostiles":   3    , "ep_limit":  60   },
#     "8m":               {"n_agents":8    ,  "n_hostiles":   8    , "ep_limit":  120  },
#     "25m":              {"n_agents":25   ,  "n_hostiles":   25   , "ep_limit":  150  },
#     "5m_vs_6m":         {"n_agents":5    ,  "n_hostiles":   6    , "ep_limit":  70   },
#     "8m_vs_9m":         {"n_agents":8    ,  "n_hostiles":   9    , "ep_limit":  120  },
#     "10m_vs_11m":       {"n_agents":10   ,  "n_hostiles":   11   , "ep_limit":  150  },
#     "27m_vs_30m":       {"n_agents":27   ,  "n_hostiles":   30   , "ep_limit":  180  },
#     "MMM":              {"n_agents":10   ,  "n_hostiles":   10   , "ep_limit":  150  },
#     "MMM2":             {"n_agents":10   ,  "n_hostiles":   12   , "ep_limit":  180  },
#     "2s3z":             {"n_agents":5    ,  "n_hostiles":   5    , "ep_limit":  120  },
#     "3s5z":             {"n_agents":8    ,  "n_hostiles":   8    , "ep_limit":  150  },
#     "3s5z_vs_3s6z":     {"n_agents":8    ,  "n_hostiles":   9    , "ep_limit":  170  },
#     "3s_vs_3z":         {"n_agents":3    ,  "n_hostiles":   3    , "ep_limit":  150  },
#     "3s_vs_4z":         {"n_agents":3    ,  "n_hostiles":   4    , "ep_limit":  200  },
#     "3s_vs_5z":         {"n_agents":3    ,  "n_hostiles":   5    , "ep_limit":  250  },
#     "1c3s5z":           {"n_agents":9    ,  "n_hostiles":   9    , "ep_limit":  180  },
#     "2m_vs_1z":         {"n_agents":2    ,  "n_hostiles":   1    , "ep_limit":  150  },
#     "corridor":         {"n_agents":6    ,  "n_hostiles":   24   , "ep_limit":  400  },
#     "6h_vs_8z":         {"n_agents":6    ,  "n_hostiles":   8    , "ep_limit":  150  },
#     "2s_vs_1sc":        {"n_agents":2    ,  "n_hostiles":   1    , "ep_limit":  300  },
#     "so_many_baneling": {"n_agents":7    ,  "n_hostiles":   32   , "ep_limit":  100  },
#     "bane_vs_bane":     {"n_agents":24   ,  "n_hostiles":   24   , "ep_limit":  200  },
#     "2c_vs_64zg":       {"n_agents":2    ,  "n_hostiles":   64   , "ep_limit":  400  },
# }




class ScenarioConfig(object):  
    '''
        ScenarioConfig: This config class will be 'injected' with new settings from JSONC.
        (E.g., override configs with ```python main.py --cfg example.jsonc```)
        (As the name indicated, ChainVars will change WITH vars it 'chained_with' during config injection)
        (please see UTIL.config_args to find out how this advanced trick works out.)
    '''
    sc_version = '2.4.10'
    map_ = 'corridor'
    step_mul = 8
    difficulty = '7'
    game_version = 'latest'
    replay_dir = ''

    # automatic select episode length limit
    episode_limit = map_param_registry[map_]['limit'] # sc2map_info[map_]["ep_limit"]
    episode_limit_cv = ChainVar(
        lambda map_: map_param_registry[map_]['limit'], 
        chained_with=['map_']
    )

    num_entity = "not avail ?"
    uid_dictionary = "not avail ?"
    N_TEAM = 1  
    SINGLE_TEAM_N_AGENT = map_param_registry[map_]['n_agents']
    SINGLE_TEAM_N_AGENT_cv = ChainVar(
        lambda map_:map_param_registry[map_]['n_agents'], 
        chained_with=['map_']
    )

    N_AGENT_EACH_TEAM = [map_param_registry[map_]['n_agents'],] 
    N_AGENT_EACH_TEAM_cv = ChainVar(
        lambda map_:[map_param_registry[map_]['n_agents'],], 
        chained_with=['map_']
    )

    AGENT_ID_EACH_TEAM = [range(0,map_param_registry[map_]['n_agents']),]
    AGENT_ID_EACH_TEAM_cv = ChainVar(
        lambda map_:[range(0,map_param_registry[map_]['n_agents']),], 
        chained_with=['map_']
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


    StateProvided = True
    AvailActProvided = True


    MaxEpisodeStep = map_param_registry[map_]['limit']
    MaxEpisodeStep_cv = ChainVar(
        lambda map_: map_param_registry[map_]['limit'], 
        chained_with=['map_']
    )

    n_actions = 6 + map_param_registry[map_]['n_enemies']
    n_actions_cv = ChainVar(
        lambda map_:6 + map_param_registry[map_]['n_enemies'], 
        chained_with=['map_']
    )
    obs_vec_length = -1
    EntityOriented = False

    obs_shape = 'Auto Load'
    state_shape = 'Auto Load'

    return_mat = False
    AvailActProvided = True # sc2 中，需要始终屏蔽掉不可用的动作
    reward_sparse=False
    render = False

    

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
        self.env = StarCraft2Env(
                            sc_version=ScenarioConfig.sc_version,
                            map_name=ScenarioConfig.map_,
                            step_mul=ScenarioConfig.step_mul,
                            difficulty=ScenarioConfig.difficulty,
                            game_version=ScenarioConfig.game_version,
                            reward_sparse=ScenarioConfig.reward_sparse,
                            return_mat=ScenarioConfig.return_mat,
                            reward_vec=ScenarioConfig.reward_vec,
                            seed=int(np.random.rand()*1e5), # 如果已经设定了主线程随机数种子，这里随机出来的数字则是确定的
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
        with HiddenPrints():
            reward, terminated, info = self.env.step(act)
            if ScenarioConfig.RewardAsUnity: reward = [reward]
            done = terminated
            ob = np.array(self.env.get_obs())
            info['state'] = self.env.get_state()
            info['avail-act'] = self.env.get_avail_actions()
            if 'battle_won' in info: info['win'] = info['battle_won']
            return (ob, reward, done, info)

    def reset(self):
        with HiddenPrints():
            self.env.reset()
            ob = np.array(self.env.get_obs())
            info = {}
            info['state'] = self.env.get_state()
            info['avail-act'] = self.env.get_avail_actions()
            if 'battle_won' in info: info['win'] = info['battle_won']
            return ob, info

    def render(self):
        return

