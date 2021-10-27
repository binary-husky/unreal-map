import os,sys

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
class ScenarioConfig(object): # ADD_TO_CONF_SYSTEM 加入参数搜索路径 do not remove this comment !!!
    map_ = '27m_vs_30m'
    step_mul = 8
    difficulty = '7'
    game_version = 'latest'
    replay_dir = ''
    episode_limit = 180

    N_TEAM = 1  
    N_AGENT_EACH_TEAM = [27,]    # because map_ = '3m'
    AGENT_ID_EACH_TEAM = [range(0,27),]
    TEAM_NAMES = [  
                    'ALGORITHM.Starcraft.star_foundation->StarFoundation',
                ] 
    ActAsUnity = False
    RewardAsUnity = True

    state_provided = True
    avail_act_provided = True

    obs_vec_length = 6
    max_steps_episode = 180
    max_steps_episode_cv = ChainVar(lambda episode_limit:episode_limit, chained_with=['episode_limit']) 
    return_mat = False

    n_action = 36

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
                            # return_mat=ScenarioConfig.return_mat,
                            # reward_only_positive=False,
                            replay_dir=ScenarioConfig.replay_dir)

        env_info = self.env.get_env_info()
        self.observation_space = {'state_shape': env_info["state_shape"], 
                                  'obs_shape':   env_info["obs_shape"]}

        self.action_space =  {'n_actions': env_info["n_actions"], 
                              'n_agents':  env_info["n_agents"]}

        assert env_info["n_agents"] == ScenarioConfig.N_AGENT_EACH_TEAM[0], ('Changed a map? Reconfig ScenarioConfig Above!!')
        assert env_info["episode_limit"] == ScenarioConfig.episode_limit, ('Changed a map? Reconfig ScenarioConfig Above!!',env_info["episode_limit"])

        self.id = rank
        pass

    def step(self, act):
        with HiddenPrints():
            reward, terminated, info = self.env.step(act)
            reward = [reward]
            done = terminated
            ob = self.env.get_obs()
            info['state'] = self.env.get_state()
            info['avail-act'] = self.env.get_avail_actions()
            return (ob, reward, done, info)

    def reset(self):
        with HiddenPrints():
            self.env.reset()
            ob = self.env.get_obs()
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