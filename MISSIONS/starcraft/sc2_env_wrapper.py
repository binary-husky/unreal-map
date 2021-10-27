import os,sys

class ScenarioConfig(object): # ADD_TO_CONF_SYSTEM 加入参数搜索路径 do not remove this comment !!!
    map_ = '3m'
    step_mul = 8
    difficulty = '7'
    game_version = 'latest'
    replay_dir = ''
    episode_limit = 60

    N_TEAM = 1  
    N_AGENT_EACH_TEAM = [3,]    # because map_ = '3m'
    AGENT_ID_EACH_TEAM = [range(0,3),]
    TEAM_NAMES = [  
                    'ALGORITHM.Starcraft.star_foundation->StarFoundation',
                ] 
    ActAsUnity = False
    RewardAsUnity = True

    state_provided = True
    avail_act_provided = True






def make_sc2_env(env_id, rank):
    return Env_Compat_Wrapper(rank)


# 一层套一层。。。这层是为了参数对齐
class Env_Compat_Wrapper():
    def __init__(self, rank):
        from smac.env import StarCraft2Env
        self.env = StarCraft2Env(map_name=ScenarioConfig.map_,
                            step_mul=ScenarioConfig.step_mul,
                            difficulty=ScenarioConfig.difficulty,
                            game_version=ScenarioConfig.game_version,
                            replay_dir=ScenarioConfig.replay_dir)

        env_info = self.env.get_env_info()
        self.observation_space = {  'state_shape': env_info["state_shape"], 
                                    'obs_shape':env_info["obs_shape"]}

        self.action_space =  {  'n_actions': env_info["n_actions"], 
                                'n_agents': env_info["n_agents"]}

        assert env_info["n_agents"] == ScenarioConfig.N_AGENT_EACH_TEAM[0], ('Changed a map? Reconfig ScenarioConfig Above!!')
        assert env_info["episode_limit"] == ScenarioConfig.episode_limit, ('Changed a map? Reconfig ScenarioConfig Above!!')

        self.id = rank
        pass

    def step(self, act):
        reward, terminated, info = self.env.step(act)
        reward = [reward]
        done = terminated
        ob = self.env.get_obs()
        info['state'] = self.env.get_state()
        info['avail-act'] = self.env.get_avail_actions()
        return (ob, reward, done, info)

    def reset(self):
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