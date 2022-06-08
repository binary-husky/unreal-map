class ScenarioConfig(object):  
    '''
        ScenarioConfig: This config class will be 'injected' with new settings from JSONC.
        (E.g., override configs with ```python main.py --cfg example.jsonc```)
        (As the name indicated, ChainVars will change WITH vars it 'chained_with' during config injection)
        (please see UTILS.config_args to find out how this advanced trick works out.)
    '''
    N_TEAM = 1
    N_AGENT_EACH_TEAM = [1]
    AGENT_ID_EACH_TEAM = [range(1)]
    TEAM_NAMES = ['ALGORITHM.stable_baselines3.my_sac->BaselineFoundation'] 

    SingleAgentCompat = True

def env_init_function(native_gym_env_name, rank):
    import gym
    env = gym.make(native_gym_env_name)
    return env