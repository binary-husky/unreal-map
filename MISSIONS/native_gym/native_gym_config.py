class ScenarioConfig(object): # ADD_TO_CONF_SYSTEM 加入参数搜索路径 do not remove this comment !!!
    N_TEAM = 1
    N_AGENT_EACH_TEAM = [1]
    AGENT_ID_EACH_TEAM = [range(1)]
    TEAM_NAMES = ['ALGORITHM.stable_baselines3.my_sac->BaselineFoundation'] 

    SingleAgentCompat = True