from config import GlobalConfig
import importlib

def make_parallel_envs(process_pool, marker=''):
    from UTILS.shm_env import SuperpoolEnv
    from config import GlobalConfig
    from MISSIONS.env_router import load_scenario_config
    load_scenario_config()
    
    env_args_dict_list = [({
        'env_name':GlobalConfig.env_name, 
        'proc_index':i, 
        'marker':marker
    },) for i in range(GlobalConfig.num_threads)]

    if GlobalConfig.env_name == 'air_fight':
        # This particular env has a dll file 
        # that must be loaded in main process
        # 艹tmd有个dll必须在主进程加载
        from MISSIONS.air_fight.environment.pytransform import pyarmor_runtime
        pyarmor_runtime()


    if GlobalConfig.num_threads > 1:
        envs = SuperpoolEnv(process_pool, env_args_dict_list)
    else:
        envs = SuperpoolEnv(process_pool, env_args_dict_list)

    return envs


def load_scenario_config():
    env_name = GlobalConfig.env_name
    if env_name == 'collective_assult':
        from MISSIONS.collective_assult.collective_assult_parallel_run import ScenarioConfig    
    elif env_name == 'collective_assult_pvp':
        from MISSIONS.collective_assult_pvp.collective_assult_parallel_run import ScenarioConfig
    elif env_name == 'collective_assult':
        from MISSIONS.collective_assult.collective_assult_parallel_run import ScenarioConfig
    elif env_name == 'air_fight':
        from MISSIONS.air_fight.environment.air_fight_compat import ScenarioConfig
    elif 'native_gym' in env_name:
        from MISSIONS.native_gym.native_gym_config import ScenarioConfig
    elif 'sr_tasks' in env_name:
        assert '->' in env_name
        _, _env_name = env_name.split('->')
        ScenarioConfig = getattr(importlib.import_module('MISSIONS.sr_tasks.multiagent.scenarios.'+_env_name), 'ScenarioConfig')
    elif env_name == 'starcraft2':
        from MISSIONS.starcraft.sc2_env_wrapper import ScenarioConfig
    elif env_name == 'unity_game':
        from MISSIONS.unity_game.unity_game_wrapper import ScenarioConfig
    else:
        assert False, ('need to find path of ScenarioConfig')
    GlobalConfig.scenario_config = ScenarioConfig

def make_env_function(env_name, rank):
    load_scenario_config()
    
    if env_name == 'collective_assult':
        from MISSIONS.collective_assult.collective_assult_parallel_run import make_collective_assult_env as env_init_function
    elif env_name == 'collective_assult':
        from MISSIONS.collective_assult.collective_assult_parallel_run import make_collective_assult_env as env_init_function
    elif env_name == 'collective_assult_pvp':
        from MISSIONS.collective_assult_pvp.collective_assult_parallel_run import make_collective_assult_env as env_init_function
    elif env_name == 'air_fight':
        from MISSIONS.air_fight.environment.air_fight_compat import make_air_fight_env as env_init_function
    elif env_name == 'starcraft2':
        from MISSIONS.starcraft.sc2_env_wrapper import make_sc2_env as env_init_function
    elif env_name == 'unity_game':
        from MISSIONS.unity_game.unity_game_wrapper import make_env as env_init_function
    elif 'native_gym' in env_name:
        assert '->' in env_name
        _, native_gym_env_name = env_name.split('->')
        def env_init_function(placeholder, rank):
            import gym
            env = gym.make(native_gym_env_name)
            return env
    elif 'sr_tasks' in env_name:
        assert '->' in env_name
        _, _env_name = env_name.split('->')
        env_init_function = sr_tasks_env
    else:
        assert False, ('what is this env?', env_name)


    def _init_lambda_():
        env = env_init_function(env_name, rank)
        return env
        
    return _init_lambda_


def sr_tasks_env(env_id, rank):
    import multiagent.scenarios as scenarios
    from multiagent.environment import MultiAgentEnv
    assert 'sr_tasks' in env_id
    assert '->' in env_id
    _, env_id = env_id.split('->')
    Scenario = getattr(importlib.import_module('MISSIONS.sr_tasks.multiagent.scenarios.'+env_id), 'Scenario')

    scenario = Scenario(process_id=rank)
    world = scenario.make_world()

    env = MultiAgentEnv(world=world,
                        reset_callback=scenario.reset_world,
                        reward_callback=scenario.reward,
                        observation_callback=scenario.observation,
                        info_callback=scenario.info if hasattr(scenario, 'info') else None,
                        discrete_action=True,
                        done_callback=scenario.done)
    return env