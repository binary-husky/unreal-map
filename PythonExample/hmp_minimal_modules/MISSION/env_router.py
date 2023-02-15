import_path_ref = {
    "collective_assult": ("MISSION.collective_assult.collective_assult_parallel_run",          'ScenarioConfig'),
    "dca_multiteam": ("MISSION.dca_multiteam.collective_assult_parallel_run",                   'ScenarioConfig'),
    "collective_assult_debug": ("MISSION.collective_assult_debug.collective_assult_parallel_run", 'ScenarioConfig'),
    "air_fight": ("MISSION.air_fight.environment.air_fight_compat",                            'ScenarioConfig'),
    "native_gym": ("MISSION.native_gym.native_gym_config",                                     'ScenarioConfig'),
    "starcraft2": ("MISSION.starcraft.sc2_env_wrapper",                                        'ScenarioConfig'),
    "sc2": ("MISSION.starcraft.sc2_env_wrapper",                                               'ScenarioConfig'),
    "unity_game": ("MISSION.unity_game.unity_game_wrapper",                                    'ScenarioConfig'),
    "sr_tasks->cargo": ("MISSION.sr_tasks.multiagent.scenarios.cargo",                         'ScenarioConfig'),
    "sr_tasks->hunter_invader": ("MISSION.sr_tasks.multiagent.scenarios.hunter_invader",       'ScenarioConfig'),
    "sr_tasks->hunter_invader3d": ("MISSION.sr_tasks.multiagent.scenarios.hunter_invader3d",   'ScenarioConfig'),
    "sr_tasks->hunter_invader3d_v2": ("MISSION.sr_tasks.multiagent.scenarios.hunter_invader3d_v2",'ScenarioConfig'),
    "bvr": ("MISSION.bvr_sim.init_env",                                                        'ScenarioConfig'),
    "mathgame": ("MISSION.math_game.env",                                                      'ScenarioConfig'),
    "uhmap": ("MISSION.uhmap.uhmap_env_wrapper",                                               'ScenarioConfig'),
}

env_init_function_ref = {
    "collective_assult": ("MISSION.collective_assult.collective_assult_parallel_run",          'make_collective_assult_env'),
    "dca_multiteam": ("MISSION.dca_multiteam.collective_assult_parallel_run",                  'make_collective_assult_env'),
    "collective_assult_debug": ("MISSION.collective_assult_debug.collective_assult_parallel_run", 'make_collective_assult_env'),
    "air_fight": ("MISSION.air_fight.environment.air_fight_compat",                            'make_air_fight_env'),
    "native_gym": ("MISSION.native_gym.native_gym_config",                                     'env_init_function'),
    "starcraft2": ("MISSION.starcraft.sc2_env_wrapper",                                        'make_sc2_env'),
    "sc2": ("MISSION.starcraft.sc2_env_wrapper",                                               'make_sc2_env'),
    "unity_game": ("MISSION.unity_game.unity_game_wrapper",                                    'make_env'),
    "sr_tasks": ("MISSION.sr_tasks.multiagent.scenario",                                       'sr_tasks_env'),
    "bvr": ("MISSION.bvr_sim.init_env",                                                        'make_bvr_env'),
    "mathgame": ("MISSION.math_game.env",                                                      'make_math_env'),
    "uhmap": ("MISSION.uhmap.uhmap_env_wrapper",                                               'make_uhmap_env'),
}

##################################################################################################################################
##################################################################################################################################
from config import GlobalConfig
import importlib, os
from UTIL.colorful import print亮蓝



def load_ScenarioConfig():
    if GlobalConfig.env_name not in import_path_ref:
        assert False, ('need to find path of ScenarioConfig')
    import_path, ScenarioConfig = import_path_ref[GlobalConfig.env_name]
    GlobalConfig.ScenarioConfig = getattr(importlib.import_module(import_path), ScenarioConfig)


def make_env_function(env_name, rank):
    load_ScenarioConfig()
    ref_env_name = env_name

    if 'native_gym' in env_name:
        assert '->' in env_name
        ref_env_name, env_name = env_name.split('->')
    elif 'sr_tasks' in env_name:
        assert '->' in env_name
        ref_env_name, env_name = env_name.split('->')

    import_path, func_name = env_init_function_ref[ref_env_name]
    env_init_function = getattr(importlib.import_module(import_path), func_name)
    return lambda: env_init_function(env_name, rank)



def make_parallel_envs(process_pool, marker=''):
    from UTIL.shm_env import SuperpoolEnv
    from config import GlobalConfig
    from MISSION.env_router import load_ScenarioConfig
    load_ScenarioConfig()
    
    env_args_dict_list = [({
        'env_name':GlobalConfig.env_name, 
        'proc_index':i if 'test' not in marker else -(i+1), 
        'marker':marker
    },) for i in range(GlobalConfig.num_threads)]

    if GlobalConfig.env_name == 'air_fight':
        # This particular env has a dll file 
        # that must be loaded in main process
        # 艹tmd有个dll必须在主进程加载
        from MISSION.air_fight.environment.pytransform import pyarmor_runtime
        pyarmor_runtime()

    if GlobalConfig.env_name == 'bvr':
        # 1、如果没用hmp的docker，请设置好 YOUR_ROOT_PASSWORD，不止这一处，请全局搜索"YOUR_ROOT_PASSWORD"替换所有
        # 2、用docker的sock挂载到容器中，方法在SetupDocker.md中
        print亮蓝('[env_router]: here goes the docker in docker check.')
        YOUR_ROOT_PASSWORD = 'clara'  # the sudo password
        os.system("echo %s|sudo -S date"%YOUR_ROOT_PASSWORD) # get sudo power
        res = os.popen("sudo docker ps").read()
        if "CONTAINER ID" not in res:
            print亮蓝('[env_router]: Error checking docker in docker, can not control host docker interface!')
            raise "Error checking docker in docker, can not control host docker interface!"
        pass

    if GlobalConfig.env_name == 'collective_assult_debug':
        # This particular env has a cython file that needs to be compiled in main process
        # that must be loaded in main process
        from MISSION.collective_assult_debug.cython_func import laser_hit_improve3
    if GlobalConfig.env_name == 'dca_multiteam':
        # This particular env has a cython file that needs to be compiled in main process
        # that must be loaded in main process
        from MISSION.dca_multiteam.cython_func import laser_hit_improve3
    if GlobalConfig.env_name == 'uhmap':
        # This particular env has a cython file that needs to be compiled in main process
        # that must be loaded in main process
        from MISSION.uhmap.SubTasks.cython_func import tear_number_apart
    
    if GlobalConfig.num_threads > 1:
        envs = SuperpoolEnv(process_pool, env_args_dict_list)
    else:
        envs = SuperpoolEnv(process_pool, env_args_dict_list)

    return envs
