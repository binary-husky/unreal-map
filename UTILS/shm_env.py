import numpy as np
import time
from MISSIONS.env_router import make_env_function
from UTILS.colorful import print亮红

N = lambda x: np.array(x)

# Here use a pool of multiprocess workers to control a bundle of environment to sync step
# SuperPool.add_target: in each process, initiate a class object named xxxx, 
#     example:
#     self.SuperPool.add_target(name='env', lam=EnvWithRay, args_list=env_args_dict_list)
# SuperPool.exec_target: in each process, make the object (id by name) to call its method
#     example:
#     self.SuperPool.exec_target(name='env', dowhat='step', args_list=actions)
#     self.SuperPool.exec_target(name='env', dowhat='reset')


# ! this class execute in child process
# Ray is much slower compare to our shm/pipe solution,
# we don't use it any more despite the class name
class EnvWithRay(object):
    def __init__(self, env_args_dict):
        env_name = env_args_dict['env_name']
        proc_index = env_args_dict['proc_index']
        env_init_fn = make_env_function(env_name=env_name, rank=proc_index)
        self.env = env_init_fn()
        # finally the env is initialized
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.echo = None

    def __del__(self):
        # print亮红('[shm_env.py] exec EnvWithRay exit')
        del self.env

    def step(self, act):
        if np.isnan(act).any(): 
            # env is paused, skip by returning previous obs
            assert self.echo is not None
            return self.echo
        # ! step here
        ob, reward, done, info = self.env.step(act)
        if isinstance(ob, list): 
            print('warning, ob is list, which is low-efficient')
            ob = np.array(ob, dtype=object)
        
        if np.any(done):
            # if the environment is terminated, 
            # first, put terminal obs into 'info'
            if info is None:
                info = {'obs-echo':ob}
            else:
                assert isinstance(info, dict), ('oh? info is not dictionary? did not expect that...')
                info.update({'obs-echo': ob.copy()})
            # second, automatically reset env
            ob = self.env.reset()
            if isinstance(ob, tuple):
                # some env like starcraft return (ob, info) tuple at reset
                # have info, then update info
                ob, info_reset = ob
                info = self.dict_update(info, info_reset)
                
        # preserve an echo here, 
        # will be use to handle unexpected env pause
        self.echo = [ob, reward, done, info]
        # give everything back to main process
        return (ob, reward, done, info)

    def dict_update(self, info, info_reset):
        for key in info_reset:
            if key in info: info[key+'-echo'] = info.pop(key)
        info.update(info_reset)
        return info

    def reset(self):
        return self.env.reset()

    def render(self):
        return self.env.render()

    def close(self):
        return None

    def get_act_space(self):
        return self.action_space

    def get_obs_space(self):
        return self.observation_space

    def get_act_space_str(self):
        return str(self.action_space)

    def get_obs_space_str(self):
        return str(self.observation_space)


# ! this class execute in main process
class SuperpoolEnv(object):
    def __init__(self, process_pool, env_args_dict_list, spaces=None):
        self.SuperPool = process_pool
        self.num_envs = len(env_args_dict_list)
        self.env_name_marker = env_args_dict_list[0][0]['marker']
        self.env = 'env' + self.env_name_marker
        self.SuperPool.add_target(name=self.env, lam=EnvWithRay, args_list=env_args_dict_list)
        try:
            self.observation_space = self.SuperPool.exec_target(name=self.env, dowhat='get_obs_space')[0]
            self.action_space =      self.SuperPool.exec_target(name=self.env, dowhat='get_act_space')[0]
        except:
            print亮红('Gym Space is unable to transfer between processes, using string instead')
            self.observation_space = self.SuperPool.exec_target(name=self.env, dowhat='get_obs_space_str')[0]
            self.action_space =      self.SuperPool.exec_target(name=self.env, dowhat='get_act_space_str')[0]
        # self.observation_space = self.SuperPool.exec_target(name=self.env, dowhat='get_obs_space_str')[0]
        # self.action_space =      self.SuperPool.exec_target(name=self.env, dowhat='get_act_space_str')[0]
        return

    def get_space(self):
        return {'obs_space': self.observation_space, 'act_space': self.action_space}

    def step(self, actions):
        # ENV_PAUSE = [np.isnan(thread_act).any() for thread_act in actions]
        results = self.SuperPool.exec_target(name=self.env, dowhat='step', args_list=actions)
        obs, rews, dones, infos = zip(*results)

        # if any(ENV_PAUSE):
        #     assert not all(ENV_PAUSE)
        #     return self.stack(ENV_PAUSE, obs, rews, dones, infos)
        # else:
        try:
            return np.stack(obs), np.stack(rews), np.stack(dones), np.stack(infos)
        except:
            assert False, ('unalign! ',obs, rews, dones)


    def reset(self):
        results = self.SuperPool.exec_target(name=self.env, dowhat='reset')
        # [ env.reset.remote() for env in self.ray_env_vector])
        if isinstance(results[0], tuple):
            obs, infos = zip(*results)
            return np.stack(obs), np.stack(infos)
        else:
            return np.stack(results)

    # def stack(self, ENV_PAUSE, *args):
    #     sample_index = ENV_PAUSE.index(False)
    #     args = list(args)
    #     for k in range(len(args)):
    #         args[k] = list(args[k])
    #         arg = args[k]
    #         sample = arg[sample_index]
    #         for thread_index in range(self.num_envs):
    #             if arg[thread_index] is not None: continue
    #             if isinstance(sample, np.ndarray):
    #                 arg[thread_index] = np.zeros_like(sample) + np.nan
    #             else:
    #                 assert False
    #         args[k] = np.stack(args[k])
    #     return args