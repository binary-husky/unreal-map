import numpy as np
import importlib
# defines scenario upon which the world is built
class BaseScenario(object):
    # create elements of the world
    def make_world(self):
        raise NotImplementedError()
    # create initial conditions of the world
    def reset_world(self, world):
        raise NotImplementedError()

def sr_tasks_env(env_id, rank):
    import multiagent.scenarios as scenarios

    Scenario = getattr(importlib.import_module('MISSION.sr_tasks.multiagent.scenarios.'+env_id), 'Scenario')
    scenario = Scenario(process_id=rank)
    world = scenario.make_world()
    if env_id == 'hunter_invader':
        from multiagent.environment_hi import MultiAgentEnv
    elif env_id == 'hunter_invader3d':
        from multiagent.environment_hi3d import MultiAgentEnv
    elif env_id == 'hunter_invader3d_v2':
        from multiagent.environment_hi3d import MultiAgentEnv
    elif env_id == 'cargo':
        from multiagent.environment_cargo import MultiAgentEnv
    else:
        from multiagent.environment import MultiAgentEnv
    env = MultiAgentEnv(world=world,
                        reset_callback=scenario.reset_world,
                        reward_callback=scenario.reward,
                        observation_callback=scenario.observation,
                        info_callback=scenario.info if hasattr(scenario, 'info') else None,
                        discrete_action=True,
                        done_callback=scenario.done)
    return env
