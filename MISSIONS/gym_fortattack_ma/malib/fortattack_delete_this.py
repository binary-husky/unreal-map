import gym
import gym_fortattack
from gym import spaces
import numpy as np
from multiagent.multi_discrete import MultiDiscrete
from malib.spaces import Box, MASpace,  MAEnvSpec
import time, os
from pygame import mixer  # Load the popular external library

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!

def make_fortattack_env(benchmark=False):
    scenario = gym.make('fortattack-v1')
    # create world
    world = scenario.world
    # create multiagent environment
    if benchmark:
        env = FortAttackGlobalEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = FortAttackGlobalEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class FortAttackGlobalEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array']
    }

    def terminate(self):
        pass

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True):

        self.world = world
        self.world.discrete_action = True
        self.agents = self.world.policy_agents
        # set required vectorized gym env property
        self.n = len(world.policy_agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        # environment parameters
        self.discrete_action_space = True
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = world.discrete_action if hasattr(self.world, 'discrete_action') else False
        # if true, every agent has the same reward
        self.shared_reward = world.collaborative if hasattr(self.world, 'collaborative') else False
        self.time = 0

        # configure spaces
        self.action_space = []
        self.observation_space = []
        obs_shapes = []
        self.agent_num = len(self.agents)
        ## I think everything inside this loop is useless, self.action_spaces is used later and not self.action_space
        for agent in self.agents:
            total_action_space = []
            # physical action space
            if self.discrete_action_space:
                u_action_space = spaces.Discrete((world.dim_p-1) * 2 + 3)   ##
            else:
                u_action_space = spaces.Box(low=-agent.u_range, high=+agent.u_range, shape=(world.dim_p,), dtype=np.float32)
            if agent.movable:
                total_action_space.append(u_action_space)
            # communication action space
            if self.discrete_action_space:
                c_action_space = spaces.Discrete(world.dim_c)
            else:
                c_action_space = spaces.Box(low=0.0, high=1.0, shape=(world.dim_c,), dtype=np.float32)
            if not agent.silent:
                total_action_space.append(c_action_space)
            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])
            # observation space
            obs_dim = len(observation_callback(agent, self.world))
            obs_shapes.append((obs_dim,))
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
            agent.action.c = np.zeros(self.world.dim_c)
        # simpified for non-comm game

        # self.action_spaces = MASpace(tuple(Box(low=-1., high=1., shape=(1,)) for _ in range(self.agent_num)))
        # self.observation_spaces = MASpace(tuple(Discrete(1) for _ in range(self.agent_num)))

        # action originally had 5 values - accel, +forcex, -forcex, +forcey, -forcey
        # I added extra 2 components, change in rotation angle and shoot 
        self.action_spaces = MASpace(tuple(Box(low=0., high=1., shape=((world.dim_p-1) * 2 + 3,)) for _ in range(self.agent_num)))  ##
        self.observation_spaces = MASpace(tuple(Box(low=-np.inf, high=+np.inf, shape=obs_shape) for obs_shape in obs_shapes))

        self.env_specs = MAEnvSpec(self.observation_spaces, self.action_spaces)
        self.action_range = [0., 1.]
        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
        mixer.init()
        soundFiles = gym_fortattack.__file__[:-11]+'envs/Game/'
        # bulletFile = os.path.realpath(__file__)[:-13]+'Game/bullet.mp3'
        mixer.music.load(soundFiles+'bullet.mp3')
        # print(gym_fortattack.__file__)
        # time.sleep(5)
        self.prevShot, self.shot = False, False     # used for rendering
        self._reset_render()

    def step(self, action_n):
        # print('first step')
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        self.agents = self.world.policy_agents
        # set action for each agent
        # action originally had 5 values - accel, +forcex, -forcex, +forcey, -forcey
        # I have some doubt on how term acceleration is used in computation - check _set_action()
        # I added 2 extra components, rotation and shoot 
        for i, agent in enumerate(self.agents):
            action = np.array(action_n[i]).reshape(((self.world.dim_p-1)*2+3,)) ##
            self._set_action(action, agent, self.action_space[i]) # sets the actions in the agent object
        # advance world state
        ## actions are already set in the objects, so we can simply pass step without any argument
        self.world.step() # world is the fortattack-v0 environment, step function is in core.py file
        
        # record observation for each agent
        for agent in self.agents: ##
            obs_n.append(self._get_obs(agent))
            reward_n.append(self._get_reward(agent))
            done_n.append(self._get_done(agent))

            info_n['n'].append(self._get_info(agent))

        # all agents get total reward in cooperative case
        reward = np.sum(reward_n)
        if self.shared_reward:
            reward_n = [reward] * self.n
        return obs_n, reward_n, done_n, info_n

    def reset(self):
        # reset world
        self.reset_callback()
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        return obs_n

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent)

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        # process action
        if isinstance(action_space, MultiDiscrete):
            act = []
            size = action_space.high - action_space.low + 1
            index = 0
            for s in size:
                act.append(action[index:(index+s)])
                index += s
            action = act
        else:
            action = [action]

        if agent.movable:
            # physical action
            if self.discrete_action_input:
                agent.action.u = np.zeros(self.world.dim_p)
                # process discrete action
                if action[0] == 1: agent.action.u[0] = -1.0
                if action[0] == 2: agent.action.u[0] = +1.0
                if action[0] == 3: agent.action.u[1] = -1.0
                if action[0] == 4: agent.action.u[1] = +1.0
            else:
                if self.force_discrete_action:
                    d = np.argmax(action[0])
                    action[0][:] = 0.0
                    action[0][d] = 1.0
                if self.discrete_action_space:
                    # print('action', action)
                    agent.action.u[0] += action[0][1] - action[0][2]    ## each is 0 to 1, so total is -1 to 1
                    agent.action.u[1] += action[0][3] - action[0][4]    ## same as above
                else:
                    agent.action.u = action[0]
            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity
            
            ## simple rotation model
            agent.action.u[2] = 2*(action[0][5]-0.5)*agent.max_rot
            
            ## simple shooting action
            agent.action.shoot = True if action[0][6]>0.5 else False   # a number greater than 0.5 would mean shoot

            ## remove used actions
            action = action[1:]
        

        if not agent.silent:
            # communication action
            if self.discrete_action_input:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0]] = 1.0
            else:
                agent.action.c = action[0]
            action = action[1:]
        
        # make sure we used all elements of action
        assert len(action) == 0

    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    # render environment
    def render(self, mode='human', close=False):
        self.shot = False
        if mode == 'human':
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            message = ''
            for agent in self.world.agents:
                comm = []
                for other in self.agents:
                    if other is agent: continue
                    if np.all(other.state.c == 0):
                        word = '_'
                    else:
                        word = alphabet[np.argmax(other.state.c)]
                    message += (other.name + ' to ' + agent.name + ': ' + word + '   ')
            # print(message)

        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                #from gym.envs.classic_control import rendering
                from gym_fortattack import rendering
                self.viewers[i] = rendering.Viewer(700,700)

        # create rendering geometry
        if self.render_geoms is None or True:
            # import rendering only if we need it (and don't import for headless machines)
            #from gym.envs.classic_control import rendering
            from gym_fortattack import rendering
            self.render_geoms = []
            self.render_geoms_xform = []

            # add black background
            doorPts = np.array([[-1,1],
                                [1,1],
                                [1,-1],
                                [-1,-1]])
            geom = rendering.make_polygon(doorPts)
            # geom.set_color(*[0.35,0.35,0.85], alpha=0.5)
            xform = rendering.Transform()
            geom.add_attr(xform)
            self.render_geoms.append(geom)
            self.render_geoms_xform.append(xform)

            ## add the fort door
            doorPts = np.array([[-0.2,1],
                                [0.2,1],
                                [0.2,0.9],
                                [-0.2,0.9]])
            geom = rendering.make_polygon(doorPts)
            geom.set_color(*[0,1,1], alpha=1)
            xform = rendering.Transform()
            geom.add_attr(xform)
            self.render_geoms.append(geom)
            self.render_geoms_xform.append(xform)

            for entity in self.world.entities:  ## won't work with obstacles
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()
                geom.add_attr(xform)
                xform.set_translation(*entity.state.p_pos)
                
                if 'agent' in entity.name:
                    geom.set_color(*entity.color, alpha=1)
                    head = rendering.make_circle(0.5*entity.size)
                    head.set_color(*entity.color, alpha=1)
                    headXform = rendering.Transform()
                    head.add_attr(headXform)
                    shift = 0.8*entity.size*np.array([np.cos(entity.state.p_ang), np.sin(entity.state.p_ang)])
                    headLoc = entity.state.p_pos+shift
                    headXform.set_translation(*headLoc)
                    self.render_geoms.append(head)
                    self.render_geoms_xform.append(headXform)

                    if entity.action.shoot:
                        self.shot = True
                        ## render the laser shots, maybe add extra delay when there is a laser shot
                        v = self.world.get_tri_pts_arr(entity)[:2,:].transpose()
                        laser = rendering.make_polygon(v)
                        laser.set_color(*entity.color, alpha=0.3)
                        laserXform = rendering.Transform()
                        laser.add_attr(laserXform)
                        self.render_geoms.append(laser)
                        self.render_geoms_xform.append(laserXform)

                else:
                    geom.set_color(*entity.color)

                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)
                

            # add geoms to viewer
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)


        results = []
        for i in range(len(self.viewers)):
            from gym_fortattack import rendering
            # update bounds to center around agent
            cam_range = 1
            if self.shared_viewer:
                pos = np.zeros(self.world.dim_p)
            else:
                pos = self.agents[i].state.p_pos
            self.viewers[i].set_bounds(pos[0]-cam_range,pos[0]+cam_range,pos[1]-cam_range,pos[1]+cam_range)
            results.append(self.viewers[i].render(return_rgb_array = mode=='rgb_array'))
        if self.shot and not self.prevShot:
            mixer.music.play()
        self.prevShot = self.shot
        return results

    # create receptor field locations in local coordinate frame
    def _make_receptor_locations(self, agent):
        receptor_type = 'polar'
        range_min = 0.05 * 2.0
        range_max = 1.00
        dx = []
        # circular receptive field
        if receptor_type == 'polar':
            for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
                for distance in np.linspace(range_min, range_max, 3):
                    dx.append(distance * np.array([np.cos(angle), np.sin(angle)]))
            # add origin
            dx.append(np.array([0.0, 0.0]))
        # grid receptive field
        if receptor_type == 'grid':
            for x in np.linspace(-range_max, +range_max, 5):
                for y in np.linspace(-range_max, +range_max, 5):
                    dx.append(np.array([x,y]))
        return dx


# vectorized wrapper for a batch of multi-agent environments
# assumes all environments have the same observation and action space
class BatchMultiAgentEnv(gym.Env):
    metadata = {
        'runtime.vectorized': True,
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, env_batch):
        self.env_batch = env_batch

    @property
    def n(self):
        return np.sum([env.n for env in self.env_batch])

    @property
    def action_space(self):
        return self.env_batch[0].action_space

    @property
    def observation_space(self):
        return self.env_batch[0].observation_space

    def step(self, action_n, time):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        i = 0
        for env in self.env_batch:
            obs, reward, done, _ = env.step(action_n[i:(i+env.n)], time)
            i += env.n
            obs_n += obs
            # reward = [r / len(self.env_batch) for r in reward]
            reward_n += reward
            done_n += done
        return obs_n, reward_n, done_n, info_n

    def reset(self):
        obs_n = []
        for env in self.env_batch:
            obs_n += env.reset()
        return obs_n

    # render environment
    def render(self, mode='human', close=True):
        results_n = []
        for env in self.env_batch:
            results_n += env.render(mode, close)
        return results_n


