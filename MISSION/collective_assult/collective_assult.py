import gym
from gym import spaces
import numpy as np
from .multi_discrete import MultiDiscrete
from malib.spaces import Box, MASpace,  MAEnvSpec
import time, os
import pyglet
from pyglet.gl import *
from .envs.collective_assult_env import collective_assultEnvV1
# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!

def make_collective_assult_env(args, num_steps, benchmark=False):
    # scenario = gym.make('collective_assult-v1')
    scenario = collective_assultEnvV1(numguards=args.num_guards, numattackers = args.num_attackers, size=args.size)
    # create world
    world = scenario.world
    world.max_time_steps = num_steps
    # create multiagent environment
    if benchmark:
        env = collective_assultGlobalEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = collective_assultGlobalEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class collective_assultGlobalEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array']
    }

    def terminate(self):
        pass

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True):
        self.ob_rms = None
        self.world = world

        self.agents = self.world.agents
        # set required vectorized gym env property
        self.n = len(world.agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        # environment parameters
        self.discrete_action_space = True
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = True #False
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False
        # if true, every agent has the same reward
        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False


        # configure spaces
        self.action_space = []
        self.observation_space = []
        obs_shapes = []
        self.agent_num = len(self.agents)
        for agent in self.agents:
            total_action_space = []
            # physical action space #执行这个
            if self.discrete_action_space:
                u_action_space = spaces.Discrete((world.dim_p) * 2 + 2)   ##
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

        # action has 8 values: 
        # nothing, +forcex, -forcex, +forcey, -forcey, +rot, -rot, shoot 
        self.action_spaces = MASpace(tuple(Box(low=0., high=1., shape=((world.dim_p) * 2 + 2,)) for _ in range(self.agent_num)))  ##
        self.observation_spaces = MASpace(tuple(Box(low=-np.inf, high=+np.inf, shape=obs_shape) for obs_shape in obs_shapes))

        self.env_specs = MAEnvSpec(self.observation_spaces, self.action_spaces)
        self.action_range = [0., 1.]
        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
        # mixer.init()
        soundFiles = collective_assult.__file__[:-11]+'envs/Game/'
        # bulletFile = os.path.realpath(__file__)[:-13]+'Game/bullet.mp3'
        # mixer.music.load(soundFiles+'bullet.mp3')
        # print(collective_assult.__file__)
        # time.sleep(5)
        self.prevShot, self.shot = False, False     # used for rendering
        self._reset_render()

    def step(self, action_n):
        # print('first step')
        ## Sequence:
        ## First set the actions for all agents
        ## 0- For each bullet, check how many agents are killed
        ##    Remove those agents and the bullet
        ## 1- Perform movement
        ## 2- Shoot? -> generate a new bullet
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        self.agents = self.world.agents
        # set action for each agent
        # action originally had 5 values - accel, +forcex, -forcex, +forcey, -forcey
        # I have some doubt on how term acceleration is used in computation - check _set_action()
        # I added 2 extra components, rotation and shoot 
        # print('action_n')
        # print(action_n)
        for i, agent in enumerate(self.agents):
            action = action_n[i]
            # print(action)
            # action = np.array(action_n[i]).reshape(((self.world.dim_p-1)*2+3,)) ##
            self._set_action(action, agent, self.action_space[i]) # sets the actions in the agent object


        # advance world state
        ## actions are already set in the objects, so we can simply pass step without any argument
        self.world.step() # world is the collective_assult-v0 environment, step function is in core.py file

        # record observation for each agent
        for agent in self.agents: ##
            obs_n.append(self._get_obs(agent))
            reward_n.append(self._get_reward(agent))
            # done_n.append(self._get_done(agent))

            info_n['n'].append(self._get_info(agent))

        ## implement single done reflecting game state
        done = self._get_done()
        # all agents get total reward in cooperative case
        # print(reward_n)
        reward = np.sum(reward_n)
        # print(reward)
        if self.shared_reward:
            reward_n = [reward] * self.n
        self.world.time_step += 1
        obs_n = np.array(obs_n)
        return obs_n, reward_n, done, info_n

    def reset(self):
        # reset world
        self.reset_callback()
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []
        self.agents = self.world.agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        obs_n = np.array(obs_n)
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

    # get done for the whole environment
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self):
        # done if any attacker reached landmark, attackers win
        # 注释代码
        # th = self.world.fortDim
        # for attacker in self.world.alive_attackers:
        #     dist = np.sqrt(np.sum(np.square(attacker.state.p_pos-self.world.doorLoc)))
        #     if dist < th:
        #         print('attacker reached fort')
        #         self.world.gameResult[2] = 1
        #         return(True)

        # done if all attackers are dead, guards win
        if self.world.numAliveAttackers == 0:
            # print('all attackers dead', guards all win)
            self.world.gameResult[0] = 1
            return(True)
        elif self.world.numAliveGuards == 0:
        # print('all guards dead', attcker all win )
            self.world.gameResult[2] = 1
            return (True)
            # done if max number of time steps over, guards win
        elif self.world.time_step == self.world.max_time_steps-1:
            # print('max number of time steps')
            if self.world.numAliveAttackers < self.world.numAliveGuards:
                # print(guards win)
                self.world.gameResult[1] = 1
            elif self.world.numAliveAttackers > self.world.numAliveGuards:
                # print(attackers win)
                self.world.gameResult[3] = 1
            else:
                # print(draw )
                self.world.gameResult[4] = 1
            return(True)


        # otherwise not done
        return False


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
            # print('self.discrete_action_input', self.discrete_action_input) # True
            # physical action
            if self.discrete_action_input:
                agent.action.u = np.zeros(self.world.dim_p)     ## We'll use this now for Graph NN
                # process discrete action
                ## if action[0] == 0, then do nothing
                if action[0] == 1: agent.action.u[0] = +1.0
                if action[0] == 2: agent.action.u[0] = -1.0
                if action[0] == 3: agent.action.u[1] = +1.0
                if action[0] == 4: agent.action.u[1] = -1.0
                if action[0] == 5: agent.action.u[2] = +agent.max_rot
                if action[0] == 6: agent.action.u[2] = -agent.max_rot
                agent.action.shoot = True #if action[0] == 7 else False

            else:
                if self.force_discrete_action:       
                    d = np.argmax(action[0])
                    action[0][:] = 0.0
                    action[0][d] = 1.0
                if self.discrete_action_space:      ## this was begin used in PR2 Paper
                    # print('action', action)
                    agent.action.u[0] += action[0][1] - action[0][2]    ## each is 0 to 1, so total is -1 to 1
                    agent.action.u[1] += action[0][3] - action[0][4]    ## same as above
                    
                    ## simple shooting action
                    agent.action.shoot = True if action[0][6]>0.5 else False   # a number greater than 0.5 would mean shoot

                    ## simple rotation model
                    agent.action.u[2] = 2*(action[0][5]-0.5)*agent.max_rot
            
                else:
                    agent.action.u = action[0]
            sensitivity = 5.0   # default if no value specified for accel
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u[:2] *= sensitivity
            
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
    def render(self, attn_list = None, mode='human', close=False):
        # attn_list = [[teamates_attn, opp_attn] for each team]
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
                from .collective_assult import rendering
                self.viewers[i] = rendering.Viewer(700,700)



        # create rendering geometry and text for the scene
        if self.render_geoms is None or True:
            # import rendering only if we need it (and don't import for headless machines)
            #from gym.envs.classic_control import rendering
            from .collective_assult import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
            self.render_texts = []
            self.render_texts_xforms = []


            # # add black background for active world, border of active region/world
            # xMin, xMax, yMin, yMax = self.world.wall_pos
            # borPts = np.array([[xMin,yMin],
            #                     [xMax,yMin],
            #                     [xMax,yMax],
            #                     [xMin,yMax]])
            # geom = rendering.make_polygon(borPts)
            # geom.set_color(*[0.5,0.5,0.5], alpha=1)
            #
            # xform = rendering.Transform()
            # geom.add_attr(xform)
            # self.render_geoms.append(geom)
            # self.render_geoms_xform.append(xform)

            # distplay text
            # label = pyglet.text.Label('Hello, world',
            #               font_name='Times New Roman',
            #               font_size=36,
            #               x=0, y=0,
            #               anchor_x='center', anchor_y='center')
            # label.draw()
            # xform = rendering.Transform()
            # self.render_geoms.append(label)
            # self.render_geoms_xform.append(xform)

            ## add the fort door
            # doorPts = np.array([[-0.2,
            #取消堡垒
            # geom = rendering.make_circle(self.world.fortDim)
            # geom.set_color(*[0,1,1], alpha=1)
            # xform = rendering.Transform()
            # geom.add_attr(xform)
            # xform.set_translation(*self.world.doorLoc)
            # self.render_geoms.append(geom)
            # self.render_geoms_xform.append(xform)

            # ---------------- visualize attention ---------------- #
            # select reference, show attn wrt this agent
            for i, agent in enumerate(self.world.agents):
                if agent.alive or i == self.world.numGuards-1:
                    k = i
                    break

            if self.world.vizAttn and attn_list is not None:
                # will plot attention for dead agents as well, we would know if the agents are able disregard the dead agents based of alive flag = 0
                # print('inside attn viz')
                for i, agent in enumerate(self.world.agents):
                    if agent.alive or self.world.vizDead:   # alive agents are always visualized, dead are visualized if asked
                        if i != k:
                            # if it is in the same team
                            if i < self.world.numGuards:
                                attn = attn_list[0][0][k,i]
                            else:
                            # if opponent
                                attn = attn_list[0][1][k,i-self.world.numGuards]
                                # print('attacker')
                            geom = rendering.make_circle(agent.size*(1+attn))
                            xform = rendering.Transform()
                            geom.add_attr(xform)
                            xform.set_translation(*agent.state.p_pos)
                            alpha = 0.9 if agent.alive else 0.3
                            geom.set_color(*[1,1,0], alpha=alpha)
                            self.render_geoms.append(geom)
                            self.render_geoms_xform.append(xform)

            # visualize the dead agents
            if self.world.vizDead:
                for agent in self.world.agents:
                    # print(agent.name, agent.alive)
                    if not agent.alive:
                        geom = rendering.make_circle(agent.size)
                        xform = rendering.Transform()
                        geom.add_attr(xform)
                        xform.set_translation(*agent.state.p_pos)
                        geom.set_color(*agent.color, alpha=1)
                        head = rendering.make_circle(0.5*agent.size)
                        head.set_color(*agent.color, alpha=1)
                        headXform = rendering.Transform()
                        head.add_attr(headXform)
                        shift = 0.8*agent.size*np.array([np.cos(agent.state.p_ang), np.sin(agent.state.p_ang)])
                        headLoc = agent.state.p_pos+shift
                        headXform.set_translation(*headLoc)
                        self.render_geoms.append(head)
                        self.render_geoms_xform.append(headXform)
                        self.render_geoms.append(geom)
                        self.render_geoms_xform.append(xform)
                        self.render_texts.append(agent.name[5:])
                        self.render_texts_xforms.append(agent.state.p_pos)

            # visualize alive agents
            for entity in self.world.active_entities:  ## won't work with obstacles
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
                    self.render_texts.append(entity.name[5:])
                    self.render_texts_xforms.append(entity.state.p_pos)
                    # print(entity.name)

                    if entity.action.shoot and entity.numbullets > 0:
                    # if entity.action.shoot:
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

            # dot for reference agent
            if self.world.vizAttn and attn_list is not None:
                # will plot attention for dead agents as well, we would know if the agents are able disregard the dead agents based of alive flag = 0
                # print('inside attn viz')
                for i, agent in enumerate(self.world.agents):
                    if agent.alive or self.world.vizDead:   # alive agents are always visualized, dead are visualized if asked
                        # select reference agent for attention viz
                        if i == k:
                            # simply put a black dot at the center of this agent
                            geom = rendering.make_circle(0.5*agent.size)
                            xform = rendering.Transform()
                            geom.add_attr(xform)
                            xform.set_translation(*agent.state.p_pos)
                            geom.set_color(*0.5*agent.color, alpha=1)
                            self.render_geoms.append(geom)
                            self.render_geoms_xform.append(xform)


              
            # # add grey strips, corners of visualization window
            # corPtsArr = [np.array([[xMin,yMax],
            #                     [xMax,yMax],
            #                     [xMax,1],
            #                     [xMin,1]]),
            #             np.array([[xMin,-1],
            #                     [xMax,-1],
            #                     [xMax,yMin],
            #                     [xMin,yMin]])]
            # for corPts in corPtsArr:
            #     geom = rendering.make_polygon(corPts)
            #     geom.set_color(*[0.5,0.5,0.5], alpha=1)
            #     xform = rendering.Transform()
            #     geom.add_attr(xform)
            #     self.render_geoms.append(geom)
            #     self.render_geoms_xform.append(xform)

            # add geoms to viewer ## viewer is object of class Viewer defined in rendering.py file inside collective_assult
            for viewer in self.viewers:
                viewer.geoms = []
                viewer.texts = []
                viewer.text_poses = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)
                for text, xform in zip(self.render_texts, self.render_texts_xforms):
                    viewer.add_text(text,xform)
                # print(self.render_texts)


        results = []
        for i in range(len(self.viewers)):
            from collective_assult import rendering
            # update bounds to center around agent
            xMin, xMax, yMin, yMax = self.world.wall_pos
            cam_range = xMax
            if self.shared_viewer:
                pos = np.zeros(self.world.dim_p)
            else:
                pos = self.agents[i].state.p_pos
            self.viewers[i].set_bounds(pos[0]-cam_range,pos[0]+cam_range,pos[1]-cam_range,pos[1]+cam_range)
            # results.append(
            self.viewers[i].render(return_rgb_array = False)#mode=='human')    # actual rendering
            #)
        # if self.shot and not self.prevShot:
            # mixer.music.play()
        self.prevShot = self.shot
        # return('ankur')
        return results          ## this thing is really doing nothing

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


