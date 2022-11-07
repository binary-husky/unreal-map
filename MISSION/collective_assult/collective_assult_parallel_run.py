import gym
from gym import spaces
import numpy as np
import sys
sys.path.append('./MISSION/collective_assult')
from .multi_discrete import MultiDiscrete
from malib.spaces import Box, MASpace, MAEnvSpec
import time, os
import json
# from pyglet.gl import *
# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
from config import ChainVar
from UTIL.colorful import print亮绿
import cProfile, pstats
class ScenarioConfig(object):  
    '''
        ScenarioConfig: This config class will be 'injected' with new settings from JSONC.
        (E.g., override configs with ```python main.py --cfg example.jsonc```)
        (As the name indicated, ChainVars will change WITH vars it 'chained_with' during config injection)
        (please see UTIL.config_args to find out how this advanced trick works out.)
    '''
    num_guards = 50
    num_attackers = 50
    size = 5.0 # 10*10
    num_steps = 180

    random_jam_prob = 0.0
    introduce_terrain = False
    terrain_parameters = [0, 0]

    MaxEpisodeStep = 180
    MaxEpisodeStep_cv = ChainVar(lambda num_steps:num_steps, chained_with=['num_steps'])

    init_distance = 4  # 5-2,5+2
    render = False
    render_with_unity = False
    render_ip_with_unity = '127.0.0.1:11088'

    episode_limit = num_steps
    episode_limit_cv = ChainVar(lambda num_steps:num_steps, chained_with=['num_steps'])

    map_ = 'default'
    benchmark = False
    N_TEAM = 1

    N_AGENT_EACH_TEAM = [50,]
    N_AGENT_EACH_TEAM_cv = ChainVar(lambda num_guards:[num_guards,], chained_with=['num_guards']) 

    AGENT_ID_EACH_TEAM = [range(0,50),]
    AGENT_ID_EACH_TEAM_cv = ChainVar(lambda num_guards:[range(0,num_guards),], chained_with=['num_guards']) 

    TEAM_NAMES = [ 'ALGORITHM.Starcraft.star_foundation->StarFoundation', ]
    obs_vec_length = 16

    num_entity = num_attackers
    num_entity_cv = ChainVar(lambda num_attackers:num_attackers, chained_with=['num_attackers']) 

    num_object = num_attackers + num_guards
    num_object_cv = ChainVar(lambda num_attackers, num_guards: num_attackers+num_guards, chained_with=['num_attackers','num_guards']) 

    uid_dictionary = {'agent_uid':range(0,50), 'entity_uid':range(50,100), 'guards_uid':range(0,50), 'attackers_uid':range(50,100)}
    uid_dictionary_cv = ChainVar(lambda num_guards,num_attackers :{
        'agent_uid':range(0,num_guards), 
        'entity_uid':range(num_guards,num_attackers+num_guards), 
        'guards_uid':range(0,num_guards), 
        'attackers_uid':range(num_guards,num_attackers+num_guards)
        }, chained_with=['num_guards','num_attackers']) 

    dec_dictionary = {'alive':0, 'pos':range(1,3), 'ang':3, 'vel':range(4,6), 'id':6}
    StateProvided = False
    AvailActProvided = False
    RewardAsUnity = False
    ObsAsUnity = False   # 减少IPC负担
    EntityOriented = True

    # 调试
    MCOM_DEBUG = False
    DISALBE_RED_FUNCTION = False
    half_death_reward = True

    n_actions = 7
    
def make_collective_assult_env(env_id, rank):
    # scenario = gym.make('collective_assult-v1')
    from .envs.collective_assult_env import collective_assultEnvV1
    scenario = collective_assultEnvV1( numguards=ScenarioConfig.num_guards, 
                                numattackers = ScenarioConfig.num_attackers, 
                                size=ScenarioConfig.size)
    # create world
    world = scenario.world
    world.max_time_steps = ScenarioConfig.num_steps
    # create multiagent environment
    if ScenarioConfig.benchmark:
        env = collective_assultGlobalEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data, id=rank)
    else:
        env = collective_assultGlobalEnv(world, scenario.reset_world, scenario.reward, scenario.observation, id=rank)
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
                 done_callback=None, shared_viewer=True, id = -1):
        self.render_on = ScenarioConfig.render
        self.render_with_unity = ScenarioConfig.render_with_unity
        self.render_ip_with_unity = ScenarioConfig.render_ip_with_unity
        # 当并行时，只有0号环境可以渲染
        self.id = id
        if self.id!=0: 
            self.render_on = False
            self.render_with_unity = False

        if self.render_with_unity:
            from VISUALIZE.mcom import tcp_client
            self.mcv = tcp_client(ip=self.render_ip_with_unity)

        self.ob_rms = None
        self.world = world

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
            if not agent.silent:
                # communication action space
                if self.discrete_action_space:
                    c_action_space = spaces.Discrete(world.dim_c)
                else:
                    c_action_space = spaces.Box(low=0.0, high=1.0, shape=(world.dim_c,), dtype=np.float32)
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
            obs_dim = observation_callback(agent, self.world, get_obs_dim=True)
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
            self.viewers = None
        else:
            assert False
        # mixer.init()
        soundFiles = os.path.dirname(__file__) + 'envs/Game/'
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
        info_n = []
        self.agents = self.world.policy_agents
        # set action for each agent
        # action originally had 5 values - accel, +forcex, -forcex, +forcey, -forcey
        # I have some doubt on how term acceleration is used in computation - check _set_action()
        # I added 2 extra components, rotation and shoot 
        # print('action_n')
        # print(action_n)
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i]) # sets the actions in the agent object
        
        # calculate terrain
        self.world.set_terrain_adv()
        # advance world state
        ## actions are already set in the objects, so we can simply pass step without any argument
        self.world.step() # world is the collective_assult-v0 environment, step function is in core.py file
        
        # record observation for each agent
        if ScenarioConfig.ObsAsUnity:
            o_, info_n = self._get_obs(self.agents[0])
            obs_n.append(o_)

        
        for agent in self.agents: ##
            if not ScenarioConfig.ObsAsUnity: 
                o_, info_ = self._get_obs(agent)
                obs_n.append(o_)
                info_n.append(info_)
            reward_n.append(self._get_reward(agent))
            # done_n.append(self._get_done(agent))

        
        ## implement single done reflecting game state
        done, info = self._get_done()
        info['others'] = info_n
        # all agents get total reward in cooperative case
        # print(reward_n)
        reward = np.sum(reward_n)
        # print(reward)
        if self.shared_reward:
            reward_n = [reward] * self.n
        self.world.time_step += 1
        obs_n = np.array(obs_n)
        if self.render_on: 
            if self.render_with_unity: 
                self.unity_render()
            else:
                self.render()
        return obs_n, reward_n, done, info

    def reset(self):
        # reset world
        self.reset_callback()
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []
        info_n = []
        self.agents = self.world.policy_agents
        # calculate terrain
        self.world.set_terrain_adv()
        if ScenarioConfig.ObsAsUnity:
            o_, info_n = self._get_obs(self.agents[0])
            obs_n.append(o_)
        else:
            for agent in self.agents:
                o_, info_ = self._get_obs(agent)
                obs_n.append(o_)       
                info_n.append(info_)
        info = {}
        info['others'] = info_n
        obs_n = np.array(obs_n)
        return obs_n, info

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0), None
        return self.observation_callback(agent, self.world)

    # get done for the whole environment
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self):
        
        terminate_cond = [    
            self.world.numAliveAttackers == 0,
            self.world.numAliveGuards == 0,
            self.world.time_step == self.world.max_time_steps-1
        ]
        if any(terminate_cond):
            guard_win = (self.world.numAliveAttackers*self.world.numGuards <= self.world.numAliveGuards*self.world.numAttackers)
            if self.world.numGuards != self.world.numAttackers:
                pass
                # with open('./test_only_profilex.txt','a+') as f:
                #     f.write(str(guard_win) + 'init_theta%.2f@%d: AliveGuards %d, AliveAttackers %d, orig: %d-%d \n'%(
                #     self.world.init_theta,
                #     self.world.time_step,
                #     self.world.numAliveGuards,
                #     self.world.numAliveAttackers,
                #     self.world.numGuards,
                #     self.world.numAttackers
                #     ))
            if terminate_cond[0]:
                self.world.gameResult[0] = 1
                return True, {'win': True}
            if terminate_cond[1]:
                self.world.gameResult[2] = 1
                return True, {'win': False}
            if terminate_cond[2]:
                return True, {'win': guard_win}


        # otherwise not done
        return False, {}


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


    def unity_render(self):
        summary_list = []
        for i, agent in enumerate(self.world.agents):
            pos = np.array([agent.state.p_pos[0],agent.state.p_pos[1]])
            shift = 0.8*agent.size*np.array([np.cos(agent.state.p_ang), np.sin(agent.state.p_ang)])
            ang = agent.state.p_ang
            gun_pos = pos + 0.8*agent.size*np.array([np.cos(ang), np.sin(ang)])
            
            agent_info = {
                "Position": [agent.state.p_pos[0],agent.state.p_pos[1]],    # 载具位置
                "IsAlive": agent.alive,     # 存活
                "IsDead": (not agent.alive),    # ~存活
                "GunFiring": (agent.action.shoot and agent.numbullets > 0), # 是否开火（显示扇形攻击区域）
                "GunPosition": [gun_pos[0],gun_pos[1]], #载具上的武器位置，和载具的中心有些许偏差
                "GunAngle": agent.state.p_ang,  # 武器指向 弧度，+x 方向为0度
                "GunAngle(deg)": agent.state.p_ang*180/np.pi,   # 武器指向 角度
                "GunRange": agent.shootRad, # 扇形半径
                "GunKillAngle": agent.shootWin, # 扇形展开夹角 弧度
                "GunKillAngle(deg)": agent.shootWin*180/np.pi, # 扇形展开夹角 角度
                "Color": "red" if str(agent.color.tolist()[1]) == "1.0" else "blue",  # 颜色（代表阵营）
                "WorldTimeStep": self.world.time_step   # 当前episode时间
            }
            summary_list.append(agent_info)
        res = json.dumps({"AgentList":summary_list})
        self.mcv.send(res)


    def render(self):

        from UTIL.tensor_ops import dir2rad
        if not hasattr(self, 'threejs_bridge'):
            from VISUALIZE.mcom import mcom
            self.threejs_bridge = mcom(path='TEMP/v2d_logger/', digit=8, rapid_flush=False, draw_mode='Threejs')
            self.threejs_bridge.v2d_init()
            # self.threejs_bridge.set_style('star')
            # self.threejs_bridge.set_style('grid')
            # self.threejs_bridge.set_style('grid3d')
            self.threejs_bridge.set_style('font', fontPath='/examples/fonts/ttf/FZYTK.TTF', fontLineHeight=1500) # 注意不可以省略参数键值'fontpath=','fontLineHeight=' ！！！
            # self.threejs_bridge.set_style('gray')
            self.threejs_bridge.set_style('skybox6side',    # 设置天空盒子，注意不可以省略参数键值 !!
                posx='/wget/mars_textures/mars_posx.jpg',   
                negx='/wget/mars_textures/mars_negx.jpg',   
                posy='/wget/mars_textures/mars_posy.jpg',
                negy='/wget/mars_textures/mars_negy.jpg',
                posz='/wget/mars_textures/mars_posz.jpg',
                negz='/wget/mars_textures/mars_negz.jpg',
            )
            self.threejs_bridge.其他几何体之旋转缩放和平移('tower', 'BoxGeometry(1,1,1)',   0,0,0,  1,1,8, 0,0,-4) # 长方体
            self.threejs_bridge.其他几何体之旋转缩放和平移('tower2', 'BoxGeometry(1,1,1)',   0,0,0,  0,0,5, 0,0,-4) # 长方体

            self.threejs_bridge.geometry_rotate_scale_translate('box',   0, 0,       0,       3, 2, 1,         0, 0, 0)
            self.threejs_bridge.geometry_rotate_scale_translate('cone',  0, np.pi/2, 0,       1.2, 0.9, 0.9,   1.5,0,0.5) # x -> y -> z
            self.threejs_bridge.terrain_theta=0
            for _ in range(20):
                # 等待加载字体
                self.threejs_bridge.空指令()
                self.threejs_bridge.v2d_show()


        self.threejs_bridge.v2dx('tower|1000|%s|0.15'%('FireBrick'), 3, 3, 1, ro_x=0, ro_y=0, ro_z=0,
            label='坐标(+3,+3)', label_color='White', opacity=1)
        self.threejs_bridge.v2dx('tower|1001|%s|0.15'%('FireBrick'), 3, -3, 1, ro_x=0, ro_y=0, ro_z=0,
            label='坐标(+3,-3)', label_color='White', opacity=1)
        self.threejs_bridge.v2dx('tower|1002|%s|0.15'%('FireBrick'), -3, 3, 1, ro_x=0, ro_y=0, ro_z=0,
            label='坐标(-3,+3)', label_color='White', opacity=1)
        self.threejs_bridge.v2dx('tower|1003|%s|0.15'%('FireBrick'), -3, -3, 1, ro_x=0, ro_y=0, ro_z=0,
            label='坐标(-3,-3)', label_color='White', opacity=1)

        if self.threejs_bridge.terrain_theta != self.world.init_theta:
            self.threejs_bridge.terrain_theta = self.world.init_theta
            self.threejs_bridge.set_env('terrain', theta=self.world.init_theta)

        n_red= len([0 for agent in self.world.agents if agent.alive and agent.attacker])
        n_blue = len([0 for agent in self.world.agents if agent.alive and not agent.attacker])
        who_is_winning = '蓝方(Conc强化学习)领先' if n_blue>n_red else '红方(游戏脚本)领先'
        self.threejs_bridge.v2dx('tower2|1004|Gray|0.2', 0, 3, 1, ro_x=0, ro_y=0, ro_z=0,
            label='蓝方剩余单位(Conc强化学习): %d\n红方剩余单位(游戏脚本): %d\n%s'%(n_blue, n_red, who_is_winning), label_color='Aqua', opacity=0)


        for index, agent in enumerate(self.world.agents):
            x = agent.state.p_pos[0]; y = agent.state.p_pos[1]
            dir_ = dir2rad(agent.state.p_vel)
            color = 'red' if agent.attacker else 'blue'
            base_color = 'LightPink' if agent.attacker else 'CornflowerBlue'
            if not agent.alive:
                color = "#330000" if agent.attacker else "#000033"
                base_color = "#330000" if agent.attacker else "#000033"
            # base_color = base_color if agent.alive else 'black'

            shape3d = 'cone' # if not agent.attacker else 'cone'
            size = 0.025 if agent.alive else 0.01

            self.threejs_bridge.v2dx(
                'cone|%d|%s|%.3f'%(agent.iden, color, size),
                x, y, (agent.terrain-1)*4,
                ro_x=0, ro_y=0, ro_z=agent.state.p_ang,  # Euler Angle y-x-z
                label='', label_color='white', attack_range=0, opacity=1)
            self.threejs_bridge.v2dx(
                'box|%d|%s|%.3f'%(agent.iden+500, base_color, size),
                x, y, (agent.terrain-1)*4-0.025,
                ro_x=0, ro_y=0, ro_z=dir_,  # Euler Angle y-x-z
                label='', label_color='white', attack_range=0, opacity=1)

            if agent.wasHitBy is not None:
                flash_type = 'lightning' if agent.attacker else 'lightning'
                flash_color = 'DeepSkyBlue' if agent.attacker else 'Magenta'
                self.threejs_bridge.flash(flash_type, src=agent.wasHitBy.iden, dst=agent.iden, dur=0.2, size=0.03, color=flash_color)
                agent.wasHitBy = None
                
        self.threejs_bridge.v2d_show()

    # def render_(self):
    #     from config import GlobalConfig as cfg
    #     from UTIL.tensor_ops import dir2rad
    #     if not hasattr(cfg, 'fak_logger'):
    #         from VISUALIZE.mcom import mcom
    #         cfg.fak_logger = mcom(ip='127.0.0.1',
    #                 port=12084,
    #                 path='%s/v2d_logger/'%cfg.logdir,
    #                 digit=16, rapid_flush=True, draw_mode='Native')
    #         cfg.fak_logger.v2d_init()

    #     self.attackers = [agent for agent in self.world.agents if (agent.attacker)]
    #     self.guards = [agent for agent in self.world.agents if (not agent.attacker)]
    #     self.mcv = cfg.fak_logger
    #     # self.mcv.v2d_clear()
    #     for index, guard in enumerate(self.guards):
    #         self.mcv.v2dx('tank|%d|b|0.04'%(index), guard.state.p_pos[0], guard.state.p_pos[1], guard.state.p_ang, vel_dir=dir2rad(guard.state.p_vel), attack_range=guard.terrain*guard.shootRad)
    #         if not guard.alive:
    #             self.mcv.v2dx('tank|%d|k|0.04'%(index), guard.state.p_pos[0], guard.state.p_pos[1], guard.state.p_ang)
    #     for index, attacker in enumerate(self.attackers):
    #         self.mcv.v2dx('tank|%d|r|0.04'%(index+len(self.guards)), attacker.state.p_pos[0], attacker.state.p_pos[1], attacker.state.p_ang, vel_dir=dir2rad(attacker.state.p_vel), attack_range=attacker.terrain*attacker.shootRad)
    #         if not attacker.alive:
    #             self.mcv.v2dx('tank|%d|k|0.04'%(index+len(self.guards)), attacker.state.p_pos[0], attacker.state.p_pos[1], attacker.state.p_ang)
    #     self.mcv.v2d_add_terrain(self.world.init_theta)
    #     self.mcv.v2d_show()
    #     # Matplotlib is ridiculously slow even after optimization! 
    #     # Please change the channel to MATLAB bridge if you want smooth display!
    #     time.sleep(1)
    #     pass


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


