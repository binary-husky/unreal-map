import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
import cmath, math, os, time
from UTILS.tensor_ops import np_normalize_last_dim, dir3d_rad

def Norm(x):  # 求长度
    return np.linalg.norm(x)

def Unit(m):
    return m * 0.05

def ToMeter(x):
    return x / 0.05

def assert_and_break(cond):
    if cond:
        return
    else:
        print("fail!")




class ScenarioConfig(object):
    hunter_num = 15
    invader_num = 5
    num_landmarks = 6
    num_dummy_coordinate = 4

    num_entity = invader_num + num_landmarks + num_dummy_coordinate
    num_subject_in_obs =  (hunter_num + invader_num) + num_landmarks + num_dummy_coordinate   # 观测向量长度

    uid_dictionary = { # need to contain at least 'agent_uid' and 'entity_uid'
        'invader_uid' : range(0,                                        invader_num),
        'agent_uid'  :  range(invader_num,                              invader_num+hunter_num),

        'landmark_uid': range(invader_num+hunter_num,                   invader_num+hunter_num+num_landmarks),
        'dummy_uid':    range(invader_num+hunter_num+num_landmarks,     invader_num+hunter_num+num_landmarks+num_dummy_coordinate),
    }
    uid_dictionary['entity_uid'] = list(uid_dictionary['invader_uid']) + list(uid_dictionary['landmark_uid']) +  list(uid_dictionary['dummy_uid'])

    N_TEAM = 2
    N_AGENT_EACH_TEAM = [invader_num, hunter_num]
    AGENT_ID_EACH_TEAM = [list(range(0,invader_num)), list(range(invader_num,  invader_num+hunter_num))]
    TEAM_NAMES = ['script-team', 'rl-team']
    obs_vec_length = 8
    obs_vec_dictionary = {
        'pos' : (0,1,2),
        'vel' : (3,4,5),
        'alive': (6),
        'other': (7),
    }

    discrete_action = True
    max_steps_episode = 140
    arena_size = Unit(m=140)
    num_MPE_agent = hunter_num + invader_num
    nest_center_pos = np.array([Unit(m=0), Unit(m=0), Unit(m=0)])

    hunter_spawn_pos_lim = Unit(m=25)
    landmark_spawn_limit = Unit(m=20)

    invader_spawn_limit = Unit(m=80)
    distance_dectection = Unit(m=25)
    intercept_hunter_needed = 3

    hunter_affect_range   = Unit(m=5.999)
    hunter_speed_pressure = Unit(m=5.999)

    Invader_Size     = Unit(m=2)
    Invader_Accel    = Unit(m=400)
    Invader_MaxSpeed = Unit(m=12)   #  12 * 0.05=0.6

    Hunter_Size      = Unit(m=1.5)
    Hunter_Accel     = Unit(m=400)
    Hunter_MaxSpeed  = Unit(m=12) # 12

    Landmark_Size = Unit(m=6)
    Invader_Kill_Range = Unit(m=5.999) #

    Invader_Spawn_Times = invader_num*2-invader_num
    invader_spawn_cd = 20

    RewardAsUnity = True
    render = False
    
    extreme_sparse = True


class Scenario(BaseScenario):
    def __init__(self, process_id=-1):
        self.invader_spawn_cd = ScenarioConfig.invader_spawn_cd
        self.num_agents = ScenarioConfig.num_MPE_agent
        self.arena_size = ScenarioConfig.arena_size
        self.discrete_action = ScenarioConfig.discrete_action
        self.cam_range = ScenarioConfig.arena_size * 1.2
        self.hunter_spawn_pos_lim = ScenarioConfig.hunter_spawn_pos_lim
        self.nest_center_pos = ScenarioConfig.nest_center_pos
        self.invader_spawn_limit = ScenarioConfig.invader_spawn_limit
        self.distance_dectection = ScenarioConfig.distance_dectection
        self.Invader_MaxSpeed = ScenarioConfig.Invader_MaxSpeed
        self.hunter_affect_range = ScenarioConfig.hunter_affect_range
        self.hunter_speed_pressure = ScenarioConfig.hunter_speed_pressure
        self.intercept_hunter_needed = ScenarioConfig.intercept_hunter_needed
        self.num_subject_in_obs = ScenarioConfig.num_subject_in_obs
        self.Invader_Kill_Range = ScenarioConfig.Invader_Kill_Range
        self.Invader_Spawn_Times = ScenarioConfig.Invader_Spawn_Times
        self.obs_vec_length = ScenarioConfig.obs_vec_length
        self.invader_num = ScenarioConfig.invader_num
        self.Invader_To_Intercept = self.invader_num + self.Invader_Spawn_Times
        # self.thread_index = thread_index  # thread_index = 0 是主进程
        self.caught = 0
        self.rew_other = 0
        self.manual_render = None
        self.eval_mode = False
        self.show_off = False if process_id != 0 else ScenarioConfig.render


    def render(self):
        if not hasattr(self, 'threejs_bridge'):
            from VISUALIZE.mcom import mcom
            self.threejs_bridge = mcom(ip='127.0.0.1', port=12084, path='RECYCLE/v2d_logger/', digit=8, rapid_flush=False, draw_mode='Threejs')
            self.threejs_bridge.v2d_init()
            # self.threejs_bridge.set_style('star')
            # self.threejs_bridge.set_style('grid')
            self.threejs_bridge.set_style('gray')
            # self.threejs_bridge.geometry_rotate_scale_translate('monkey',0, 0,       np.pi/2, 1, 1, 1,         0,0,0)
            self.threejs_bridge.geometry_rotate_scale_translate('box',   0, 0,       0,       0.5, 0.5, 1.5,         0,0,0)
            self.threejs_bridge.geometry_rotate_scale_translate('ball',  0, 0,      0,        1, 1, 1,         0,0,0)
            self.threejs_bridge.geometry_rotate_scale_translate('cone',  0, np.pi/2, 0,       1.2, 0.9, 0.9,   1.5,0,0.5) # x -> y -> z
            self.threejs_bridge.其他几何体之旋转缩放和平移('oct', 'OctahedronGeometry(1,0)', 0,0,0,  1,1,1, 0,0,0)   # 八面体

            self.threejs_bridge.agent_alive_pos = {}
            self.threejs_bridge.agent_alive_time = {}


        for index, agent in enumerate(self.invaders + self.hunters):
            dir_1, dir_2 = dir3d_rad(agent.state.p_vel)   # # Euler Angle y-x-z
            x = agent.state.p_pos[0]; y=agent.state.p_pos[1]; z=agent.state.p_pos[2]
            if agent.live:
                self.threejs_bridge.agent_alive_pos[index] = (x,y,z)
                if agent.IsInvader: 
                    color = 'Red'
                    size = 0.04
                if not agent.IsInvader: 
                    color = 'Blue'
                    size = 0.01
                opacity = 0.99
                if index not in self.threejs_bridge.agent_alive_time: self.threejs_bridge.agent_alive_time[index] = 0
                self.threejs_bridge.agent_alive_time[index] += 1
            else:
                color = 'black'
                opacity = 0
                self.threejs_bridge.agent_alive_time[index] = 0
        
            self.threejs_bridge.v2dx(
                'cone|%d|%s|%.2f'%(index, color, size),
                self.threejs_bridge.agent_alive_pos[index][0], 
                self.threejs_bridge.agent_alive_pos[index][1], 
                self.threejs_bridge.agent_alive_pos[index][2], 
                ro_x=0, ro_y=-dir_2, ro_z=dir_1, # rotation
                label='', label_color='white',
                opacity=opacity if agent.live else 0,
                track_n_frame = max(min(self.threejs_bridge.agent_alive_time[index]-1, 10),0)
            )

        for index, agent in enumerate(self.hunters):
            dir_1, dir_2 = dir3d_rad(agent.state.p_vel)   # Euler Angle y-x-z
            dis2invader = self.distance[index, :]
            mindis2invader_id = np.argmin(dis2invader)
            if dis2invader[mindis2invader_id]<self.hunter_affect_range:
                self.threejs_bridge.flash('beam', src=index+len(self.invaders), dst=mindis2invader_id, dur=0.5, size=0.03, color='DeepSkyBlue')


        for index, agent in enumerate(self.landmarks):
            nearest_invader_dis = min(self.distance_landmark[:, index])
            self.threejs_bridge.v2dx(
                'oct|%d|green|0.2'%(index+999),
                agent.state.p_pos[0],
                agent.state.p_pos[1],
                agent.state.p_pos[2],
                ro_x=0, ro_y=0, ro_z=0,  # Euler Angle y-x-z
                label='invader@ %.2f'%nearest_invader_dis, 
                label_color='Black' if nearest_invader_dis>1 else 'red', 
                opacity=0.5
            )

        self.threejs_bridge.v2dx(
            'oct|%d|green|0.45'%(1999),
            0,
            0,
            1.6,
            ro_x=0, ro_y=0, ro_z=0,  # Euler Angle y-x-z
            label='Time %d/%d'%(self.step, ScenarioConfig.max_steps_episode), 
            label_color='white', 
            opacity=0
        )
        self.threejs_bridge.v2d_show()
        if (self.step)==0: 
            self.threejs_bridge.set_env('clear_track')
            self.threejs_bridge.set_env('clear_flash')

    def scenario_step(self, agent, world):
        invaders = self.invaders
        hunters = self.hunters
        # 计算距离矩阵（invader2landmark）
        self.distance, self.distance_landmark = self.get_distance_landmark()

        # 二重循环，对出现在周围的invader造成减速
        self.threat_clear = True     # 检查invader是否被驱除出
        for invader in invaders:
            invader.tracked_by = []
            if np.linalg.norm(invader.state.p_pos) < ScenarioConfig.invader_spawn_limit*1.11:
                self.threat_clear = False
            
        for i, hunter in enumerate(hunters):
            hunter_index = i
            assert hunter.live
            for j, invader in enumerate(invaders):
                distance_i_j = self.distance[i, j]
                if distance_i_j > self.hunter_affect_range:  continue
                invader.tracked_by.append(hunter_index)

        # push invaders to opposite direction
        for invader_index, invader in enumerate(invaders):
            if invader.state.previous_pos is None: continue # special situations
            if not hasattr(invader, 'force'): continue # special situations

            invader.state.p_vel = invader.state.previous_vel * (1 - world.damping)  # read vel
            if (invader.force is not None): # use force to update vel
                invader.force_real = invader.force +  (-invader.force) * len(invader.tracked_by) * 0.55
                invader.state.p_vel += (invader.force_real / invader.mass) * world.dt

            # limit max speed
            if (invader.max_speed is not None):
                speed = np.linalg.norm(invader.state.p_vel)
                if speed > invader.max_speed:
                    invader.state.p_vel = invader.state.p_vel / speed * invader.max_speed

            # update position
            invader.state.p_pos = invader.state.previous_pos + invader.state.p_vel * world.dt


        # 检查landmark是否被摧毁 self.distance_landmark [self.num_invaders, self.num_landmarks]
        if np.min(self.distance_landmark) <= self.Invader_Kill_Range:
            self.hunter_failed = True


    def observation(self, agent, world):
        if agent.iden == 0:
            # by now the agents has already moved according to action
            self.scenario_step(agent, world)    # 第一步更新距离矩阵，更新智能体live的状态
            self.joint_rewards = self.reward_forall(world)      #第二步更新奖励
            if self.show_off: self.render()     # 第三步更新UI
            self.step += 1

        self.obs_dimension = self.obs_vec_length*self.num_subject_in_obs
        self.obs_pointer = 0
        self.obs = np.zeros(shape=(self.obs_dimension,))

        self.load_obs(
            np.concatenate(
                [
                np.concatenate(
                    (entity.state.p_pos, entity.state.p_vel, [entity.live, agent.iden])
                )
                for entity in world.agents]
            )
        )
        self.load_obs(
            np.concatenate(
                [
                np.concatenate(
                    (entity.state.p_pos, entity.state.p_vel, [1, agent.iden])
                ) 
                for entity in world.landmarks]
            )
        )
        # dummy coordinate
        self.load_obs(
            np.concatenate(
                [
                    np.concatenate( ([0,  0, 0],   [0,0,0],  [world.steps, agent.iden]) ),
                    np.concatenate( ([1,  0, 0],   [0,0,0],  [world.steps, agent.iden]) ),
                    np.concatenate( ([0,  1, 0],   [0,0,0],  [world.steps, agent.iden]) ),
                    np.concatenate( ([0,  0, 1],   [0,0,0],  [world.steps, agent.iden]) ),
                ]
            )
        )

        return self.obs.copy()

    def reward_forall(self, world):
        # 初始化奖励列表
        hunter_reward = np.array([0.] * self.num_hunters)
        invader_reward = np.array([0.] * self.num_invaders)
        win_cond1 = world.steps >= world.max_steps_episode
        win_cond2 = self.threat_clear
        win_cond = win_cond1 or win_cond2
        loss_cond = self.hunter_failed
        if loss_cond: win_cond = False  # lose condition has higher priority

        if ScenarioConfig.extreme_sparse:
            # 极度稀疏奖励
            if win_cond:
                hunter_reward += 1
                invader_reward -= 1

            if loss_cond:
                hunter_reward -= 1
                invader_reward += 1
        else:
            # 一般型稀疏奖励
            HUNT_INVDR_SUCCESSFUL_REWARD = 0.05
            REWARD_WHEN_TRACKED_BY_N = 3
            WIN_REWARD = 1

            for invader_index, invader in enumerate(self.invaders):
                if len(invader.tracked_by) >= REWARD_WHEN_TRACKED_BY_N and (not invader.intercepted):
                    invader.intercepted = True
                    hunter_reward += HUNT_INVDR_SUCCESSFUL_REWARD
                    invader_reward -= HUNT_INVDR_SUCCESSFUL_REWARD

            if win_cond:
                hunter_reward += WIN_REWARD
                invader_reward -= WIN_REWARD
            if loss_cond:
                hunter_reward -= WIN_REWARD
                invader_reward += WIN_REWARD
            
        self.reward_sample += hunter_reward[0]
        return invader_reward.tolist() + hunter_reward.tolist()




    def spawn_position(self, agent, world):
        if not agent.IsInvader:
            # 处理hunter
            # 初始化，随机地分布在一个正方形内
            agent.state.p_pos = np.random.uniform(-self.hunter_spawn_pos_lim, self.hunter_spawn_pos_lim,
                                                  world.dim_p) + self.nest_center_pos
            agent.state.previous_pos = agent.state.p_pos.copy()
            # 速度，初始化为0
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.previous_vel = agent.state.p_vel.copy()
            agent.state.c = np.zeros(world.dim_c)
            agent.live = True
            agent.movable = True
            agent.intercepted = False

        else:
            # 处理invader
            # spawn direction relative to nest
            self.process_invader_pos(agent)
            # 速度，初始化为0
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.previous_vel = agent.state.p_vel.copy()
            agent.state.c = np.zeros(world.dim_c)
            agent.live = True
            agent.movable = True
            agent.intercepted = False

    def process_invader_pos(self, agent):
        while True:
            theta = (np.random.rand() * 2 * np.pi - np.pi)*0.35
            phi = (np.random.rand() * 2 * np.pi - np.pi)
            d = self.rand(low=1.0 * self.invader_spawn_limit, high=1.1 * self.invader_spawn_limit)
            agent.state.p_pos = d * np.array([ np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), np.sin(theta) ]) + self.nest_center_pos
            agent.state.previous_pos = agent.state.p_pos.copy()
            x = agent.state.p_pos[0]
            y = agent.state.p_pos[1]
            z = agent.state.p_pos[2]
            if x < self.arena_size and x > -self.arena_size \
                and y < self.arena_size and y > -self.arena_size \
                and z < self.arena_size and z > -self.arena_size:
                break
            else:
                assert False

    def invader_revise(self, agent, world):
        self.spawn_position(agent, world)
        agent.live = True
        agent.movable = True
        agent.spawn_cd = -1


    def reset_world(self, world):
        self.step = 0
        for agent in world.agents:
            self.spawn_position(agent, world)
        self.spawn_vip_landmark(world)
        world.steps = 0
        self.rew_other = 0
        self.hunter_failed = False
        self.threat_clear = False
        self.reward_sample = 0
        # self.invader_spawn_time_left = self.Invader_Spawn_Times
        # self.indader_left_to_hunt = self.Invader_To_Intercept
        if self.show_off:
            print('reset world')

    def spawn_vip_landmark(self, world):
        theta = np.pi/2
        phi = np.random.rand() * 2 * np.pi - np.pi
        phi_dis = np.pi * 2 / self.num_landmarks
        for i, landmark in enumerate(world.landmarks):
            self.landmark_spawn_position(landmark, world, theta, phi + phi_dis * i)

    # 按顺序 return all agents that are not invaders
    def hunters(self, world):
        return [agent for agent in world.agents if not agent.IsInvader]

    # 按顺序 return all adversarial agents
    def invaders(self, world):
        return [agent for agent in world.agents if agent.IsInvader]

    def reward(self, agent, world):
        assert self.joint_rewards is not None
        reward = self.joint_rewards[agent.iden]
        if agent.iden == self.num_agents:
            self.joint_rewards = None
        return reward

    def get_distance_landmark(self):
        hunters = self.hunters
        invaders = self.invaders
        landmarks = self.landmarks
        distance = np.zeros(shape=(self.num_hunters, self.num_invaders), dtype=np.float32)
        distance_landmark = np.zeros(shape=(self.num_invaders, self.num_landmarks), dtype=np.float32)
        for b, B in enumerate(invaders):
            for a, A in enumerate(hunters):
                distance[a, b] = Norm(A.state.p_pos - B.state.p_pos) if B.live else np.inf
            for c, C in enumerate(landmarks):
                distance_landmark[b, c] = Norm(B.state.p_pos - C.state.p_pos) if B.live else np.inf
        return distance, distance_landmark

    def done(self, agent, world):
        win_cond1 = world.steps >= world.max_steps_episode
        win_cond2 = self.threat_clear
        win_cond = win_cond1 or win_cond2
        self.is_success = win_cond

        loss_cond = self.hunter_failed
        if loss_cond: win_cond = False  # lose condition has higher priority


        done = win_cond or loss_cond
        if done:  assert loss_cond != win_cond

        if agent.iden==0 and self.show_off and win_cond:
            print('hunt success')

        return done

    def load_obs(self, fragment):
        L = len(fragment) if isinstance(fragment, np.ndarray) else 1
        # assert self.obs_pointer + L <= self.obs_dimension
        self.obs[self.obs_pointer:self.obs_pointer + L] = fragment
        # print('[%d ~ %d] filled / Total Length %d / total [0 ~ %d]'%(self.obs_pointer, self.obs_pointer + L -1, self.obs_pointer + L, self.obs_dimension-1))
        self.obs_pointer = self.obs_pointer + L

    def check_obs(self):
        assert self.obs_pointer == self.obs_dimension


    def info(self, agent, world):
        return {'hunter_failed': self.hunter_failed, 'world_steps': world.steps, 'is_success': self.is_success}


    def make_world(self):
        self.num_good_agents = ScenarioConfig.hunter_num
        self.num_hunters = ScenarioConfig.hunter_num
        self.num_adversaries = ScenarioConfig.invader_num
        self.num_invaders = ScenarioConfig.invader_num
        self.num_landmarks = ScenarioConfig.num_landmarks
        world = World() # set any world properties first
        world.dim_c = 3
        world.dim_p = 3
        num_agents = self.num_agents
        # add agents, 包括 hunter 和 invader
        world.agents = [Agent(iden=i) for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False  # no collide any more
            agent.silent = True
            # 前面的 num_adversaries 是 Invader，剩下的是 Hunter， Invader is adversary
            if i < self.num_adversaries:
                agent.IsInvader = True
                agent.adversary = True
                agent.id_in_team = i
                agent.require_mannual_control = True
            else:
                agent.IsInvader = False
                agent.adversary = False
                agent.id_in_team = i - self.num_invaders
            agent.size = ScenarioConfig.Invader_Size if agent.IsInvader else ScenarioConfig.Hunter_Size  # size 中的数值是半径
            agent.accel = ScenarioConfig.Invader_Accel if agent.IsInvader else ScenarioConfig.Hunter_Accel
            agent.max_speed = self.Invader_MaxSpeed if agent.IsInvader else ScenarioConfig.Hunter_MaxSpeed
            agent.caught = False
            # agent.life = 1
            agent.live = True
            agent.movable = True
            agent.live_adv = 1
            agent.initial_mass = 14
        # add landmarks
        world.landmarks = [Landmark() for i in range(self.num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = ScenarioConfig.Landmark_Size
            landmark.boundary = False
        # make initial conditions
        # self.reset_world(world)
        world.max_steps_episode = ScenarioConfig.max_steps_episode
        self.hunters = [agent for agent in world.agents if not agent.IsInvader]
        self.invaders = [agent for agent in world.agents if agent.IsInvader]
        self.landmarks = world.landmarks
        return world

    @staticmethod
    def rand(low, high):
        return np.random.rand() * (high - low) + low

    def landmark_spawn_position(self, landmark, world, theta=45 * np.pi / 180, phi = 45 * np.pi / 180):
        d = ScenarioConfig.landmark_spawn_limit
        offset = d * np.array([ np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta) ])
        # offset = d * np.array([  np.cos(theta), np.sin(theta)   ])
        landmark.state.p_pos = self.nest_center_pos + offset
        landmark.state.p_vel = np.zeros(world.dim_p)