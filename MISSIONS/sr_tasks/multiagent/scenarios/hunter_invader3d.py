import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
import cmath, math, os, time
from UTILS.tensor_ops import dir2rad, dir3d_rad

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
    max_steps_episode = 200
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
    Hunter_MaxSpeed  = Unit(m=12)

    Landmark_Size = Unit(m=6)
    Invader_Kill_Range = Unit(m=2) #

    Invader_Spawn_Times = invader_num*2-invader_num
    invader_spawn_cd = 20

    RewardAsUnity = True
    render = False



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
            self.threejs_bridge.set_style('grid')
            self.threejs_bridge.set_style('gray')
            self.threejs_bridge.use_geometry('monkey')
            # self.threejs_bridge.geometry_rotate_scale_translate('monkey',0, 0,       np.pi/2, 1, 1, 1,         0,0,0)
            self.threejs_bridge.geometry_rotate_scale_translate('box',   0, 0,       0,       0.5, 0.5, 3,         0,0,0)
            self.threejs_bridge.geometry_rotate_scale_translate('ball',  0, 0,      0,       1, 1, 1,         0,0,0)
            self.threejs_bridge.geometry_rotate_scale_translate('cone',  0, np.pi/2, 0,       1.2, 0.9, 0.9,   1.5,0,0.5) # x -> y -> z
        uid = 0
        for agent in self.invaders:
            dir_1, dir_2 = dir3d_rad(agent.state.p_vel)   # # Euler Angle y-x-z

            self.threejs_bridge.v2dx(
                'cone|%d|red|0.05'%uid,
                agent.state.p_pos[0],   # x coordinate
                agent.state.p_pos[1],   # y coordinate
                agent.state.p_pos[2],   # z coordinate
                ro_x=0, ro_y=-dir_2, ro_z=dir_1, # rotation
                label='', 
                label_color='white', 
                opacity=0.99 if agent.live else 0
            )

            uid += 1
        for index, agent in enumerate(self.hunters):
            dir_1, dir_2 = dir3d_rad(agent.state.p_vel)   # # Euler Angle y-x-z
            self.threejs_bridge.v2dx(
                'cone|%d|blue|0.02'%uid,
                agent.state.p_pos[0], 
                agent.state.p_pos[1], 
                agent.state.p_pos[2],
                ro_x=0, ro_y=-dir_2, ro_z=dir_1, # rotation
                label='', label_color='white',
                opacity=1 if agent.live else 0
            )
            dis2invader = self.distance[index, :]
            mindis2invader_id = np.argmin(dis2invader)
            if dis2invader[mindis2invader_id]<self.hunter_affect_range:
                self.threejs_bridge.flash('beam', src=uid, dst=mindis2invader_id, dur=0.5, size=0.03, color='DeepSkyBlue')

            uid += 1
        for index, agent in enumerate(self.landmarks):
            nearest_invader_dis = min(self.distance_landmark[:, index])
            self.threejs_bridge.v2dx(
                'box|%d|green|0.1'%uid,
                agent.state.p_pos[0],
                agent.state.p_pos[1],
                agent.state.p_pos[2],
                ro_x=0, ro_y=0, ro_z=0,  # Euler Angle y-x-z
                label='TH-DIS:%.2f'%nearest_invader_dis, 
                label_color='white' if nearest_invader_dis>1 else 'red', 
                opacity=1
            )
            uid += 1      
        self.threejs_bridge.v2d_show()

    def scenario_step(self, agent, world):
        invaders = self.invaders
        hunters = self.hunters
        # 计算距离矩阵（invader2landmark）
        self.distance, self.distance_landmark = self.get_distance_landmark()
        self.check_invader_tracking_and_slow_down(hunters, invaders)
        # 拦截成功判定
        for invader_index, invader in enumerate(invaders):
            if len(invader.tracked_by) >= self.intercept_hunter_needed:
                invader.live = False
                invader.movable = False
                invader.spawn_cd = self.invader_spawn_cd
                invader.state.p_pos *= 0
                invader.state.p_vel *= 0
                self.indader_left_to_hunt -= 1
                if self.indader_left_to_hunt == 0:
                    self.threat_clear = True
        # 再生
        for invader in invaders:
            if not invader.live:
                if invader.spawn_cd <= 0 and self.invader_spawn_time_left > 0:
                    self.invader_spawn_time_left -= 1
                    self.invader_revise(invader, world)
                else: invader.spawn_cd -= 1

        # 检查landmark是否被摧毁
        if np.min(self.distance_landmark) <= self.Invader_Kill_Range:
            self.hunter_failed = True


    def observation(self, agent, world):
        if agent.iden == 0:
            # by now the agents has already moved according to action
            self.scenario_step(agent, world)    # 第一步更新距离矩阵，更新智能体live的状态
            self.joint_rewards = self.reward_forall(world)      #第二步更新奖励
            if self.show_off: 
                self.render()     # 第三步更新UI

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
        self.step+=1
        # hunter 的奖励有如下几条
        # <3> +10    HUNT_INVDR_SUCCESSFUL_REWARD   拦截成功奖励
        # <4> 20    LANDMARK_DESTORYED_REV_REWARD   invader接触landmark，直接失败

        HUNT_INVDR_SUCCESSFUL_REWARD = 0.1
        HUNT_ALL_INVDR_SUCCESSFUL_REWARD = 1
        LANDMARK_DESTORYED_REV_REWARD = 3

        # MIN_MIN_DIS_REV_REWARD_MAXDIS = Unit(25)
        # MIN_MIN_DIS_REV_REWARD_MINDIS = Unit(5)
        # MIN_MIN_DIS_REV_REWARD_MAXPEL = 0

        # 获取智能体列表30
        agents = world.agents
        hunters = self.hunters
        invaders = self.invaders
        landmars = self.landmarks
        # 初始化奖励列表
        hunter_reward = np.array([0.] * self.num_hunters)
        invader_reward = np.array([0.] * self.num_invaders)
        # 计算距离矩阵（agent2agent）


        min_min_distance = np.min(self.distance_landmark)


        # 看看哪些invader被足够多的hunter围攻，判定拦截成功
        for invader_index, invader in enumerate(invaders):
            #  消灭奖励
            if len(invader.tracked_by) >= self.intercept_hunter_needed:
            #     for hunter_index in invader.tracked_by:
                hunter_reward += HUNT_INVDR_SUCCESSFUL_REWARD

        if self.threat_clear: # 当所有invader都被消灭的时刻获取奖励
            hunter_reward += HUNT_ALL_INVDR_SUCCESSFUL_REWARD
            invader_reward -= HUNT_ALL_INVDR_SUCCESSFUL_REWARD

        # 检查landmark是否被摧毁
        min_distance = np.min(self.distance_landmark)
        if min_distance <= self.Invader_Kill_Range:
            hunter_reward = hunter_reward - LANDMARK_DESTORYED_REV_REWARD
            invader_reward = invader_reward + LANDMARK_DESTORYED_REV_REWARD

        self.reward_sample += hunter_reward[0]
        return invader_reward.tolist() + hunter_reward.tolist()

    def check_invader_tracking_and_slow_down(self, hunters, invaders):
        # detect speed down and hunt result 二重循环，对出现在周围的invader造成减速
        for invader in invaders:
            invader.tracked_by = []
            invader.max_speed = self.Invader_MaxSpeed
        for i, hunter in enumerate(hunters):
            hunter_index = i
            assert hunter.live
            for j, invader in enumerate(invaders):
                distance_i_j = self.distance[i, j]
                # filter far away ones
                if distance_i_j > self.hunter_affect_range:  continue
                invader.max_speed = max(invader.max_speed - self.hunter_speed_pressure, 0)
                invader.tracked_by.append(hunter_index)



    def spawn_position(self, agent, world):
        if not agent.IsInvader:
            # 处理hunter
            # 初始化，随机地分布在一个正方形内
            agent.state.p_pos = np.random.uniform(-self.hunter_spawn_pos_lim, self.hunter_spawn_pos_lim,
                                                  world.dim_p) + self.nest_center_pos
            # 速度，初始化为0
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.live = True
            agent.movable = True
        else:
            # 处理invader
            # spawn direction relative to nest
            self.process_invader_pos(agent)
            # 速度，初始化为0
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.live = True
            agent.movable = True

    def process_invader_pos(self, agent):
        while True:
            theta = np.random.rand() * 2 * np.pi - np.pi
            phi = np.random.rand() * 2 * np.pi - np.pi
            d = self.rand(low=1.0 * self.invader_spawn_limit, high=1.1 * self.invader_spawn_limit)
            agent.state.p_pos = d * np.array([ np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta) ]) + self.nest_center_pos
            x = agent.state.p_pos[0]
            y = agent.state.p_pos[1]
            z = agent.state.p_pos[2]
            if x < self.arena_size and x > -self.arena_size \
                and y < self.arena_size and y > -self.arena_size \
                and z < self.arena_size and z > -self.arena_size:
                break

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
        self.invader_spawn_time_left = self.Invader_Spawn_Times
        self.indader_left_to_hunt = self.Invader_To_Intercept
        if self.show_off:
            print('reset world')

    def spawn_vip_landmark(self, world):
        theta = np.pi/2
        phi = np.random.rand() * 2 * np.pi - np.pi
        phi_dis = np.pi * 2 / self.num_landmarks
        for i, landmark in enumerate(world.landmarks):
            self.landmark_spawn_position(landmark, world, theta, phi + phi_dis * i)

    def is_collision(self, agent1, agent2):  # 检测是否碰到了一起
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def is_dectective(self, agent1, agent2):  # A 和 B 之间的距离是否能相互可见
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # dist_min = agent1.size + agent2.size
        return True if dist < self.distance_dectection else False

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
        condition1 = world.steps >= world.max_steps_episode
        # if self.show_off and condition1:
        #     print('time up reset')
        self.is_success = False if self.hunter_failed else True
        # if self.show_off and self.hunter_failed:
        #     print('landmark destoryed')
        condition2 = self.threat_clear
        if agent.iden==0 and self.show_off and self.threat_clear:
            if self.eval_mode:
                time.sleep(3)
            print('hunt success')
        return condition1 or condition2 or self.hunter_failed

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
        self.reset_world(world)
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