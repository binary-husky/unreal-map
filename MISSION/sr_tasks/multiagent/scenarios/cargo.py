import numpy as np
from multiagent.core import World, Agent
from multiagent.scenario import BaseScenario
import cmath, math
try:
    from numba import njit
except:
    from UTIL.tensor_ops import dummy_decorator as njit

def Norm(x):  # 求长度
    return np.linalg.norm(x)


def assert_and_break(cond):
    if cond:
        return
    else:
        print("fail!")


def convert_to_pole2D(vec):
    cn = complex(vec[0], vec[1])
    Range, rad_angle = cmath.polar(cn)
    angle = math.degrees(rad_angle)
    # print(Range, angle)    # (-180,+180]
    return Range, angle


class ScenarioConfig(object):  
    '''
        ScenarioConfig: This config class will be 'injected' with new settings from JSONC.
        (E.g., override configs with ```python main.py --cfg example.jsonc```)
        (As the name indicated, ChainVars will change WITH vars it 'chained_with' during config injection)
        (please see UTIL.config_args to find out how this advanced trick works out.)
    '''
    discrete_action = True
    MaxEpisodeStep = 150

    reach_distance = 0.07

    n_worker = 50
    weight_percent = 0.85 # 50个单位的智能体运送 50*80%=40单位的货物
    n_cargo = 4
    # n_destination = 2
    acc = 20

    # ~!
    N_TEAM = 1
    N_AGENT_EACH_TEAM = [n_worker, ]
    AGENT_ID_EACH_TEAM = [range(0, n_worker), ]
    TEAM_NAMES = ['ALGORITHM.hmp_native.foundation->ReinforceAlgorithmFoundation', ]

    num_agent = n_worker
    num_entity = n_cargo * 2

    num_object = n_worker + n_cargo + n_cargo
    show_game = False

    uid_dictionary = {
        'agent_uid': range(0, n_worker),        # (self.n_worker + 2 * self.n_cargo)
        'entity_uid': range(n_worker, n_worker + n_cargo * 2)   # (self.n_worker + 2 * self.n_cargo)
    }
    assert len(uid_dictionary['entity_uid']) == num_entity
    assert len(uid_dictionary['agent_uid']) == num_agent
    obs_vec_length = 12
    obs_vec_dictionary = {
        'pos': (0, 1, 2),
        'vel': (3, 4, 5),
        'mass': (6),
        'other': (7,8,9,10,11),  # self-weight, // dragging-who, dispose-for-who, target-weight, weight-rem, env_step
    }
    ObsAsUnity = True
    RewardAsUnity = True
    render = False
    
class Scenario(BaseScenario):
    def __init__(self, process_id=-1):
        self.n_worker = ScenarioConfig.n_worker
        self.n_agent = sum(ScenarioConfig.N_AGENT_EACH_TEAM)
        self.n_cargo = ScenarioConfig.n_cargo
        self.acc = ScenarioConfig.acc
        self.obs_vec_length = ScenarioConfig.obs_vec_length
        self.reach_distance = ScenarioConfig.reach_distance
        self.weight_percent = ScenarioConfig.weight_percent
        self.process_id = process_id
        self.show_off = False 
        if ScenarioConfig.render and process_id == 0:
            self.show_off = True
        self.cargo_previous = None

    def render(self, world):
        if not hasattr(self, 'threejs_bridge'):
            from VISUALIZE.mcom import mcom
            self.threejs_bridge = mcom(path='TEMP/v2d_logger/', digit=8, rapid_flush=False, draw_mode='Threejs')
            self.threejs_bridge.v2d_init()
            # self.threejs_bridge.set_style('star')
            # self.threejs_bridge.set_style('grid')
            # self.threejs_bridge.set_style('gray')
            self.threejs_bridge.set_style('background', color='SkyBlue')
            # self.threejs_bridge.geometry_rotate_scale_translate('monkey',0, 0,       np.pi/2, 1, 1, 1,         0,0,0)
            self.threejs_bridge.geometry_rotate_scale_translate('box',   0, 0,       0,      1, 1, 1,         0,0,0)
            self.threejs_bridge.geometry_rotate_scale_translate('ball',  0, 0,      0,        1, 1, 1,         0,0,0)
            self.threejs_bridge.geometry_rotate_scale_translate('cone',  0, np.pi/2, 0,       1.2, 0.9, 0.9,   -0.2,0,0) # x -> y -> z
            self.threejs_bridge.其他几何体之旋转缩放和平移('oct', 'OctahedronGeometry(1,0)', 0,0,0,  1,1,1, 0,0,0)   # 八面体
            self.threejs_bridge.其他几何体之旋转缩放和平移('dst', 'TorusGeometry( 1, 0.1, 16, 100 )', np.pi/2,0,0,  1,1,1, 0,0,0)   # 甜甜圈
            self.threejs_bridge.advanced_geometry_material('box', 
                map='/wget/bz.jpg',
            )  
            self.threejs_bridge.advanced_geometry_material('dst', 
                map='/wget/wen.jpg',
            )  
            self.threejs_bridge.agent_alive_pos = {}
            self.threejs_bridge.agent_alive_time = {}


        for index, worker in enumerate(self.workers):
            if worker.dragging < 0:
                self.threejs_bridge.v2dx(
                    'oct|%d|%s|%.2f'%(index, 'Red', 0.01),
                    worker.state.p_pos[0], worker.state.p_pos[1], 
                    0,  ro_x=0, ro_y=-0, ro_z=0,
                    label='', label_color='white',
                    opacity=0.99,
                    track_n_frame=5,
                )
            else:
                # m/n
                c = worker.dragging
                n = len(self.cargo_dragged_by[c])
                m = self.cargo_dragged_by[c].index(index)
                self.threejs_bridge.v2dx(
                    'oct|%d|%s|%.2f'%(index, 'Red', 0.01),
                    worker.state.p_pos[0] + np.cos(m / n * 2 * np.pi) * 0.1, 
                    worker.state.p_pos[1] + np.sin(m / n * 2 * np.pi) * 0.1, 
                    0,  ro_x=0, ro_y=-0, ro_z=0,
                    label='', label_color='white',
                    opacity=0.99,
                    track_n_frame=5,
                )

        for index, cargo_pos in enumerate(self.cargo):
            if self.cargo_hot[index]:
                color = 'Yellow'
            else:
                color = 'Black'

            self.threejs_bridge.v2dx(
                'box|%d|%s|%.2f'%(index+1000, color, 0.1*self.cargo_weight[index]/10),
                cargo_pos[0], cargo_pos[1], 
                0,  ro_x=0, ro_y=-0, ro_z=0,
                label='BioHazard #%d\ndrag %d:weight %d'%(index,len(self.cargo_dragged_by[index]), self.cargo_weight[index]), 
                label_color='Crimson',
                label_size=self.cargo_weight[index]*0.01,
                opacity=0.5,
                renderOrder=64,
            )

            # self.threejs_bridge.v2dx(
            #     'box|%d|%s|%.2f'%(index+5000, color, 0.05),
            #     cargo_pos[0], cargo_pos[1]-0.2, 
            #     0.2,  ro_x=0, ro_y=-0, ro_z=0,
            #     label='drag %d:weight %d'%(), 
            #     label_color='Magenta' if (len(self.cargo_dragged_by[index]) != self.cargo_weight[index]) else 'Green',
            #     opacity=0,
            #     renderOrder=64,
            # )

        for index, drop_off_pos in enumerate(self.cargo_drop_off):
            if self.cargo_hot[index]:
                color = 'LimeGreen'
            else:
                color = 'Black'
            self.threejs_bridge.v2dx(
                'dst|%d|%s|%.2f'%(index+1500, color, 0.05),
                drop_off_pos[0], drop_off_pos[1], 
                0,  ro_x=0, ro_y=-0, ro_z=0,
                label='dst #%d'%index, label_color='Chocolate',
                opacity=0.99,
                renderOrder=0,
            )

            if self.cargo_hot[index]:
                x1=drop_off_pos[0]
                y1=drop_off_pos[1]
                x2=self.cargo[index][0]
                y2=self.cargo[index][1]
            else:
                x1=0
                y1=0
                x2=0
                y2=0
            self.threejs_bridge.line3d(
                'fat|%d|%s|%.3f'%(index+2000, 'Black', 0.005),
                x_arr=np.array([x1, x2]),
                y_arr=np.array([y1, y2]),
                z_arr=np.array([0, 0]),
                dashScale=20,   # to make dash denser, Increase this instead of decrease !!
                dashSize=1,
                gapSize=5,
                tension=0,
                opacity=0.5,
            )
        self.threejs_bridge.v2dx(
            'ball|%d|%s|%.2f'%(2000, color, 0.2),
            0, 2, 
            1,  ro_x=0, ro_y=-0, ro_z=0,
            label='Exposure ETA %d'%(ScenarioConfig.MaxEpisodeStep-world.steps), 
            label_color='Blue' if (ScenarioConfig.MaxEpisodeStep-world.steps)>25 else 'Red',
            opacity=0,
        )
        self.threejs_bridge.v2d_show()

        # remain_weight = [0,0,0,0,0]
        # for cargo_pos, weight, index, dragged_by_L in zip(self.cargo, self.cargo_weight, range(self.n_cargo), self.cargo_dragged_by):
        #     remain_weight[index] = weight - len(dragged_by_L)
        # self.mcv.xlabel(('step: %d,reward: %.2f,' % (self.step, self.reward_sample))+str(remain_weight))
        # self.mcv.drawnow()
        return

    def observation(self, world):
        # by now the agents has already moved according to action
        # self.scenario_step(agent, world)  # 第一步更新距离矩阵，更新智能体live的状态
        self.joint_rewards = self.reward_forall(world)  # 第二步更新奖励
        if self.show_off: self.render(world)  # 第三步更新UI

        self.obs_dimension = self.obs_vec_length * (self.n_worker + 2 * self.n_cargo)
        self.obs_pointer = 0  # this is important for function load_obs
        self.obs = np.zeros(shape=(self.obs_dimension,))
        # self.obs_vec_length * (self.n_worker)
        self.load_obs(
            np.concatenate(
                [
                    np.concatenate(
                        (entity.state.p_pos,[0], # pos3d
                         entity.state.p_vel,[0], # vel3d
                         [1, entity.dragging, -1, -1, -1, world.steps/ScenarioConfig.MaxEpisodeStep])      
                         # self-weight, // dragging-who, dispose-for-who, target-weight, weight-rem, env_step
                    )
                    for entity in world.agents]
            )
        )
        # self.obs_vec_length * (self.n_cargo)
        self.load_obs(
            np.concatenate(
                [
                    np.concatenate(
                        (cargo_pos,[0],
                         [0, 0, 0],    # 空着也是空着，加点其他的信息
                         [weight, -1, -1, -1, weight-len(dragged_by_L), world.steps/ScenarioConfig.MaxEpisodeStep]
                         # self-weight, // dragging-who, dispose-for-who, target-weight, weight-rem, env_step
                        )
                    )
                    for cargo_pos, weight, index, dragged_by_L in zip(self.cargo, self.cargo_weight, range(self.n_cargo), self.cargo_dragged_by)]
            )
        )
        # self.obs_vec_length * (self.n_cargo)
        self.load_obs(
            np.concatenate(
                [
                    np.concatenate(
                        (drop_off_pos,[0],
                         [0, 0, 0],
                         [-1, -1, corrisponding_cargo, -1, -1, world.steps/ScenarioConfig.MaxEpisodeStep])
                         # self-weight, // dragging-who, dispose-for-who, target-weight, weight-rem, env_step
                    )
                    for corrisponding_cargo, drop_off_pos in enumerate(self.cargo_drop_off)]
            )
        )
        assert self.obs_pointer == self.obs_dimension
        return self.obs.copy()

    def load_obs(self, fragment):
        L = len(fragment) if isinstance(fragment, np.ndarray) else 1
        # assert self.obs_pointer + L <= self.obs_dimension
        self.obs[self.obs_pointer:self.obs_pointer + L] = fragment
        # print('[%d ~ %d] filled / Total Length %d / total [0 ~ %d]'%(self.obs_pointer, self.obs_pointer + L -1, self.obs_pointer + L, self.obs_dimension-1))
        self.obs_pointer = self.obs_pointer + L


    # def scenario_step(self, agent, world):
    #     pass


    def update_matrix(self):
        return self.update_worker_cargo_matrix(self.n_worker, self.n_cargo, self.worker_pos_arr, self.cargo)
    

    @staticmethod
    @njit
    def update_worker_cargo_matrix(n_worker, n_cargo, workers, cargo):
        worker_cargo_dis = np.zeros((n_worker, n_cargo),np.float64)
        for i_worker, worker_obj in enumerate(workers):
            for j_cargo, cargo_pos in enumerate(cargo):
                worker_cargo_dis[i_worker, j_cargo] = np.linalg.norm(cargo_pos - worker_obj)
        return worker_cargo_dis

    
    def reward_forall(self, world):
        self.step += 1
        # worker 奖励有如下2条
        # <1> CARGO_START_MOVING_REWARD = 0.1
        # <2> CARGO_REACH_DST_REWARD = 1
        worker_reward = np.zeros(self.n_worker)

        CARGO_START_MOVING_REWARD = 0.1
        CARGO_REACH_DST_REWARD = 0.1
        CARGO_ALL_DONE = 1
        assert CARGO_ALL_DONE > ScenarioConfig.n_cargo * (CARGO_START_MOVING_REWARD+CARGO_REACH_DST_REWARD)
        # 获取智能体列表30
        cargo = self.cargo
        cargo_dst = self.cargo_drop_off
        assert cargo is not None
        assert cargo_dst is not None
        self.worker_pos_arr = np.array([w.state.p_pos for w in self.workers])



        shift = self.worker_pos_arr - self.worker_previous_pos \
            if self.worker_previous_pos is not None else np.array([0, 0])

        # 处理cargo运动
        for c in range(self.n_cargo):
            # sum up the agent force(shift) direction, update cargo position
            push_by_n_worker = len(self.cargo_dragged_by[c])
            if push_by_n_worker >= self.cargo_weight[c] and self.cargo_hot[c]:
                holding_agents = [True if w in self.cargo_dragged_by[c] else False for w in range(self.n_worker)]
                shift_ = shift[holding_agents]
                shift_ = shift_.mean(axis=0)
                shift_ = shift_ / (Norm(shift_) + 1e-6) * 0.5 / 10
                if self.cargo_moving[c]:
                    self.cargo[c] += shift_
                if not self.cargo_moving[c]:  # this is the moment that cargo start moving
                    self.cargo_moving[c] = True
                    if self.cargo_hot[c]: 
                        worker_reward += CARGO_START_MOVING_REWARD
                        self.n_cargo_lifted += 1
                        if self.n_cargo_lifted >= self.n_cargo:
                            if self.show_off: print('all cargo lifted!')

            # hold old dragging workers at the cargo position
            for w in self.cargo_dragged_by[c]:
                self.worker_pos_arr[w] = self.cargo[c].copy()   # set pos

        # if any cargo reach its destination
        cargo_distance = np.linalg.norm((cargo - cargo_dst), axis=-1)
        for c, cargo_reached in enumerate(cargo_distance < self.reach_distance):
            if cargo_reached:
                if self.cargo_hot[c]:
                    self.cargo_hot[c] = False  # 每个货物只能带来一次奖励
                    worker_reward += CARGO_REACH_DST_REWARD
                    # self.cargo_weight[c] = self.n_worker+1 # set weight so large that cannot be dragged again
                    if not any(self.cargo_hot):
                        # deliver all done
                        if self.show_off: print('deliver all done')
                        worker_reward += CARGO_ALL_DONE
                        self.cargo_all_delivered = True
                else:
                    pass


                for w in self.cargo_dragged_by[c]:
                    self.workers[w].state.p_vel = np.array([0, 0])
                    self.workers[w].movable = False
                    self.workers[w].live = False

                # self.cargo_dragged_by[c].clear()


        self.worker_cargo_dis = self.update_matrix()
        contact_mat = self.worker_cargo_dis < self.reach_distance
        contact_arr = contact_mat.sum(1)
        cargo_contact_arr = contact_mat.sum(0)

        # check if any worker starts a new cargo
        for w in range(self.n_worker):
            if contact_arr[w] <= 0:
                c = self.workers[w].dragging    # the cargo has reached dst
                if c >= 0:
                    assert False
                    # assert w not in self.cargo_dragged_by[c]    # cargo_dragged_by has been clear()
                    self.workers[w].dragging = -1
                continue    # cargo 抵达目的地
            # otherwise if contact_arr[w] > 0:
            c = int(np.argmin(self.worker_cargo_dis[w]))
            if self.workers[w].dragging != c and self.workers[w].dragging != -1:  # 已经在运输其他
                c = self.workers[w].dragging
                assert (w in self.cargo_dragged_by[c])
                continue    # 已经在运输其他
            # if not self.workers[w].dragging != c and self.workers[w].dragging != -1:
            if w not in self.cargo_dragged_by[c]:   # add new drag worker
                bool_cargo_sticky = self.cargo_hot[c] and len(self.cargo_dragged_by[c])<self.cargo_weight[c]
                if not bool_cargo_sticky:
                    continue
                self.cargo_dragged_by[c].append(w)
                assert self.workers[w].dragging == -1
                self.workers[w].dragging = c
                self.worker_pos_arr[w] = self.cargo[c].copy()   # set pos
                self.workers[w].state.p_vel *= 0                # clear vel
            else:
                assert self.workers[w].dragging == c, (self.workers[w].dragging, self.cargo_dragged_by[c])


        self.worker_previous_pos = self.worker_pos_arr.copy()

        self.reward_sample += worker_reward[0]
        self.cargo_previous = self.cargo

        for w in range(self.n_worker):  
            self.workers[w].state.p_pos = self.worker_pos_arr[w]
        return worker_reward.tolist()


    def reset_world(self, world):
        self.step = 0
        self.workers = world.agents

        n_lines = np.sqrt(self.n_worker).astype(np.int)
        n_cols = np.ceil(self.n_worker / n_lines).astype(np.int)

        n_objects = self.n_cargo * 2 + 1
        angle_div = 2 * np.pi / n_objects
        arm = 1.3

        init_pos_sel = np.zeros(shape=(n_objects, 2))
        for i in range(n_objects):
            angle_ = np.pi - i * angle_div
            init_pos_sel[i] = np.array([np.cos(angle_), np.sin(angle_)]) + np.array([0, 0])
            init_pos_sel[i] *= arm

        y = np.arange(n_objects)
        np.random.shuffle(y)
        init_pos_sel = init_pos_sel.copy()[y]

        worker_init_center = init_pos_sel[0]*0
        for i in range(n_lines):
            for j in range(n_cols):
                if i * n_cols + j >= self.n_worker: break
                world.agents[int(i * n_cols + j)].state.p_pos = \
                    np.array([0.1 * i - 0.05 * n_lines, 0.1 * j - 0.05 * n_cols]) + worker_init_center

        for agent in self.workers:
            agent.state.p_vel = np.zeros(world.dim_c)
            agent.movable = True
            agent.dragging = -1

        self.cargo = init_pos_sel[1: 1 + self.n_cargo]
        self.cargo_hot = [True for _ in range(self.n_cargo)]
        self.cargo_init_pos = self.cargo.copy()

        self.cargo_weight = np.zeros(self.n_cargo)

        self.cargo_drop_off = init_pos_sel[1 + self.n_cargo:]

        self.cargo_dragged_by = [[] for _ in range(self.n_cargo)]

        # init cargo_weight

        def part_an_int(an_int, n_piece):
            s = np.arange(an_int)
            np.random.shuffle(s)
            s = s[:n_piece-1]
            s = np.concatenate(([0, an_int], s))
            s = np.sort(s)
            w = []
            for i in range(self.n_cargo):
                w.append(s[i + 1] - s[i])
            return w


        # 60% random, 40% average, ?% weight
        sum_of_weight = int(self.n_worker*self.weight_percent)
        base = np.floor(sum_of_weight/self.n_cargo*0.4) # AVERAGE

        left_space = sum_of_weight - base*self.n_cargo
        grow = part_an_int(left_space, self.n_cargo)
        for i in range(self.n_cargo):
            self.cargo_weight[i] = int(grow[i] + base)

        assert self.cargo_weight.sum() == sum_of_weight
        # print(self.cargo_weight)

        self.cargo_moving = [False for _ in range(self.n_cargo)]
        self.n_cargo_lifted = 0
        self.worker_previous_pos = None
        world.steps = 0
        self.reward_sample = 0
        self.cargo_all_delivered = False
        if self.show_off:
            print('reset world, world 0 weight',self.cargo_weight)

        for i, agent in enumerate(world.agents):
            agent.live = True


    def reward(self, agent, world):
        assert self.joint_rewards is not None
        reward = self.joint_rewards[agent.iden]
        if agent.iden == self.n_agent:
            self.joint_rewards = None
        return reward

    def done(self, agent, world):
        # DEBUG!!
        condition1 = world.steps >= world.MaxEpisodeStep
        # print(world.steps, q)
        condition_success = self.cargo_all_delivered
        return condition1 or condition_success


    def check_obs(self):
        assert self.obs_pointer == self.obs_dimension

    def info(self, agent, world):
        return {'world_steps': world.steps, 'is_success':self.cargo_all_delivered}

    def make_world(self):
        world = World()  # set any world properties first
        world.dim_c = 2
        n_agent = self.n_worker
        world.agents = [Agent(iden=i) for i in range(n_agent)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False  # no collide
            agent.silent = True
            agent.id_in_team = i
            agent.size = 1
            agent.accel = self.acc
            agent.max_speed = 25 * 0.15
            agent.live = True
            agent.movable = True
            agent.initial_mass = 15
        self.workers = world.agents
        self.cargo = None
        self.cargo_drop_off = None
        self.reset_world(world)
        world.MaxEpisodeStep = ScenarioConfig.MaxEpisodeStep
        return world

    @staticmethod
    def rand(low, high):
        return np.random.rand() * (high - low) + low
