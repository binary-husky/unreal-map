import numpy as np
import time
try: 
    from numba import njit, jit
except: 
    from UTIL.tensor_ops import dummy_decorator as jit
    from UTIL.tensor_ops import dummy_decorator as njit

# written by qth,2021/04/22
from scipy.optimize import linear_sum_assignment
from scipy.cluster.vq import kmeans2
from sklearn.cluster import DBSCAN

# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical orientation
        self.p_ang = None ## extra angle attribute, can shoot only towards angle, can move along any direction
        # physical velocity
        self.p_vel = None

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None

# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None   ## first two components for x,y. third component for rotation
        self.shoot = False   ## number greater than 0 means to shoot
        # communication action
        self.c = None

# properties and state of physical world entity
class Entity(object):
    def __init__(self, size = 0.05 ,color = None):
        # name 
        self.name = ''
        # properties:
        self.size = size
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = color
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0
        
    @property
    def mass(self):
        return self.initial_mass

## bullet is an entity
class Bullet(Entity):
    def __init__(self, bulletType):
        super(Bullet, self).__init__(size = 0.01)
        self.name = 'bullet'
        self.movable = True
        self.type = bulletType    # 'attacker' or 'guard' bullet
        self.color = np.array([0, 0.85, 0]) if self.type == 'guard' else np.array([0.85, 0.35, 0.35])

def distance_matrix_AB(A, B):
    assert A.shape[-1] == 2 # assert 2D situation
    assert B.shape[-1] == 2 # assert 2D situation
    n_A_subject = A.shape[-2]
    n_B_subject = B.shape[-2]
    A = np.repeat(np.expand_dims(A,-2), n_B_subject, axis=-2) # =>(64, Na, Nb, 2)
    B = np.repeat(np.expand_dims(B,-2), n_A_subject, axis=-2) # =>(64, Nb, Na, 2)
    Bt = np.swapaxes(B,-2,-3) # =>(64, Na, Nb, 2)
    dis = Bt-A # =>(64, Na, Nb, 2)
    dis = np.linalg.norm(dis, axis=-1)
    return dis

# properties of landmark entities
class Landmark(Entity):
     def __init__(self):
        super(Landmark, self).__init__()

# properties of agent entities
class Agent(Entity):
    def __init__(self, iden=None):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None
        # script behavior to execute
        self.action_callback_test = None
        ## number of bullets hit
        self.numHit = 0         # overall
        self.numWasHit = 0
        self.hit = False        # in last time
        self.wasHit = False
        ## shooting cone's radius and width (in radian)
        self.shootRad = 0.4 # default value (same for guards and attackers, can be changed in collective_assult_env)
        self.shootWin = np.pi/4
        self.alive = True   # alive/dead
        self.justDied = False   # helps compute reward for agent when it just died
        self.prevDist = None
        if iden is not None:
            self.iden = iden

# multi-agent world
class World(object):
    def __init__(self):
        ## lists of agents, entities and bullets (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        self.bullets = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 3  ## x, y, angle
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-10 # 1e-3
        ## wall positions
        # self.wall_pos = [-1,1,-0.8,0.8] # (xmin, xmax) vertical and  (ymin,ymax) horizontal walls

        self.wall_pos = [-1, 1, -1, 1]  # (xmin, xmax) vertical and  (ymin,ymax) horizontal walls

        # written by qth, 2021/04/20,用于判断是否第一次初始化
        self.start_flag = True
        self.target_index = 0
        # self.leader_id = 4
        self.tar_pos = np.full((4, 2), 0, "float")
        # red
        # self.tar_pos[0][0] = 0
        # self.tar_pos[0][1] = -0.5
        # self.tar_pos[1][0] = 0.5
        # self.tar_pos[1][1] = 0
        # self.tar_pos[2][0] = -0.5
        # self.tar_pos[2][1] = 0
        # self.tar_pos[3][0] = 0
        # self.tar_pos[3][1] = 1
        self.teams_result_step1 = None
        self.team_centroid_step1 = None
        from .collective_assult_parallel_run import ScenarioConfig
        self.s_cfg = ScenarioConfig

    # return all alive agents
    @property
    def alive_agents(self):
        return [agent for agent in self.agents if agent.alive]

    # return all agents that are not adversaries
    @property
    def alive_guards(self):
        return [agent for agent in self.agents if (agent.alive and not agent.attacker)]

    # return all agents that are not adversaries
    @property
    def guards(self):
        return [agent for agent in self.agents if not agent.attacker]

    # return all adversarial agents
    @property
    def alive_attackers(self):
        return [agent for agent in self.agents if (agent.alive and agent.attacker)]


    # return all adversarial agents
    @property
    def attackers(self):
        return [agent for agent in self.agents if agent.attacker]

    # return all active in the world
    @property
    def active_entities(self):
        return [agent for agent in self.agents if agent.alive] + self.landmarks + self.bullets ## now bullets are also entities


    # return all entities in the world
    @property
    def entities(self):
        return [agent for agent in self.agents] + self.landmarks + self.bullets ## now bullets are also entities

    # return all agents controllable by external policies
    @property
    def alive_policy_agents(self):
        return [agent for agent in self.agents if (agent.alive and agent.action_callback is None)]


    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]


    # return all agents controlled by world scripts
    @property
    def active_scripted_agents(self):
        return [agent for agent in self.agents if (agent.alive and agent.action_callback is not None)]


    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents_test(self):
        return [agent for agent in self.agents if agent.action_callback_test is not None]
    # update state of the world
    def step(self):
        # print('second step')
        # set actions for scripted agents
        ## IGNORE FOLLOWING: scripted agents are probably non-learning heuristic agents 
        # for agent in self.scripted_agents_test:
        #     agent.action_callback_test(agent, self)
        # for agent in self.scripted_agents:
        #     agent.action_callback(agent, self)
            # print(agent.action)

        # 判断是否存在脚本策略，同时存活着的的对手，如果有则调用脚本策略
        # written by qth,2021/04/16
        if len(self.scripted_agents) != 0:
            self.attackers_policy_1(self.scripted_agents, self.alive_guards)

        ## The following is where are actions are actually applied for learning agents

        
        ## -------- apply effects of laser ------------- ##
        self.apply_laser_effect()      ## calling it effect as it might apply force, kill, etc...  
        
        # ------------- Calculate total physical (p_force) on each agent ------------- #
        p_force = [None] * len(self.active_entities)
        # apply agent physical controls
        p_force = self.apply_action_force(p_force)
        # apply environment forces
        p_force = self.apply_environment_force(p_force)
        ## apply wall collision forces
        p_force = self.apply_wall_collision_force(p_force)
        # integrate physical state
        # calculates new state based on forces
        self.integrate_state(p_force)

        ## The following is for communication - IGNORE --------------- ##
        # update agent communication state
        for agent in self.alive_agents:
            self.update_agent_state(agent)


    def attackers_policy_1(self, attackers_agent, guards_agent):
        '''
        大规模集群协同策略
        Args:
            attackers_agent: 攻击方智能体
            guards_agent: 防守方智能体
        Returns:
        written by qth, 2021/04/13
        '''
        # 初始化
        for i, agent in enumerate(attackers_agent):
            agent.action.u = np.zeros(self.dim_p)  ## We'll use this now for Graph NN
            agent.action.shoot = False

        # 设定我方智能体的感知范围
        obversation_radius =  (self.wall_pos[1] - self.wall_pos[0]) #/3 # 即整个地图x轴方向的1/3
        joint_forces = [0, 0]

        ##############  初始化参数  ###############
        eps = 0.5   # 态势评估时，以该值作为半径，对所有对手分布进行聚类成簇
        MinPoints = 2  # 表示每簇中最少的对手个数
        num_members = 5  # 我方agent每组成员个数
        min_members =3  # 我方agent每组成员最少个数，一旦小于该值，则重新分配组队。

        ################  态势评估  ###############
        # 通过聚类算法，对感知到的对手分布情况进行聚类，评估对手的整体分布情况
        # 聚类的个数为我方agent的分组数
        # 下面先从全局感知角度，利用聚类方法，计算对手分布情况。
        guards_agent_position = np.zeros((len(guards_agent), 2))
        for i_g, guard_agent in enumerate(guards_agent):
            guards_agent_position[i_g] = guard_agent.state.p_pos
        # print(guards_agent_position)


        result = DBSCAN(eps, min_samples = MinPoints).fit(guards_agent_position)
        # labels_表示聚类后的类别，通常为0,1,2..,最大数表示聚类后的簇的个数，-1表示属于噪音，
        # 不在要求的范围内的或者个数小于2个的类别都会标识为-1
        # label = result.labels_
        # print(label)
        # cluster_results用于存储落在各个不同簇的元素，其中键值即为簇的类别（0,1,2...)
        cluster_results = {}
        for cluster_index, cluster_class in enumerate(result.labels_):
            if cluster_class not in cluster_results.keys():
                cluster_results[cluster_class] = [guards_agent_position[cluster_index]]
                # print(guards_agent_position[cluster_index])
            else:
                cluster_results[cluster_class].append(guards_agent_position[cluster_index])

        # 对手分布中各簇的中心位置列表，作为我方agent的打击目标
        cluster_centroid = []
        cluster_radius = []
        for key in cluster_results.keys():
            cluster_index_position = np.array(cluster_results[key])
            # print("cluster_class:%d" % key)
            # print(cluster_index_position)
            # 对各个簇的各元素再次聚类，得到该簇的中心点，用于引导我方agent
            # 其中，centroid表示该簇的中心，label表示k-means2方法聚类得到的标签类别，由于聚类k为1
            # 因此，此处的label都是同一类的簇
            # 其中，key=-1的簇不能作为中心，因为所有噪点都标记为-1类，该中心不具有实际意义。
            if key != -1:
                for i in range(5):
                    try:
                        centroid, label  = kmeans2(cluster_index_position, 1, iter=20, minit='++',seed=np.random.randint(100), missing='raise')
                        break
                    except:
                        pass
                    if i >= 4:
                        team_centroid, team_labels = kmeans2(cluster_index_position, 1, iter=20, minit='++',seed=np.random.randint(100), missing = 'warn')
                        print('处理空聚类')
                        break
                        # assert False                
                cluster_centroid.append(centroid)
            else:
                # centroid, label = kmeans2(cluster_index_position, len(cluster_results[key]), iter=20, minit='points')
                # centroid = [centroid]
                for item in cluster_index_position:
                    cluster_centroid.append([item])

            # 根据对手分布情况，即每簇中距离最远的两个个体的距离作为我方的直径。
            dists_to_centroid = np.array(
            [np.linalg.norm(cluster_centroid[-1] - cluster_index_position_pos) for cluster_index_position_pos in cluster_index_position])
            
            cluster_radius.append(max(dists_to_centroid))


        ##################### 对我方agents进行分组 ##################
        # print("num_member4:")
        # print(num_members)
        if self.start_flag:
            # team_centroid_step1聚类中心标准顺序 ；；；teams_result_step1 字典，key=类别，成员=元素，
            self.teams_result_step1, self.team_centroid_step1 = self.get_groups(attackers_agent, num_members)
            self.start_flag = False
        # print(teams_result_step1)
        teams_result_step2 = self.teams_result_step1
        team_centroid_step2 = self.team_centroid_step1
        # 判断各组成员数量是否过少，小于3个时，就需要重新组队
        for key in self.teams_result_step1.keys():
            team_index_agents = np.array(self.teams_result_step1[key])
            if len(team_index_agents) <= min_members:
                # print("num_member5:")
                # print(num_members)
                teams_result_step2, team_centroid_step2 = self.get_groups(attackers_agent, num_members)
                break

        ##################### 对我方agents各分组进行指派 ##################
        target_cluster_centroid = cluster_centroid
        # print("target_cluster_centroid1:")
        # print(target_cluster_centroid)
        if len(cluster_centroid) < len(teams_result_step2):
            num_cluster_centroid = len(cluster_centroid)
            if num_cluster_centroid ==0:
                num_cluster_centroid = 1
            temp_num_members = len(attackers_agent) // num_cluster_centroid
            # print("temp_num_members:")
            # print(temp_num_members)
            # print(cluster_centroid)
            # teams_result_step2, team_centroid_step2 = self.get_groups(attackers_agent, temp_num_members)
            for index in range(len(teams_result_step2) - len(cluster_centroid) ):
                target_cluster_centroid.append(target_cluster_centroid[-1])            
                # print("target_cluster_centroid3:")
            # print(len(teams_result_step2))
            # print(teams_result_step2)
        else:
            target_cluster_centroid = []
            target_cluster_centroid = cluster_centroid[:(len(teams_result_step2))]
            # print("target_cluster_centroid2:")
            # print(target_cluster_centroid)
            # print(cluster_centroid)

        # 此时，先默认k-means计算出来的各聚类中心列表，与聚类列别:0,1,2,3..,-1对应
        # 但上面这个是需要测试的，如果不对应，那么下面这种分配就有问题；
        # print("team_centroid_step2:")
        # print(team_centroid_step2)
        # print("target_cluster_centroid:")
        # 聚类中心标准顺序 -> 计算聚类中心、对方聚类中心矩阵
        # dists_to_target_cluster = np.array(
        #     [[np.linalg.norm(team_centroid_position - target_cluster_centroid_pos) for target_cluster_centroid_pos in target_cluster_centroid]
        #      for i, team_centroid_position in enumerate(team_centroid_step2)])  # 计算距离矩阵
        # 计算距离矩阵
        A = team_centroid_step2
        B = np.array(target_cluster_centroid).squeeze()
        dists_to_target_cluster2 = distance_matrix_AB(A, B)
        # assert (dists_to_target_cluster2 == dists_to_target_cluster).all()
        # print(dists_to_target_cluster)
        # 聚类中心标准顺序 -> 计算聚类中心、对方聚类中心矩阵 -> 匈牙利算法（聚类中心-聚类中心的距离）
        ri_team_to_cluster, ci_team_to_cluster = linear_sum_assignment(dists_to_target_cluster2)
        # 由于下面没有用leader_id,暂时设置一个随机的，后面必须取消。
        # print(len(ci_team_to_cluster))
        self.leader_id = -10
        # print("teams_result_step2:")
        # print(teams_result_step2)
        # team_i = 0
        for key, team_agents in teams_result_step2.items():
            # 我方的每一个聚类 key=类别，team_agents=成员列表，
            # 计算各follower的位置
            # print(team_agents)
            delta_angle = (float)(np.pi / len(team_agents) -1)
            expected_poses_patrol = []
            # print("team_i:%d" % team_i)
            # print("team_class_key:%d" % key)
            # print(teams_result_step2)
            # print(target_cluster_centroid)    # 目标中心（10，2） 对应10个聚类 
            # print(key)
            # print(ci_team_to_cluster[key])    # ci_team_to_cluster （10，1）
            if key >= len(target_cluster_centroid): 
                key = len(target_cluster_centroid) - 1

            target_cluster = ci_team_to_cluster[key]
            leader_position_patrol = np.array(target_cluster_centroid[target_cluster])  # 领航者的位置
            # team_i += 1
            circle_radiu = 0.8
            # circle_radiu = Agent().shootRad * (len(team_agents) - 1) / (2 * np.pi)
            # print(key)
            # circle_radiu = cluster_radius[ci_team_to_cluster[key]] * 0.8

            for i, agent in enumerate(team_agents):
                if agent.iden != self.leader_id:
                    # if fighter_live_num_list[i] > self.leader_id:
                    #     expected_poses_patrol.append(np.array([leader_position_patrol + circle_radiu * np.array(
                    #         [np.cos(angle * (i - 1) * np.pi / 180), np.sin(angle * (i - 1) * np.pi / 180)])]))
                    # else:
                    expected_poses_patrol.append([leader_position_patrol + circle_radiu * np.array(
                        [np.cos(delta_angle * i), np.sin(delta_angle * i)])])
            dists_patrol = np.array(
                [[np.linalg.norm(np.array([agent.state.p_pos[0], agent.state.p_pos[1]]) - pos) for pos in expected_poses_patrol]
                 for i, agent in enumerate(team_agents) if agent.iden != self.leader_id])
            ri, ci = linear_sum_assignment(dists_patrol)

            for i, agent in enumerate(team_agents):
                if agent.iden == self.leader_id:
                    continue
                # if y == fighter_live_num_list[i]:
                #     if y > self.leader_id:
                #         expected_poses_for_it = expected_poses_patrol[ci[i - 1]]
                #     else:
                # print(expected_poses_patrol)
                expected_poses_for_it = expected_poses_patrol[ci[i]]
                # print("expected_poses_for_it")
                # print(expected_poses_for_it)
                relative_value_patrol = expected_poses_for_it - np.array([agent.state.p_pos[0], agent.state.p_pos[1]])
                # print(relative_value_patrol)
                theta_patrol = np.arctan2(relative_value_patrol[0][0][1], relative_value_patrol[0][0][0])
                if theta_patrol < 0:
                    theta_patrol += 2 * np.pi

                # print(expected_poses_for_it[0])
                agent.action.u[0] = -agent.state.p_pos[0] + expected_poses_for_it[0][0][0]
                agent.action.u[1] = -agent.state.p_pos[1] + expected_poses_for_it[0][0][1]

                if self.s_cfg.DISALBE_RED_FUNCTION:
                    agent.action.u[0] = -agent.action.u[0]
                    agent.action.u[1] = -agent.action.u[1]

                # 计算各跟随者的打击角度，超打击范围内，距离最近的对手方向射击
                for guard_i, guard_agent in enumerate(guards_agent):
                    # guard_agent.action.shoot = False
                    if (agent.state.p_pos[0] - agent.shootRad) <= guard_agent.state.p_pos[0] <= (
                            agent.state.p_pos[0] + agent.shootRad) and (
                            agent.state.p_pos[1] - agent.shootRad) <= guard_agent.state.p_pos[1]  <= (
                            agent.state.p_pos[1] + agent.shootRad):
                        relative_to_guard = guard_agent.state.p_pos - agent.state.p_pos
                        theta_patrol = np.arctan2(relative_to_guard[1], relative_to_guard[0])
                        if theta_patrol < 0:
                            theta_patrol += 2 * np.pi
                        # print("attackID:%d, guardID:%d" % (agent.iden,guard_agent.iden))
                if agent.alive:
                    agent.state.p_ang = theta_patrol
                agent.action.shoot = True
                if self.s_cfg.DISALBE_RED_FUNCTION:
                    agent.action.shoot = False



    def get_groups(self, policy_agents, num_members):
        '''
        通过k-means方法对我方agents进行分组
        Args:
            policy_agents: 我方存活的agents
            num_members: 每组要求的最少成员数量

        Returns: 分组后的列表字典，键为分组编号，值为agent类实体。
        written by qth，2021/04/25
        '''
        # print("num_member3:")
        # print(num_members)
        if num_members == 0:
            num_members = 1
        # print("num_member:")
        # print(num_members)
        num_team = len(policy_agents) // num_members
        if num_team == 0:
            num_team = 1
        # print("num_member2:")
        # print(num_members)
        policy_agents_position = []
        for i, agent in enumerate(policy_agents):
            policy_agents_position.append(agent.state.p_pos)
        # team_centroid中心列表[10]， team_labels智能体归属列表[50]
        for i in range(5):
            try:    # its here
                team_centroid, team_labels = kmeans2(policy_agents_position, num_team, iter=20, minit='++',seed=np.random.randint(100), missing = 'raise')
                break
            except:
                pass
            if i >= 4:
                team_centroid, team_labels = kmeans2(policy_agents_position, num_team, iter=20, minit='++',seed=np.random.randint(100), missing = 'warn')
                print('处理空聚类')
                break

        if min(team_labels) != 0:
            assert min(team_labels) == 0
        team_results = {}
        for team_index, team_class in enumerate(team_labels):
            if team_class not in team_results.keys(): # 把属于该组的成员，存储到一个列表字典里，键为分组的编号
                team_results[team_class] = [policy_agents[team_index]]
            else:
                team_results[team_class].append(policy_agents[team_index])
        return team_results, team_centroid


    # gather agent action forces
    def apply_action_force(self, p_force):
        # set applied forces
        for i,agent in enumerate(self.alive_agents):
            force_dim = agent.action.u.shape[0]-1 ## 3rd u dimension is for rotation
            if agent.movable:
                noise = np.random.randn(*force_dim) * agent.u_noise if agent.u_noise else 0.0     ##
                p_force[i] = agent.action.u[:2] + noise           ##    
        return p_force

    # gather physical forces acting on entities
    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response
        for a,entity_a in enumerate(self.active_entities):
            for b,entity_b in enumerate(self.active_entities):
                if(b <= a): continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if(f_a is not None):
                    if(p_force[a] is None): p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a] 
                if(f_b is not None):
                    if(p_force[b] is None): p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]        
        return p_force

    def apply_wall_collision_force(self, p_force):
        for a,agent in enumerate(self.alive_agents):
            f = self.get_wall_collision_force(agent)
            if(f is not None):
                if(p_force[a] is None): p_force[a] = 0.0
                p_force[a] = f + p_force[a] 
        return p_force

    def set_terrain_adv(self):
        terrain_A = self.s_cfg.terrain_parameters[0]
        terrain_B = self.s_cfg.terrain_parameters[1]
        if self.s_cfg.introduce_terrain:
            pos_arr = np.array([a.state.p_pos for a in self.alive_agents])
            terrain = self.get_terrain(pos_arr, theta=self.init_theta, A=terrain_A, B=terrain_B)
            for i,entity in enumerate(self.alive_agents):
                entity.terrain = terrain[i]
        else:
            for i,entity in enumerate(self.alive_agents):
                entity.terrain = 1.0


    def apply_laser_effect(self):
        ## reset bullet hitting states
        for i,entity in enumerate(self.alive_agents):
            entity.hit = False
            entity.wasHit = False
            entity.wasHitBy = None

        for i,entity in enumerate(self.alive_agents):
            # print(entity.action.shoot)
            if entity.action.shoot and entity.numbullets > 0:
                # if entity.attacker:
                    # print(i,entity.numbullets)
                if entity.bullets_is_limited:
                    entity.numbullets -= 1
                # print()
                # let's use lasers - instantaneously hit entities within a cone (no approximate)
                # compute cone
                A, fire_range_fix = self.get_tri_pts_arr(entity)

                for b,entity_b in enumerate(self.alive_agents):
                    if entity.attacker == entity_b.attacker:    # the same team
                        continue

                    hit__3 = self.laser_hit_improve2(A, entity_b.state.p_pos, fire_range_fix)
                    # assert hit__3 == hit__2
                    if hit__3:
                        entity.hit = True
                        entity.numHit += 1
                        entity_b.wasHit = True
                        entity_b.wasHitBy = entity
                        entity_b.numWasHit += 1
        
        # update just died state of dead agents
        for agent in self.agents:
            if not agent.alive:
                agent.justDied = False

        ## laser directly kills with one shot
        for agent in self.alive_agents:
            if agent.wasHit:
                agent.alive = False
                agent.justDied = True
                agent.color *= 0.5
                if agent.attacker:
                    self.numAliveAttackers -= 1
                else:
                    self.numAliveGuards -= 1

    # integrate physical state
    def integrate_state(self, p_force):
        def reg_angle(rad):
            return (rad + np.pi)%(2*np.pi) -np.pi
        for i,entity in enumerate(self.active_entities):
            if not entity.movable: continue
            if not 'bullet' in entity.name:
                entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
                if (p_force[i] is not None):
                    entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
                    # if entity.attacker:
                    #     entity.state.p_vel[0] = entity.state.p_pos[0] - entity.state.p_pos[0]
                    #     entity.state.p_vel[1] = self.wall_pos[3] - entity.state.p_pos[1]
                if entity.max_speed is not None:
                    speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                    if speed > entity.max_speed:
                        entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
                                                                      np.square(entity.state.p_vel[1])) * entity.max_speed
                ## simple model for rotation
                # entity.state.p_ang += entity.action.u[2]%(2*np.pi)    ## 导致数值爆炸，这是哪个伞兵写的？脑子被驴踢了
                if entity.alive:
                    entity.state.p_ang += entity.action.u[2]
                    entity.state.p_ang = reg_angle(entity.state.p_ang)
                
            entity.state.p_pos += entity.state.p_vel * self.dt


    # taking higher position, taking shoot range advantage
    @staticmethod
    @jit
    def get_terrain(arr, theta, A, B):
        # A = 0.05; B=0.2
        X=arr[:,0]; Y=arr[:,1]
        X_ = X*np.cos(theta) + Y*np.sin(theta)
        Y_ = -X*np.sin(theta) + Y*np.cos(theta)
        Z = -1 +B*( (0.1*X_) ** 2 + (0.1*Y_) ** 2 )- A * np.cos(2 * np.pi * (0.3*X_))  - A * np.cos(2 * np.pi * (0.5*Y_))
        return -Z

    def get_tri_pts_arr(self, agent):
        max_fire_range = agent.shootRad
        terrain = agent.terrain
        # assert terrain > 0.7 and terrain <= 1.2, (terrain, 'overflow')
        fire_range_fix = max_fire_range*terrain
        ang = agent.state.p_ang
        pt1 = agent.state.p_pos + agent.size*np.array([np.cos(ang), np.sin(ang)])
        pt2 = pt1 + fire_range_fix*np.array([np.cos(ang+agent.shootWin/2), np.sin(ang+agent.shootWin/2)])
        pt3 = pt1 + fire_range_fix*np.array([np.cos(ang-agent.shootWin/2), np.sin(ang-agent.shootWin/2)])
        A = np.array([[pt1[0], pt2[0], pt3[0]],
                      [pt1[1], pt2[1], pt3[1]]])       
        return A, fire_range_fix


    # def get_tri_pts_arr2(self, agent):
    #     max_fire_range = agent.shootRad
    #     terrain = agent.terrain
    #     assert terrain > 0.7 and terrain <= 1.2, (terrain, 'overflow')
    #     fire_range_fix = max_fire_range*((terrain-1)*4 +1)
    #     ang = agent.state.p_ang
    #     pt1 = agent.state.p_pos + agent.size*np.array([np.cos(ang), np.sin(ang)])
    #     pt2 = pt1 + fire_range_fix*np.array([np.cos(ang+agent.shootWin/2), np.sin(ang+agent.shootWin/2)])
    #     pt3 = pt1 + fire_range_fix*np.array([np.cos(ang-agent.shootWin/2), np.sin(ang-agent.shootWin/2)])
    #     A = np.array([[pt1[0], pt2[0], pt3[0]],
    #                   [pt1[1], pt2[1], pt3[1]]])       
    #     return A, fire_range_fix

    @staticmethod
    @jit
    def laser_hit_improve2(A, p_pos, shootRad):
        origin = A[:2, 0]
        vpb = p_pos - origin
        dis = np.sqrt(vpb[0]*vpb[0]+vpb[1]*vpb[1])
        # dis = np.linalg.norm(vpb)
        if dis > shootRad: return False

        vp1 = A[:2, 1] - origin
        vp2 = A[:2, 2] - origin

        # cos12 = vp1.dot(vp2)/(agent.shootRad*agent.shootRad) # cos12 = self.getcos(vp1, vp2)
        # cos1b = vp1.dot(vpb)/(agent.shootRad*dis) # self.getcos(vp1, vpb)
        # cos2b = vp2.dot(vpb)/(agent.shootRad*dis) # self.getcos(vp2, vpb)
        # ↓ ↓ ↓ 消去没用的分母
        cos12 = (vp1[0]*vp2[0] + vp1[1]*vp2[1])/shootRad*dis
        cos1b = vp1[0]*vpb[0] + vp1[1]*vpb[1] # self.getcos(vp1, vpb)
        cos2b = vp2[0]*vpb[0] + vp2[1]*vpb[1] # vp2.dot(vpb)# self.getcos(vp2, vpb)
        if (cos12 < cos1b) and (cos12 < cos2b):
            return True
        else:
            return False

    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise      

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None] # not a collider
        if (entity_a is entity_b):
            return [None, None] # don't collide against itself
        # compute actual distance between entities
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]

    # collision force with wall
    def get_wall_collision_force(self, entity):
        if not entity.collide_wall:
            return None # not a collider
        xmin,xmax,ymin,ymax = self.wall_pos
        x,y = entity.state.p_pos
        size = entity.size
        dists = np.array([x-size-xmin, xmax-x-size, y-size-ymin, ymax-y-size])

        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -dists/k)*k
        fx1,fx2,fy1,fy2 = self.contact_force * penetration
        force = [fx1-fx2,fy1-fy2]
        return force