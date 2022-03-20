import numpy as np
import time
from numba import jit

# written by qth,2021/04/22
from scipy.optimize import linear_sum_assignment
from scipy.cluster.vq import kmeans2
from sklearn.cluster import DBSCAN
from .cheat_script_ai import CheatScriptAI
from .cython_func import laser_hit_improve3
# action of the agent


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
        self.initial_mass = 1.0
        
    @property
    def mass(self):
        return self.initial_mass



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
        self.act = None
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
class World(CheatScriptAI):
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

        # 留给内置AI的后门
        if len(self.scripted_agents) != 0: self.attackers_policy_1(self.scripted_agents, self.guards)
        
        ## -------- apply effects of laser ------------- ##
        self.apply_laser_effect()  
        
        # ------------- Calculate total physical (p_force) on each agent ------------- #
        p_force = [None] * len(self.active_entities)
        # apply agent physical controls
        p_force = self.apply_action_force(p_force)
        # apply environment forces
        # p_force = self.apply_environment_force(p_force)
        ## apply wall collision forces
        p_force = self.apply_wall_collision_force(p_force)
        # integrate physical state
        # calculates new state based on forces
        self.integrate_state(p_force)

    # gather agent action forces
    def apply_action_force(self, p_force):
        for i,agent in enumerate(self.alive_agents):
            p_force[i] = agent.act[:2] 
        return p_force


    def apply_wall_collision_force(self, p_force):
        for a,agent in enumerate(self.alive_agents):
            f = self.get_wall_collision_force(agent)
            if(f is not None):
                assert p_force[a] is not None
                p_force[a] = f + p_force[a] 
        return p_force

    def set_terrain_adv(self):
        terrain_A = self.s_cfg.terrain_parameters[0]
        terrain_B = self.s_cfg.terrain_parameters[1]
        if self.s_cfg.introduce_terrain:
            pos_arr = np.array([a.pos for a in self.alive_agents])
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
            if entity.can_fire:
                for b, entity_b in enumerate(self.alive_agents):
                    if entity.attacker == entity_b.attacker: continue # both attacker or both defender

                    fanRadius  = entity.shootRad*entity.terrain
                    fanOpenRad = entity.shootWin
                    fanDirRad  = entity.atk_rad
                    hit__4 = laser_hit_improve3(
                        entity.pos, entity_b.pos, 
                        fanRadius, fanOpenRad, fanDirRad
                    )

                    # assert hit__3==hit__4
                    if hit__4:
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
                entity.vel = entity.vel * (1 - self.damping)
                if (p_force[i] is not None):
                    entity.vel += (p_force[i] / entity.mass) * self.dt
                if entity.max_speed is not None:
                    speed = np.sqrt(np.square(entity.vel[0]) + np.square(entity.vel[1]))
                    if speed > entity.max_speed:
                        entity.vel = entity.vel / np.sqrt(np.square(entity.vel[0]) +
                                                                      np.square(entity.vel[1])) * entity.max_speed
                ## simple model for rotation
                # entity.atk_rad += entity.act[2]%(2*np.pi)    ## 导致数值爆炸，这是哪个伞兵写的？脑子被驴踢了
                if entity.alive:
                    entity.atk_rad += entity.act[2]
                    entity.atk_rad = reg_angle(entity.atk_rad)
                
            entity.pos += entity.vel * self.dt


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


    #  fanRadius = agent.shootRad*agent.terrain
    #  fanOpenRad = agent.shootWin
    #  fanDirRad = agent.atk_rad
    def get_tri_pts_arr(self, agent):
        max_fire_range = agent.shootRad
        terrain = agent.terrain
        # assert terrain > 0.7 and terrain <= 1.2, (terrain, 'overflow')
        fire_range_fix = max_fire_range*terrain
        ang = agent.atk_rad
        # pt1 = agent.pos + agent.size*np.array([np.cos(ang), np.sin(ang)]) # 这句代码把发射点从中心点往atk_rad方向偏移一点
        pt1 = agent.pos # 去除掉，使得智能体的size不再影响攻击范围
        pt2 = pt1 + fire_range_fix*np.array([np.cos(ang+agent.shootWin/2), np.sin(ang+agent.shootWin/2)])
        pt3 = pt1 + fire_range_fix*np.array([np.cos(ang-agent.shootWin/2), np.sin(ang-agent.shootWin/2)])
        A = np.array([[pt1[0], pt2[0], pt3[0]],
                      [pt1[1], pt2[1], pt3[1]]])       
        return A, fire_range_fix





    # collision force with wall
    def get_wall_collision_force(self, entity):
        if not entity.collide_wall:
            return None # not a collider
        xmin,xmax,ymin,ymax = self.wall_pos
        x,y = entity.pos
        size = entity.size
        dists = np.array([x-(size+xmin), xmax-x-size, y-size-ymin, ymax-y-size])
        if (dists>0).all(): return np.array([0,0])

        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -dists/k)*k
        fx1,fx2,fy1,fy2 = self.contact_force * penetration
        force = [fx1-fx2,fy1-fy2]
        return force