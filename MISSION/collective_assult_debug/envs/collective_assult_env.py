import time, gym
import numpy as np
from UTIL.tensor_ops import my_view
from ..core import World, Agent

class collective_assultEnvV1(gym.Env):  
    metadata = {'render.modes': ['human']}   
    def __init__(self,numguards =5, numattackers = 5, size=1.0):
        from ..collective_assult_parallel_run import ScenarioConfig
        self.init_dis = ScenarioConfig.init_distance
        self.half_death_reward = ScenarioConfig.half_death_reward
        self.random_jam_prob = ScenarioConfig.random_jam_prob
        self.world = World() 
        self.world.wall_pos=[-1*size,1*size,-1*size,1*size]
        self.world.init_box=[-1*5,1*5,-1*5,1*5]
        self.world.fortDim = 0.15   # radius
        self.world.doorLoc = np.array([0,0]) #
        self.world.numGuards = numguards  # initial number of guards, attackers and bullets
        self.world.numAttackers = numattackers
        self.world.numAgents = self.world.numGuards + self.world.numAttackers
        self.world.numAliveGuards, self.world.numAliveAttackers, self.world.numAliveAgents = self.world.numGuards, self.world.numAttackers, self.world.numAgents
        self.world.atttacker_reached = False     ## did any attacker succeed to reach the gate?
        self.world.shared_resorce = {}
        self.attacker_reward_sum = 0
        self.guard_reward_sum = 0
        self.world.agents = [Agent(iden=i) for i in range(self.world.numAgents)]
        for i, agent in enumerate(self.world.agents):
            agent.name = 'agent %d' % (i+1)
            agent.collide = False
            agent.collide_wall = True
            agent.silent = True
            agent.bullets_is_limited = False #
            agent.attacker = False if i < self.world.numGuards else True
            agent.accel = 3  ## guard
            agent.max_speed = 1.0   #
            agent.max_rot = 0.17 ## a
            agent.action_callback = self.action_callback if agent.attacker else None #

        self.viewers = [None]
        self.render_geoms = None
        self.shared_viewer = True
        self.world.time_step = 0
        self.world.max_time_steps = None #  set inside malib/environments/collective_assult 最大步数为100 在外围初始化
        self.world.vizDead = True         # whether to visualize the dead agents
        self.world.vizAttn = True        # whether to visualize attentions
        self.world.gameResult = np.array([0,0,0,0,0]) #  [guards all win, guard win, attacker all win, attcker win, draw]
        self.reset_world()        

    # a fake callback, don't know what's for, do not del it 
    def action_callback(self,agent,world):
        pass


    def reset_world(self):
        # light green for guards and light red for attackers
        self.world.render_reset_flag = True
        self.world.time_step = 0
        self.world.bullets = [] ##
        self.world.numAliveAttackers = self.world.numAttackers
        self.world.numAliveGuards = self.world.numGuards
        self.world.numAliveAgents = self.world.numAgents
        self.world.gameResult[:] = 0
        theta = (2*np.random.rand()-1)*np.pi
        self.world.init_theta = theta
        rotate = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        for i, agent in enumerate(self.world.agents):
            agent.alive = True
            agent.color = np.array([0.0, 1.0, 0.0]) if not agent.attacker else np.array([1.0, 0.0, 0.0])
            agent.vel = np.zeros(self.world.dim_p-1)    ##
            # agent.state.c = np.zeros(self.world.dim_c)
            agent.atk_rad = (theta+np.pi) + (np.random.rand()-0.5)/12 if agent.attacker else (theta + (np.random.rand()-0.5)/12)

            xMin, xMax, yMin, yMax = self.world.init_box
            xMid = xMin/2 + xMax/2
            yMid = yMin/2 + yMax/2
            xInitDis = self.init_dis

            if agent.attacker:
                #随机初始化位置
                x_ = xMid+xInitDis/2
                y_ = (yMax-yMin)/self.world.numAttackers*(agent.iden - self.world.numGuards +0.5) + yMin
                agent.pos = np.array([x_, y_])
                agent.pos += (np.random.randn(2,)-0.5)/10

                if self.world.numAttackers>48:
                    centering = np.array([xMid, yMid])
                    ratio = 1
                    if agent.iden%3 == 0:
                        ratio = 0.5
                    if agent.iden%3 == 1:
                        ratio = 0.75
                    agent.pos = centering + (agent.pos-centering)*ratio

                agent.pos = np.dot(agent.pos, rotate.T)

            else:
                #随机初始化位置
                x_ = xMid-xInitDis/2 
                y_ = (yMax-yMin)/self.world.numGuards*(agent.iden+0.5) + yMin
                agent.pos = np.array([x_, y_])
                agent.pos += (np.random.randn(2,)-0.5)/10

                if self.world.numGuards>48:
                    centering = np.array([xMid, yMid])
                    ratio = 1
                    if agent.iden%3 == 0: 
                        ratio = 0.5
                    if agent.iden%3 == 1:
                        ratio = 0.75
                    agent.pos = centering + (agent.pos-centering)*ratio # 以地图中心为中心，按照ratio放射向量变换位置

                agent.pos = np.dot(agent.pos, rotate.T)

            agent.numHit = 0
            agent.numWasHit = 0
            agent.hit = False
            agent.wasHit = False

    # return all agents that are attackers
    def alive_attackers(self):
        return [agent for agent in self.world.agents if ( (agent.alive or agent.justDied) and agent.attacker)]
    # return all agents that are not attackers
    def alive_guards(self):
        return [agent for agent in self.world.agents if ( (agent.alive or agent.justDied) and not agent.attacker)]


    # return all agents that are attackers
    def attackers(self):
        return [agent for agent in self.world.agents if (agent.attacker)]
    # return all agents that are not attackers
    def guards(self):
        return [agent for agent in self.world.agents if (not agent.attacker)]

    def reward(self, agent):
        if agent.alive or agent.justDied:
            main_reward = self.attacker_reward(agent) if agent.attacker else self.guard_reward(agent)
        else:
            main_reward = 0
        return main_reward

    def attacker_reward(self, agent):
        rew3, rew4 = 0,0
        for agents in self.alive_attackers():
            if agents.hit:
                rew3 = +1
            if agents.wasHit:
                rew4 = -1 if not self.half_death_reward else -0.5

        self.attacker_reward_sum = rew3+rew4
        return self.attacker_reward_sum

    def guard_reward(self, agent):
        rew5, rew6 = 0,0
        if agent.hit:
            rew5 += 1
        if agent.wasHit:
            rew6 = -1 if not self.half_death_reward else -0.5
        self.guard_reward_sum = rew5+rew6

        return self.guard_reward_sum
        
    raw_obs_size = -1
    class raw_obs_array(object):
        def __init__(self):
            if collective_assultEnvV1.raw_obs_size==-1:
                self.guards_group = []
                self.nosize = True
            else:
                self.guards_group = np.zeros(shape=(collective_assultEnvV1.raw_obs_size), dtype=np.float32)
                self.nosize = False
                self.p = 0

        def append(self, buf):
            if self.nosize:
                self.guards_group.append(buf)
            else:
                L = len(buf)
                self.guards_group[self.p:self.p+L] = buf[:]
                self.p += L

        def get(self):
            if self.nosize:
                self.guards_group = np.concatenate(self.guards_group)
                collective_assultEnvV1.raw_obs_size = len(self.guards_group)
            return self.guards_group

        
    @staticmethod
    def get_binary_array(n_int, n_bits=8, dtype=np.float32):
        arr = np.zeros((*n_int.shape, n_bits), dtype=dtype)
        pointer = 0
        for i in range(n_bits):
            arr[:, i] = (n_int%2==1).astype(int)
            n_int = n_int / 2
            n_int = n_int.astype(np.int8)
        return arr

    @staticmethod
    def item_random_mv(src,dst,prob,rand=False):
        assert len(src.shape)==1; assert len(dst.shape)==1
        if rand: np.random.shuffle(src)
        len_src = len(src)
        n_mv = (np.random.rand(len_src) < prob).sum()
        item_mv = src[range(len_src-n_mv,len_src)]
        src = src[range(0,0+len_src-n_mv)]
        dst = np.concatenate((item_mv, dst))
        return src, dst

    def observation(self, agent, world, get_obs_dim=False):
        if get_obs_dim: return 12*16
        if agent.iden == 0:
            num_guards = self.world.numGuards
            num_attackers = self.world.numAttackers
            n_int = np.arange(num_guards+num_attackers)
            bi_hot = self.get_binary_array(n_int, 8)

            self.obs_arr = self.raw_obs_array()
            for guard in self.guards():
                self.obs_arr.append([guard.alive])
                self.obs_arr.append(guard.pos)
                self.obs_arr.append([guard.atk_rad])
                self.obs_arr.append(guard.vel)
                self.obs_arr.append([guard.iden])
                self.obs_arr.append([guard.terrain])
                self.obs_arr.append(bi_hot[guard.iden])
            for attacker in self.attackers():
                self.obs_arr.append([attacker.alive])
                self.obs_arr.append(attacker.pos)
                self.obs_arr.append([attacker.atk_rad])
                self.obs_arr.append(attacker.vel)
                self.obs_arr.append([attacker.iden])
                self.obs_arr.append([attacker.terrain])
                self.obs_arr.append(bi_hot[attacker.iden])
            shit = self.obs_arr.get()
            '''
            from VISUALIZE.mcom import mcom
            from config import GlobalConfig as cfg
            if not hasattr(cfg, 'ak_logger'):
                cfg.ak_logger = mcom(ip='127.0.0.1',
                        port=12084,
                        path='%s/v2d_logger/'%cfg.logdir,
                        digit=16, rapid_flush=True, draw_mode='Native')
                cfg.ak_logger.v2d_init()
            
            self.mcv = cfg.ak_logger
            self.mcv.v2d_clear()
            for index, guard in enumerate(self.guards()):
                self.mcv.v2dx('cir|%d|b|0.04'%(index), guard.pos[0], guard.pos[1])
                if not guard.alive:
                    self.mcv.v2dx('cir|%d|k|0.04'%(index), guard.pos[0], guard.pos[1])
            for index, attacker in enumerate(self.attackers()):
                self.mcv.v2dx('cir|%d|r|0.04'%(index+50), attacker.pos[0], attacker.pos[1])
                if not attacker.alive:
                    self.mcv.v2dx('cir|%d|k|0.04'%(index+50), attacker.pos[0], attacker.pos[1])
            self.mcv.v2d_show()
            '''
            self.new_obs = shit.astype(np.float32)
            self.dec = {'alive':0, 
                        'pos':range(1,3), 
                        'ang':3, 
                        'vel':range(4,6), 
                        'id':6, 
                        'terrain':7, 
                        'bi_hot':range(8, 16)}
            self.obs_range = 2.0
            self.n_object = self.world.numGuards + self.world.numAttackers
            self.obs = my_view(self.new_obs, [self.n_object, -1])
            self.dis = distance_matrix(self.obs[:,self.dec['pos']])
            # set almost inf distance for dead agents
            self.h_alive = np.array([attacker.alive for attacker in self.attackers()])
            self.f_alive = np.array([guard.alive for guard in self.guards()])
            alive_all = np.concatenate((self.f_alive, self.h_alive))
            self.dis[~alive_all,:] = +np.inf
            self.dis[:,~alive_all] = +np.inf
            # 没有考虑智能体是否存活？？？
            guards_uid = range(0,num_guards)
            attackers_uid = range(num_guards,num_attackers+num_guards)
            self.f2h_dis = self.dis[guards_uid, :][:, attackers_uid]
            self.f2f_dis = self.dis[guards_uid, :][:, guards_uid]
            self.agent_emb = self.obs[guards_uid]
            self.hostile_emb = self.obs[attackers_uid]
            self.world.shared_resorce['distance_matrix'] = self.dis
            self.world.shared_resorce['guards_uid'] = np.array(guards_uid)
            self.world.shared_resorce['attackers_uid'] = np.array(attackers_uid)

        A_id = agent.iden
        a2h_dis = self.f2h_dis[A_id]
        a2f_dis = self.f2f_dis[A_id]
        vis_n = 6
        h_iden_sort = np.argsort(a2h_dis)[:vis_n] 
        f_iden_sort = np.argsort(a2f_dis)[:vis_n] 
        # np.random.shuffle(h_iden_sort)
        # np.random.shuffle(f_iden_sort)
        if not agent.alive:
            agent_obs = np.zeros(shape=(self.agent_emb.shape[-1] *vis_n*2,))
            info_n = {'vis_f': None, 'vis_h':None, 'alive': False}
            return agent_obs, info_n


        # observe hostile:: dis array([4, 6, 3, 5, 2, 7])  shuf array([5, 2, 3, 6, 7, 4])
        a2h_dis_sorted = a2h_dis[h_iden_sort]
        hostile_vis_mask = (a2h_dis_sorted<=self.obs_range) & (self.h_alive[h_iden_sort])
        vis_index = h_iden_sort[hostile_vis_mask]
        invis_index = h_iden_sort[~hostile_vis_mask]
        vis_index, invis_index = self.item_random_mv(src=vis_index, dst=invis_index,prob=self.random_jam_prob, rand=True)
        _ind = np.concatenate((vis_index, invis_index))
        _msk = np.concatenate((vis_index<0, invis_index>=0)) # "<0" project to False; ">=0" project to True
        a2h_sort = self.hostile_emb[_ind]
        a2h_sort[_msk] = 0
        a2h_sort_filtered = a2h_sort
        
        a2f_dis_sorted = a2f_dis[f_iden_sort]
        friend_vis_mask = (a2f_dis_sorted<=(self.obs_range*1.5)) & (self.f_alive[f_iden_sort])
        vis_index = f_iden_sort[friend_vis_mask]
        self_index = vis_index[:1]  # 自身的索引
        vis_index = vis_index[1:]  # 可见友方的索引
        invis_index = f_iden_sort[~friend_vis_mask] # 不可见友方的索引
        vis_index, invis_index = self.item_random_mv(src=vis_index, dst=invis_index,prob=self.random_jam_prob, rand=True)
        _ind = np.concatenate((self_index, vis_index, invis_index))
        _msk = np.concatenate((self_index<0, vis_index<0, invis_index>=0)) # "<0" project to False; ">=0" project to True
        a2f_sort = self.agent_emb[_ind]
        a2f_sort[_msk] = 0
        a2f_sort_filtered = a2f_sort

        agent_obs = np.concatenate((a2f_sort_filtered.flatten(), a2h_sort_filtered.flatten()))
        
        '''
        # # 视野内敌我 画连线
        for hid in hostile_vis_id:
            huid = int(hid)+50
            self.mcv.v2d_line_object(A_id,huid)

        # # # 智能体之间 画连线
        for fid in friend_vis_id:
            fuid = int(fid)
            self.mcv.v2d_line_object(A_id,fuid)
        '''

        info_n = {'vis_f': f_iden_sort, 'vis_h':h_iden_sort[a2h_dis_sorted<self.obs_range], 'alive': True}
        return agent_obs, info_n
# @njit
def distance_matrix(A):
    assert A.shape[-1] == 2 # assert 2D situation
    n_subject = A.shape[-2] # is 2
    A = np.repeat(np.expand_dims(A,-2), n_subject, axis=-2) # =>(64, 100, 100, 2)
    At = np.swapaxes(A,-2,-3) # =>(64, 100, 100, 2)
    dis = At-A # =>(64, 100, 100, 2)
    dis = np.linalg.norm(dis, axis=-1)
    return dis