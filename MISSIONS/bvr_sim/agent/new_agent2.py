from typing import List
from agent.agent import Agent
from  env.env_cmd import CmdEnv
from utils.utils_math import TSVector3
import copy
import random
import numpy as np
import time
from .tools import distance_matrix

'''
    y     [-150000, +150000]
    red_x [-150000, -125000]
    blue_x [125000,  150000]
'''
class Special():
    NE = np.array([125000,   125000])
    N = np.array([0,   125000])
    E = np.array([125000,   0])
    NW = np.array([-125000,  125000])
    SE = np.array([125000,  -125000])
    SW = np.array([-125000, -125000])

    init_X = 125000
    init_Y = 0
    init_Z = 9500

    init_speed_vip = 400
    init_speed_drone = 300 * 0.6

    Dive_Horizental = 10000 # 躲导弹时，下潜点的水平距离

def nearby(dir, L):
    dir = dir / (np.linalg.norm(dir)+1e-3) * L


class Drone():
    MaxSpeed = 300
    MinSpeed = 100
    FollowSpeed = 150
    MaxHeight = 10000
    EscapeHeight = 5000
    MinHeight = 2000
    MaxAcc = 2
    MaxOverload = 12

class Vip():
    MaxSpeed = 400
    MinSpeed = 150
    FollowSpeed = 150
    MaxHeight = 15000
    EscapeHeight = 5000
    MinHeight = 2000
    MaxAcc = 1
    MaxOverload = 6




class plane(object):
    def __init__(self, data=None) -> None:
        super().__init__()
        if data is not None:
            self.update_info(data)

    def update_info(self, data):
        self.data = data
        for key in data:
            setattr(self, key, data[key])
            
        if not hasattr(self, 'is_drone'):
            self.is_drone = (self.Type ==2)
            self.is_vip = (self.Type ==1)
            Ability = Drone if self.is_drone else Vip
            for key in Ability.__dict__:
                if '__' in key: continue
                setattr(self, key, getattr(Ability,key))

        # self        
            

    def incoming_msid(self):
        return 0

    def step_init(self):
        self.under_attack = []
        self.state = ''

class NewAgent(Agent):
    def __init__(self, name, config):
        super(NewAgent, self).__init__(name, config["side"])
        self.my_uvas_infos = []
        self.my_manned_info = []
        self.my_allplane_infos = []
        self.enemy_uvas_infos = []
        self.enemy_manned_info = []
        self.enemy_allplane_infos = []
        self.enemy_missile_infos = []
        self.my_missile_infos = []
        self.attack_handle_enemy = {}

        self.n_uav = 4
        self.cmd_list = []
        self.state_recall = {}
        self.STATE = 'Full_Assult'
        self.individual_state = ['Assult']*self.n_uav

        self.my_planes = None

    def get_uav(self):
        return { id_:self.my_planes[id_] for id_ in self.my_planes if self.my_planes[id_].is_drone}

    def get_vip(self):
        return { id_:self.my_planes[id_] for id_ in self.my_planes if self.my_planes[id_].is_vip}

    def init_planes(self):
        self.my_planes = {
             p['ID']:plane(p)   for p in self.my_allplane_infos
        }
        return

    def reset(self, **kwargs):
        self.attack_handle_enemy.clear()
        pass
    
    def step(self,sim_time, obs_side, **kwargs) -> List[dict]:
        self.cmd_list = []
        self.process_observation_and_show(obs_side)
        if self.my_planes is None:
            self.init_planes()

        self.process_decision(sim_time, obs_side)
        return copy.deepcopy(self.cmd_list)      

    def process_decision(self, time, obs_side):
        # self.show()
        if time == 3: self.init_pos(self.cmd_list); self.print_()
        elif time > 5: self.make_decision()
        return

    def make_decision(self):
        if self.STATE == 'Full_Assult':
            self.Full_Assult()


    def Full_Assult(self):
        '''
            cmd_list.append(CmdEnv.make_attackparam(plane_info['ID'], close_target['enemy_data']['ID'], 1))
            cmd_list.append();
            cmd_list.append(CmdEnv.make_followparam(plane['ID'], enemy_leader['ID'], data['move_max_speed'], data['move_max_acc'], data['move_max_g']))
        '''


        # my_uav_id = [ p['ID']   for p in self.my_uvas_infos    ]
        # op_uav_id = [ p['ID']   for p in self.enemy_uvas_infos ]

        my_uav = self.get_uav()
        my_vip = self.get_vip()

        for plane_info in self.my_allplane_infos:
            plane_id = plane_info['ID']
            self.my_planes[plane_id].step_init()
            self.my_planes[plane_id].update_info(plane_info)

        everything = self.my_allplane_infos + self.my_missile_infos + self.enemy_allplane_infos + self.enemy_missile_infos 
        everything_pos = np.array([ np.array([p['X'],p['Y'],p['Z']])  for p in everything ])
        self.id2index = { p['ID']:i for i,p in enumerate(everything)}
        self.everything_dis = distance_matrix(everything_pos)

        def id2index(arr):
            return [self.id2index[id_] for id_ in arr]

        def get_dis(id1,id2):
            index1 = self.id2index[id1]
            index2 = self.id2index[id2]
            return self.everything_dis[index1, index2]


        if 'ID_Bind' not in self.state_recall:
            self.state_recall['ID_Bind'] = { p['ID']:self.enemy_uvas_infos[i]['ID']   for i,p in enumerate(self.my_uvas_infos) }

        # deal ms
        for ms in self.enemy_missile_infos:
            my_plane_targeted = ms['EngageTargetID']
            self.my_planes[my_plane_targeted].under_attack.append(ms)

        # state change
        for pid in my_uav:
            p = my_uav[pid]
            if len(p.under_attack) >= 1:
                p.state = 'escape'

        def follow_cmd(uav):
            self.cmd_list.append(
                CmdEnv.make_followparam(
                    uav.ID,
                    self.state_recall['ID_Bind'][uav.ID],
                    uav.FollowSpeed,
                    uav.MaxAcc, 
                    uav.MaxOverload))

        def escape_cmd(uav):
            # Dst = nearby(Special.NE, Special.Dive_Hori/zental)
            goto_location = [
                {
                    "X": Special.E[0], 
                    "Y": Special.E[1],
                    "Z": uav.EscapeHeight
                }
            ]
            self.cmd_list.append(
                CmdEnv.make_linepatrolparam(
                    uav.ID,
                    goto_location, 
                    uav.MaxSpeed,
                    uav.MaxAcc, 
                    uav.MaxOverload
                )
            )

        for pid in my_uav:
            p = my_uav[pid]
            if p.state == '':
                follow_cmd(p)
            elif p.state == 'escape':
                escape_cmd(p)
        return 

    def print_(self):
        return 

   
    def process_observation_and_show(self, obs_side):
        my_entity_infos = obs_side['platforminfos'] # 拿到己方阵营有人机、无人机在内的所有飞机信息
        if len(my_entity_infos) < 1:
            return
        my_manned_info = [] # 用以保存当前己方有人机信息
        my_uvas_infos = []  # 用以保存当前己方无人机信息
        my_allplane_infos = []    # 用以保存当前己方所有飞机信息
        for uvas_info in my_entity_infos:
            if uvas_info['ID'] != 0 and uvas_info['Availability'] > 0.0001: # 判断飞机是否可用 飞机的ID即为飞机的唯一编号 飞机的Availability为飞机当前生命值
                uvas_info["Z"] = uvas_info["Alt"]    
                if uvas_info['Type'] == 1:           
                    my_manned_info.append(uvas_info) 
                if uvas_info['Type'] == 2:           
                    my_uvas_infos.append(uvas_info)  
                my_allplane_infos.append(uvas_info)  
        if len(my_manned_info) < 1:      
            return
        enemy_entity_infos = obs_side['trackinfos']   # 拿到敌方阵营的飞机信息,包括敌方有人机、无人机在内的所有飞机信息
        enemy_manned_info = []  # 用以保存当前敌方有人机信息
        enemy_uvas_infos = []   # 用以保存当前敌方无人机信息
        enemy_allplane_infos = []     # 用以保存当前敌方所有飞机信息
        for uvas_info in enemy_entity_infos:
            if uvas_info['ID'] != 0 and uvas_info['Availability'] > 0.0001:  # 判断飞机是否可用 飞机的ID即为飞机的唯一编号 飞机的Availability为飞机当前生命值
                uvas_info['Z'] = uvas_info['Alt']         # 飞机的 Alt 即为飞机的当前高度
                if uvas_info['Type'] == 1:               # 所有类型为 1 的飞机是 有人机
                    enemy_manned_info.append(uvas_info)  # 将有人机保存下来 一般情况，每方有人机只有1架
                if uvas_info['Type'] == 2:               # 所有类型为 2 的飞机是 无人机
                    enemy_uvas_infos.append(uvas_info)   # 将无人机保存下来 一般情况，每方无人机只有4架
                enemy_allplane_infos.append(uvas_info)         # 将己方所有飞机信息保存下来 一般情况，每方飞机实体总共5架
        my_allplane_maps = {}
        for input_entity in my_allplane_infos:
            my_allplane_maps[int(input_entity['ID'])] = input_entity
        missile_infos = obs_side['missileinfos']  # 拿到空间中已发射且尚未爆炸的导弹信息
        enemy_missile_infos = [] #  用以保存敌方已发射且尚未爆炸的导弹信息
        my_missile_infos = [] #  用以保存敌方已发射且尚未爆炸的导弹信息
        for missile_info in missile_infos:
            if missile_info['LauncherID'] in my_allplane_maps:#  判断导弹是否为己方导弹 导弹的LauncherID即为导弹的发射者
                missile_info["Z"] = missile_info["Alt"]     # 导弹的 Alt 即为导弹的当前高度
                my_missile_infos.append(missile_info)    # 保存敌方已发射且尚未爆炸的导弹信息
            if (missile_info['ID'] != 0 and missile_info['Availability'] > 0.0001): # 判断导弹是否可用 导弹的ID即为导弹的唯一编号 导弹的Availability为导弹当前生命值
                missile_info["Z"] = missile_info["Alt"]     # 导弹的 Alt 即为导弹的当前高度
                enemy_missile_infos.append(missile_info)    # 保存敌方已发射且尚未爆炸的导弹信息
        self.my_uvas_infos = my_uvas_infos                  # 保存当前己方无人机信息
        self.my_manned_info = my_manned_info                # 保存当前己方有人机信息
        self.my_allplane_infos = my_allplane_infos          # 保存当前己方所有飞机信息
        self.enemy_uvas_infos = enemy_uvas_infos            # 保存当前敌方无人机信息
        self.enemy_manned_info = enemy_manned_info          # 保存当前敌方有人机信息
        self.enemy_allplane_infos = enemy_allplane_infos    # 保存当前敌方所有飞机信息
        self.enemy_missile_infos = enemy_missile_infos      # 保存敌方已发射且尚未爆炸的导弹信息
        self.my_missile_infos = my_missile_infos            # 保存敌方已发射且尚未爆炸的导弹信息
        # if not hasattr(self, 'mcv'):
        #     from .VISUALIZE.mcom import mcom
        #     self.mcv = mcom(ip='127.0.0.1',
        #                 port=12084,
        #                 path='./TMP/v2d_logger/',
        #                 digit=16, rapid_flush=True, draw_mode='pyqtgraph')
        #     self.mcv.v2d_init()
        # id_ = 0
        # for uav in my_uvas_infos:
        #     self.mcv.v2dx('rec|%d|r|2000'%(id_), uav['X'], uav['Y'], 0);id_ += 1
        # for uav in my_manned_info:
        #     self.mcv.v2dx('cir|%d|r|4000'%(id_), uav['X'], uav['Y'], 0);id_ += 1
        # for uav in enemy_uvas_infos:
        #     self.mcv.v2dx('rec|%d|b|2000'%(id_), uav['X'], uav['Y'], 0);id_ += 1
        # for uav in enemy_manned_info:
        #     self.mcv.v2dx('cir|%d|b|4000'%(id_), uav['X'], uav['Y'], 0);id_ += 1
        # self.mcv.v2d_show()
        # time.sleep(0.1)






    def init_pos(self, cmd_list):
        '''
            y     [-150000, +150000]
            red_x [-150000, -125000]
            blue_x [125000,  150000]
        '''
        leader_original_pos = {}    # 用以初始化当前方的位置
        if self.name == "red":
            leader_original_pos = {"X": -Special.init_X, "Y": -Special.init_Y, "Z": Special.init_Z}
        else :
            leader_original_pos = {"X":  Special.init_X, "Y":  Special.init_Y, "Z": Special.init_Z}

        interval_distance = 5000   # 间隔 5000米排列
        for leader in self.my_manned_info: # 为己方有人机设置初始位置
            # CmdEnv.make_entityinitinfo 指令，可以将 飞机实体置于指定位置点，参数依次为 实体ID，X坐标，Y坐标，Z坐标，初始速度，初始朝向
            cmd_list.append(
                CmdEnv.make_entityinitinfo(leader['ID'], leader_original_pos['X'], leader_original_pos['Y'], 
                leader_original_pos['Z'], Special.init_speed_vip, 45))

        #己方无人机在有人机的y轴上分别以9500的间距进行部署
        sub_index = 0  # 编号 用以在有人机左右位置一次排序位置点
        for sub in self.my_uvas_infos: # 为己方每个无人机设置初始位置
            sub_pos = copy.deepcopy(leader_original_pos)  # 深拷贝有人机的位置点
            if sub_index & 1 == 0: # 将当前编号放在有人机的一侧
                # CmdEnv.make_entityinitinfo 指令，可以将 飞机实体置于指定位置点，参数依次为 实体ID，X坐标，Y坐标，Z坐标，初始速度，初始朝向
                cmd_list.append(
                    CmdEnv.make_entityinitinfo(sub['ID'], 
                    sub_pos['X'], sub_pos['Y'] + interval_distance, 
                    sub_pos['Z'], Special.init_speed_drone, 45))
            else:                   # 将当前编号放在有人机的另一侧
                # CmdEnv.make_entityinitinfo 指令，可以将 飞机实体置于指定位置点，参数依次为 实体ID，X坐标，Y坐标，Z坐标，初始速度，初始朝向
                cmd_list.append(
                    CmdEnv.make_entityinitinfo(sub['ID'], 
                    sub_pos['X'], sub_pos['Y'] - interval_distance, 
                    sub_pos['Z'], Special.init_speed_drone, 45))
                interval_distance *= 2 # 编号翻倍
            sub_index += 1 # 编号自增


