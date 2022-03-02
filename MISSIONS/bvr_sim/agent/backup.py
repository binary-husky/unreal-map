from typing import List
from agent.agent import Agent
from  env.env_cmd import CmdEnv
from utils.utils_math import TSVector3
from .UTILS.colorful import *
from .UTILS.tensor_ops import dir2rad, np_softmax

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
    Dive_Z = 5000

def nearby(dir, L):
    dir = dir / (np.linalg.norm(dir)+1e-3) * L


class Drone():
    MaxSpeed = 300
    MinSpeed = 100
    FollowSpeed = 300
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












class Plane(object):
    dis_mat = None
    dis_mat_id2index = None
    # self.all_ms = []
    def __init__(self, data=None) -> None:
        super().__init__()
        if data is not None: self.update_info(data)
        self.persistent_state = 'follow'

    def update_info(self, data):
        self.alive = (data['Availability'] != 0)
        self.data = data
        if hasattr(self, 'ID'): 
            assert data['ID'] == self.ID
        for key in data:
            setattr(self, key, data[key])

        if not hasattr(self, 'is_drone'):
            self.is_drone = (self.Type ==2)
            self.is_vip = (self.Type ==1)
            Ability = Drone if self.is_drone else Vip
            for key in Ability.__dict__:
                if '__' in key: continue
                setattr(self, key, getattr(Ability,key))
        self.pos3d = np.array([self.X,self.Y,self.Alt], dtype=float)
        self.pos2d = np.array([self.X,self.Y], dtype=float)

    def incoming_msid(self):
        return 0

    def step_init(self):
        self.attacked_by_ms = []
        self.my_ms = []
        self.step_state = ''
        self.alive = False # 如果没有被置 True....

    def get_dis(self, target_id):
        index1 = Plane.dis_mat_id2index[self.ID]
        index2 = Plane.dis_mat_id2index[target_id]
        return Plane.dis_mat[index1, index2]

    def delta_oppsite_to_ms(self):
        delta_pos = np.array([ self.pos3d[:2] - np.array([ms['X'],ms['Y']], dtype=float)
                        for ms in self.attacked_by_ms ])
        dis = np.array([ self.get_dis(ms['ID']) for ms in self.attacked_by_ms ])
        delta_dir = delta_pos / dis
        weight = np_softmax(dis.max() - dis)
        delta = (delta_dir*weight).sum(0)
        # math_rad_dir = dir2rad(delta)
        return delta/ np.linalg.norm(delta)



class MS(object):
    def __init__(self, data) -> None:
        super().__init__()
        if data is not None: self.update_info(data)
        self.flying_time = 0

    def update_info(self, data):
        self.alive = (data['Availability'] != 0)
        self.data = data
        if hasattr(self, 'ID'): 
            assert data['ID'] == self.ID
        for key in data:
            setattr(self, key, data[key])

        self.pos3d = np.array([self.X,self.Y,self.Alt], dtype=float)
        self.pos2d = np.array([self.X,self.Y], dtype=float)

    def step_init(self):
        self.flying_time += 1
        self.still_alive = False




class DebugAgentBlue(Agent):
    def __init__(self, name, config):
        super(DebugAgentBlue, self).__init__(name, config["side"])
        self.init()

    def init(self):
        self.n_uav = 4
        self.cmd_list = []
        self.state_recall = {}
        self.STATE = 'Full_Assult'
        self.individual_state = ['Assult']*self.n_uav
        self.id_mapping = None
        self.Name_mapping = None

        self.my_planes = None
        self.op_planes = None

        self.ms = []


    def find_plane_by_name(self, name):
        for p in self.my_planes + self.op_planes:
            if p.Name == name: return p
        return None

    def find_plane_by_id(self, ID):
        for p in self.my_planes + self.op_planes:
            if p.ID == ID: return p
        return None

    def find_ms_by_id(self, ID):
        for ms in self.ms + self.ms:
            if ms.ID == ID: return ms
        return None



    def reset(self, **kwargs):
        self.init()
        pass
    
    def my_process_observation_and_show(self, obs_side):
        # need init?
        if self.my_planes is None:
            self.my_planes = [Plane() for _ in range(self.n_uav +1)]
            self.op_planes = [Plane() for _ in range(self.n_uav +1)]

            # register info for the first time!
            for idx, uvas_info in  enumerate(obs_side['platforminfos']):
                self.my_planes[idx].update_info(uvas_info)
            for idx, uvas_info in  enumerate(obs_side['trackinfos']):
                self.op_planes[idx].update_info(uvas_info)
            
            if len(obs_side['platforminfos']) != (self.n_uav +1):
                print亮红('Error! 没有从起始状态执行！可能导致一系列错误！')
                self.my_planes = list(filter(lambda p: hasattr(p, 'ID'), self.my_planes))

            if len(obs_side['trackinfos']) != (self.n_uav +1):
                print亮红('Error! 没有从起始状态执行！可能导致一系列错误！')
                self.op_planes = list(filter(lambda p: hasattr(p, 'ID'), self.op_planes))
            # self.update_mapping()
        # set every plane object to step_init
        for obj in self.my_planes + self.op_planes + self.ms:
            obj.step_init()

        everything = []
        # part 1
        my_entity_infos = obs_side['platforminfos']
        enemy_entity_infos = obs_side['trackinfos']
        if len(my_entity_infos) < 1: return
        for uvas_info in (my_entity_infos+enemy_entity_infos):
            if not (uvas_info['ID'] != 0 and uvas_info['Availability'] > 0.0001): continue
            uvas_info["Z"] = uvas_info["Alt"]
            p = self.find_plane_by_id(uvas_info['ID'])
            if p is None: 
                continue
            p.update_info(uvas_info)
            everything.append(uvas_info)

        # part 2
        missile_infos = obs_side['missileinfos']
        for missile_info in missile_infos:
            ms_active = (missile_info['ID'] != 0 and missile_info['Availability'] > 0.0001)
            if not ms_active: continue
            missile_info["Z"] = missile_info["Alt"]
            host_id = missile_info['LauncherID']
            target_id = missile_info['EngageTargetID']
            self.find_plane_by_id(host_id).my_ms.append(missile_info)
            self.find_plane_by_id(target_id).attacked_by_ms.append(missile_info)
            ms = self.find_ms_by_id(missile_info["ID"])
            if ms is None: self.ms.append(MS(missile_info))
            else: ms.update_info(missile_info)
            everything.append(missile_info)

        # part 3 distance matrix
        everything_pos = np.array([ np.array([p['X'],p['Y'],p['Z']])  for p in everything ])
        self.id2index = { p['ID']:i for i,p in enumerate(everything)}
        self.active_index = self.id2index
        self.everything_dis = distance_matrix(everything_pos)
        Plane.dis_mat = self.everything_dis
        Plane.dis_mat_id2index = self.id2index

        for p in self.my_planes:
            if not p.alive: 
                print亮蓝('Our Plane is Destoried:', p.ID)
        for p in self.op_planes:
            if not p.alive: 
                print亮红('Hostile Plane is Destoried:', p.ID)

        self.my_planes = list(filter(lambda p: p.alive, self.my_planes))
        self.op_planes = list(filter(lambda p: p.alive, self.op_planes))
        # self.update_mapping()



    def id2index(self, arr):
        return [self.id2index[id_] for id_ in arr]

    def get_dis(self, id1,id2):
        index1 = self.id2index[id1]
        index2 = self.id2index[id2]
        return self.everything_dis[index1, index2]

    def step(self,sim_time, obs_side, **kwargs) -> List[dict]:
        self.my_process_observation_and_show(obs_side)
        self.cmd_list = []
        self.process_decision(sim_time, obs_side)
        return copy.deepcopy(self.cmd_list)      

    def process_decision(self, time, obs_side):
        # self.show()
        if time == 3: self.init_pos(self.cmd_list); self.print_()
        elif time > 3: self.make_decision()
        return

    def make_decision(self):
        A_uav = self.my_planes[0]  
        if not A_uav.is_drone: A_uav = self.my_planes[1]
        # my_vip = self.get_vip()

        if not "蓝无人机1" in A_uav.Name: return
        A_uav.bind = self.find_plane_by_name('红无人机1').ID

        # state change
        for p in self.my_planes:
            if len(p.attacked_by_ms) >= 1:
                dis = [ self.get_dis(p.ID, ms['ID']) for ms in p.attacked_by_ms ]
                min_ms_dis = min(dis)
                if min_ms_dis <= 20e3: # 55 不可逃逸， 56 可逃逸
                    p.step_state = 'escape'
                    if min_ms_dis <= 5*1000: #2e3: # 2000ok 1500ok
                        p.step_state = 'emergent_escape'
                        print亮靛('emergent_escape')

                    for ms_dict in p.attacked_by_ms:
                        ms = self.find_ms_by_id(ms_dict['ID']) 
                        print亮紫('ID %d, speed %.2f, Availability %.2f, flying time %.2f, (z: %.2f)'%(ms.ID, ms.Speed, ms.Availability, ms.flying_time, ms.Z))
                        ft = ms.flying_time
                        break
                    print亮绿(p.Z, ft, min_ms_dis)
                    pass
                    
                    
        def follow_cmd(uav):
            follow = hasattr(uav, 'bind') and (uav.bind is not None)
            if not follow: return
            self.cmd_list.append(
                CmdEnv.make_followparam(
                    uav.ID,
                    uav.bind,
                    uav.FollowSpeed,
                    uav.MaxAcc, 
                    uav.MaxOverload))
            return
        def emergent_escape_cmd(uav):
            if uav.persistent_state == 'emergent_escape': return
            uav.persistent_state = 'emergent_escape'
            unit_delta = uav.delta_oppsite_to_ms()
            unit_delta = np.append(unit_delta,0)
            unit_delta = np.cross(unit_delta, np.array([0,0,1]))
            H2 = unit_delta[:2]*100e3 + uav.pos2d
            def check_dis(goto_location):
                d = goto_location[0]
                dis =np.linalg.norm(uav.pos3d - np.array([d['X'],d['Y'],d['Z']]))
                assert dis > 10e3
            goto_location = [
                {
                    "X": H2[0], 
                    "Y": H2[1],
                    "Z": uav.Z
                }
            ]
            check_dis(goto_location)
            self.cmd_list.append(
                CmdEnv.make_linepatrolparam(
                    uav.ID,
                    goto_location, 
                    uav.MaxSpeed,
                    uav.MaxAcc, 
                    uav.MaxOverload
                )
            )


        def escape_cmd(uav):
            ## STATE_TRANS
            if uav.persistent_state == 'escaping': return
            uav.persistent_state = 'escaping'
            # Dst = nearby(Special.NE, Special.Dive_Hori/zental)
            unit_delta = uav.delta_oppsite_to_ms()
            # Special.Dive_Z = uav.Z
            # dive_z = Special.Dive_Z if uav.Z>Special.Dive_Z else uav.Z
            # dz = uav.Z - Special.Dive_Z
            # dz = 0
            # r = 0.5
            # dh = max(dz * r, 10e3)
            # H = unit_delta*dh + uav.pos2d
            H2 = unit_delta*100e3 + uav.pos2d
            def check_dis(goto_location):
                d = goto_location[0]
                dis =np.linalg.norm(uav.pos3d - np.array([d['X'],d['Y'],d['Z']]))
                assert dis > 10e3
            goto_location = [
                # {
                #     "X": H[0], 
                #     "Y": H[1],
                #     "Z": dive_z
                # },
                {
                    "X": H2[0], 
                    "Y": H2[1],
                    "Z": uav.Z
                }
            ]
            check_dis(goto_location)
            self.cmd_list.append(
                CmdEnv.make_linepatrolparam(
                    uav.ID,
                    goto_location, 
                    uav.MaxSpeed,
                    uav.MaxAcc, 
                    uav.MaxOverload
                )
            )
        
        # 任务优先级

        for p in self.my_planes:
            if p.step_state == '':
                follow_cmd(p)
            elif p.step_state == 'escape':
                escape_cmd(p)
            elif p.step_state == 'emergent_escape':
                emergent_escape_cmd(p)

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
            elif (missile_info['ID'] != 0 and missile_info['Availability'] > 0.0001): # 判断导弹是否可用 导弹的ID即为导弹的唯一编号 导弹的Availability为导弹当前生命值
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



    def init_pos(self, cmd_list):
        '''
            y     [-150000, +150000]
            red_x [-150000, -125000]
            blue_x [125000,  150000]
        '''
        leader_original_pos = {}    # 用以初始化当前方的位置
        if self.name == "red":
            leader_original_pos = {"X": -Special.init_X, "Y": -Special.init_Y, "Z": Special.init_Z}
            init_dir = 90
        else :
            leader_original_pos = {"X":  Special.init_X, "Y":  Special.init_Y, "Z": Special.init_Z}
            init_dir = 360-90

        interval_distance = 5000   # 间隔 5000米排列
        sub_index = 0
        for p in self.my_planes:
            if p.is_vip:
                cmd_list.append(
                    CmdEnv.make_entityinitinfo(
                        p.ID, 
                        leader_original_pos['X'], leader_original_pos['Y'], leader_original_pos['Z'], 
                        Special.init_speed_vip, init_dir))
            elif p.is_drone:
                sub_pos = copy.deepcopy(leader_original_pos)
                if sub_index & 1 == 0:
                    cmd_list.append(
                    CmdEnv.make_entityinitinfo(p.ID, 
                        sub_pos['X'], sub_pos['Y'] + interval_distance, 
                        sub_pos['Z'], Special.init_speed_drone, init_dir))
                else:
                    cmd_list.append(
                        CmdEnv.make_entityinitinfo(p.ID, 
                        sub_pos['X'], sub_pos['Y'] - interval_distance, 
                        sub_pos['Z'], Special.init_speed_drone, init_dir))
                    interval_distance *= 2 # 编号翻倍
                sub_index += 1
            else:
                assert False, ('???')


'''
        def escape_cmd(uav):
            ## STATE_TRANS
            if uav.persistent_state == 'escaping': return
            uav.persistent_state = 'escaping'
            # Dst = nearby(Special.NE, Special.Dive_Hori/zental)
            unit_delta = uav.delta_oppsite_to_ms()
            # Special.Dive_Z = uav.Z
            dive_z = Special.Dive_Z if uav.Z>Special.Dive_Z else uav.Z
            dz = uav.Z - Special.Dive_Z
            dz = 0
            r = 0.5
            dh = max(dz * r, 10e3)
            H = unit_delta*dh + uav.pos2d
            H2 = unit_delta*150e3 + uav.pos2d
            def check_dis(goto_location):
                d = goto_location[0]
                dis =np.linalg.norm(uav.pos3d - np.array([d['X'],d['Y'],d['Z']]))
                assert dis > 10e3
            goto_location = [
                {
                    "X": H[0], 
                    "Y": H[1],
                    "Z": dive_z
                },
                {
                    "X": H2[0], 
                    "Y": H2[1],
                    "Z": dive_z
                }
            ]
            check_dis(goto_location)
            assert goto_location
            self.cmd_list.append(
                CmdEnv.make_linepatrolparam(
                    uav.ID,
                    goto_location, 
                    uav.MaxSpeed,
                    uav.MaxAcc, 
                    uav.MaxOverload
                )
            )
        

'''