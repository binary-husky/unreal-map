from typing import List
from ..agent import Agent
from ..env_cmd import CmdEnv
from .UTILS.colorful import *
from .UTILS.tensor_ops import dir2rad, np_softmax, reg_deg_at, repeat_at
from .maneuver import maneuver_cold_to_ms, maneuver_vertical_to_ms, maneuver_angle_to_ms
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
# 定义了地图中一些特殊点和量
class Special():
    NE = np.array([125000, 125000])
    N = np.array([0, 125000])
    E = np.array([125000, 0])
    NW = np.array([-125000, 125000])
    SE = np.array([125000, -125000])
    SW = np.array([-125000, -125000])

    init_X = 125000
    init_Y = 0
    init_Z = 9900

    init_speed_vip = 400
    init_speed_drone = 300 * 0.6

    # Dive_Horizental = 10000  # 躲导弹时，下潜点的水平距离
    Dive_Z = 5000

# 无人机的参量
class Drone():
    MIN_X = -145e3
    MAX_X = 145e3
    MIN_Y = -145e3
    MAX_Y = 145e3
    
    MaxSpeed = 300
    MinSpeed = 100
    FollowSpeed = 300
    MaxHeight = 10000
    EscapeHeight = 5000
    MinHeight = 2000
    MaxAcc = 2
    MaxOverload = 12

    RadarHorizon = 30
    RadarVertical = 10
    RadarDis = 39e3

    AttackRadarHorizon = 20
    AttackRadarVertical = 10
    AttackRadarDis = 45e3

    DeadZooRadarHorizon = 12
    DeadZooRadarVertical = 12
    DeadZooRadarDis = 20e3

    prepare_escape_distance = 12e3
    escape_angle = 180

    Flying_to_distance = 55e3
    DeadZooRadarHorizon = 3


# 有人机的参量
class Vip():
    MIN_X = -145e3
    MAX_X = 145e3
    MIN_Y = -145e3
    MAX_Y = 145e3

    MaxSpeed = 400
    MinSpeed = 150
    FollowSpeed = 400
    MaxHeight = 15000
    EscapeHeight = 5000
    MinHeight = 2000
    MaxAcc = 1
    MaxOverload = 6
    RadarHorizon = 60
    RadarVertical = 60
    RadarDis = 58e3

    AttackRadarHorizon = 50
    AttackRadarVertical = 50
    AttackRadarDis = 60e3

    DeadZooRadarHorizon = 50
    DeadZooRadarVertical = 50
    DeadZooRadarDis = 20e3

    prepare_escape_distance = 30e3
    escape_distance = 1.8e3

    Flying_to_distance = 55e3
    DeadZooRadarHorizon = 3

    # emergent_escape_close_distance = 1.3e3


# 飞机的类
class Plane(object):
    dis_mat = None
    dis_mat_id2index = None
    # self.all_ms = []
    def __init__(self, data=None, manager=None) -> None:
        self.manager = manager
        super().__init__()
        if data is not None: self.update_info(data, init=True)
        self.persistent_state = 'follow'
        self.advantage_state = 'default'
        # self.hanging_attack_by = []
        self.attacked_by_ms = []
        self.previous_ms_list = []

    def update_info(self, data, init=False):
        self.alive = (data['Availability'] != 0)
        self.data = data
        if hasattr(self, 'ID'):
            assert data['ID'] == self.ID
        for key in data:
            setattr(self, key, data[key])

        if not hasattr(self, 'is_drone'):
            self.is_drone = (self.Type == 2)
            self.is_vip = (self.Type == 1)
            Ability = Drone if self.is_drone else Vip
            for key in Ability.__dict__:
                if '__' in key: continue
                setattr(self, key, getattr(Ability, key))
        self.pos3d = np.array([self.X, self.Y, self.Alt], dtype=float)
        self.pos2d = np.array([self.X, self.Y], dtype=float)
        self.h_angle = 90-self.Heading*180/np.pi

        if init and ('LeftWeapon' not in data):
            self.OpLeftWeapon = 4 if self.is_vip else 2

    def incoming_msid(self):
        return 0

    def step_init(self):
        self.previous_ms_list = copy.copy(self.attacked_by_ms)
        self.attacked_by_ms = []
        self.wish_list_by = []
        self.my_ms = []
        self.fired_ms = []  # 发射出去，正在飞行的导弹
        self.step_state = ''
        self.alive = False  # 如果没有被置 True....

    def get_dis(self, target_id):
        index1 = Plane.dis_mat_id2index[self.ID]
        index2 = Plane.dis_mat_id2index[target_id]
        return Plane.dis_mat[index1, index2]

    def delta_oppsite_to_ms(self):
        delta_pos = np.array([self.pos3d[:2] - np.array([ms['X'], ms['Y']], dtype=float)
                              for ms in self.attacked_by_ms])
        dis = np.array([self.get_dis(ms['ID']) for ms in self.attacked_by_ms])
        delta_dir = delta_pos / repeat_at(dis, -1, 2)
        weight = np_softmax(dis.max() - dis)
        delta = (delta_dir * repeat_at(weight, -1, 2)).sum(0)
        # math_rad_dir = dir2rad(delta)
        return delta / (np.linalg.norm(delta)+1e-7)

    # 获取距离最近的导弹是哪个
    def nearest_ms(self):
        # 距离最近导弹在 self.attacked_by_ms 中的index
        if len(self.attacked_by_ms)==0: return None
        index = np.argmin(np.array([ self.get_dis(ms['ID']) for ms in self.attacked_by_ms ]))
        return self.manager.find_ms_by_id(self.attacked_by_ms[index]['ID'])

    def ms_number_changed(self):
        return len(self.previous_ms_list) != len(self.attacked_by_ms)

    def pos2d_prediction(self):
        heading_rad = self.h_angle*np.pi/180
        return self.pos2d + 10e3*np.array([np.cos(heading_rad), np.sin(heading_rad)])

    def get_nearest_op_with_ms(self):
        dis = np.array([self.get_dis(op.ID) for op in self.manager.op_planes if op.OpLeftWeapon>0])
        if len(dis)>0:
            return self.manager.op_planes[np.argmin(dis)]
        else:
            return None

    def get_nearest_threat(self):
        res = None
        threat_dis = 9999e3
        ms = self.nearest_ms()
        nearest_op = self.get_nearest_op_with_ms()
        if ms is not None and threat_dis > self.get_dis(ms.ID):
            res = ms
            threat_dis = self.get_dis(ms.ID)
        if nearest_op is not None and threat_dis > self.get_dis(nearest_op.ID):
            res = nearest_op
            threat_dis = self.get_dis(nearest_op.ID)
        return res, threat_dis

    def 调试_正在被高度优势vip导弹追踪_且距离小于25km_估计撞击点预计在万米之上(self):
        # 调用该函数的前提条件是 1、调用者是无人机, 2、没有任何导弹在飞机的12km范围内
        assert self.is_drone
        ms = self.nearest_ms()
        if len(self.attacked_by_ms)==0: return False
        if ms is None: return False
        if not ms.tracking_target: return False
        if (ms.host is not None) and ms.host.is_vip: # 正在被vip打
            if ms.Z > 10e3: # 超出uav的飞行高度
                dis = self.manager.get_dis(ms.ID, self.ID)
                if dis < 25e3: # 距离小于25km
                    # 估计撞击点
                    撞击点高度 = self.Z + 0.25*(ms.Z - self.Z)
                    if 撞击点高度 > 9500 or hasattr(ms, '估计撞击点预计在万米之上_特殊规避'):
                        ms.估计撞击点预计在万米之上_特殊规避 = True
                        return True
                    else:
                        return False
        return False

    def 调试_正在被高度优势vip导弹追踪_sudden_dive_towards_goto_location(self):
        ms = self.nearest_ms()
        assert self.manager.get_dis(ms.ID, self.ID) > 10e3
        vec2ms_2d = ms.pos2d - self.pos2d
        vec2ms_2d_unit_direction = vec2ms_2d / (np.linalg.norm(vec2ms_2d)+1e-7)
        vec2ms_2d_1km = vec2ms_2d_unit_direction*3e3    # 1km 长度的方向向量

        return [{"X": self.X+vec2ms_2d_1km[0], "Y":self.Y+vec2ms_2d_1km[1], "Z": self.MinHeight}]

# 导弹的类
class MS(object):
    def __init__(self, data) -> None:
        super().__init__()

        # import os
        # folder = './log'
        # if not os.path.exists(folder):  #判断是否存在文件夹如果不存在则创建为文件夹
        #     os.makedirs(folder)

        self.previous_pos3d = None
        self.previous_speed = None
        self.ms_at_terminal = False     # 导弹是否进入末制导
        self.tracking_target = True     # 导弹是否在跟踪目标
        self.delta_traj = []
        self.target_speed = []
        self.self_speed = []
        self.distance = []
        self.time = []
        self.delta_dis = []
        self.ms_speed = []
        self.flying_time = 0
        self.previous_change_taking_effect = 0
        self.D_list = [-842.194, -940.260, -1038.325, -1136.391, -1234.457, -1298.654, -1299.993, -1299.993, -1299.993,
                       -1299.993, -1299.992, -1299.992, -1299.992, -1299.992, -1299.991, -1299.991, -1299.991,
                       -1299.991, -1299.991, -1299.991, -1299.991, -1299.991, -1299.990, -1299.990, -1299.990,
                       -1299.990, -1299.990, -1299.990, -1299.990, -1294.523, -1284.720, -1275.109, -1265.683,
                       -1256.438, -1247.368, -1238.469, -1229.735, -1221.163, -1212.747, -1204.484, -1196.369,
                       -1188.398, -1180.568, -1172.875, -1165.315, -1157.884, -1150.579,
                       -1139.507, -1132.512, -1125.633, -1118.867, -1112.21, -1105.661, -1099.217, -1092.875, -1086.633,
                       -1080.489, -1074.44, -1068.484, -1062.6190000000001, -1056.8429999999998, -1051.153, -1045.549,
                       -1040.027, -1034.587, -1029.2269999999999, -1023.944, -1018.737, -1013.604, -1008.544, -1003.556,
                       -998.637, -993.786, -989.003, -984.285, -979.631, -975.04, -970.511, -966.042, -961.632,
                       -957.281, -952.986, -948.747, -944.563, -940.432, -936.354, -932.328, -928.352, -924.426,
                       -920.549, -916.719, -912.937, -909.2, -905.509, -901.863, -898.26, -894.7, -891.182, -887.706,
                       -884.27, -880.874, -877.518, -874.199, -870.919, -867.677, -864.47, -861.3, -858.165, -855.066,
                       -852.0, -848.968, -845.969, -843.003, -840.069, -837.166, -834.294, -831.453, -828.642, -825.861]
        self.speed_list = [596.133, 694.199, 792.266, 890.332, 988.399, 1000.000, 1000.000, 1000.000, 1000.000,
                           1000.000, 1000.000, 1000.000, 1000.000, 1000.000, 1000.000, 1000.000, 1000.000, 1000.000,
                           1000.000, 1000.000, 1000.000, 1000.000, 1000.000, 1000.000, 1000.000, 1000.000, 1000.000,
                           1000.000, 999.000, 989.109, 979.412, 969.903, 960.578, 951.430, 942.454, 933.647, 925.003,
                           916.517, 908.186, 900.005, 891.970, 884.077, 876.323, 868.703, 861.216, 853.856, 846.621,
                           839.507, 832.512, 825.633, 818.867, 812.210, 805.661, 799.217, 792.875, 786.633, 780.489,
                           774.440, 768.484, 762.619, 756.843, 751.153, 745.549, 740.027, 734.587, 729.227, 723.944,
                           718.737, 713.604, 708.544, 703.556, 698.637, 693.786, 689.003, 684.285, 679.631, 675.040,
                           670.511, 666.042, 661.632, 657.281, 652.986, 648.747, 644.563, 640.432, 636.354, 632.328,
                           628.352, 624.426, 620.549, 616.719, 612.937, 609.200, 605.509, 601.863, 598.260, 594.700,
                           591.182, 587.706, 584.270, 580.874, 577.518, 574.199, 570.919, 567.677, 564.470, 561.300,
                           558.165, 555.066, 552.000, 548.968, 545.969, 543.003, 540.069, 537.166, 534.294, 531.453,
                           528.642, 525.861]
        self.speed2ddis = [  # [x,y]
            [296.13, -640.26],
            [346.13, -690.26],
            [394.20, -738.33],
            [444.20, -788.32],
            [471.13, -815.25],
            [496.13, -840.26],
            [594.20, -938.33],
            [692.27, -1036.39],
            [790.33, -1134.46],
            [888.40, -1232.53],
            [915.91, -1259.62],
            [938.40, -1277.47],
            [961.47, -1290.46],
            [986.47, -1298.27],
            [1000.00, -1299.993]
        ]
        self.speed_peak = False
        if data is not None: self.update_info(data)
        self.flying_dis = 0

        self.debug_estimate_next_pos = None
        self.debug_estimate_uav_next_pos = None

    def estimate_terminal_dis(self):
        p_dis = self.distance[-1]
        p_time = self.flying_time
        p_ddis = self.D_list[p_time] if p_time < len(self.D_list) else self.D_list[-1]
        self.dis_arr = []
        while True:
            tmp = p_dis + p_ddis
            if tmp < 0:
                self.dis_arr.append(tmp)
                return p_dis, self.speed_list[p_time] if p_time < len(self.speed_list) else self.speed_list[-1]
            p_dis = tmp
            p_time += 1
            p_ddis = self.D_list[p_time] if p_time < len(self.D_list) else self.D_list[-1]
            self.dis_arr.append(p_dis)

    @staticmethod
    def interpolant(x, list):
        len_list = len(list)
        left = -1;
        right = len_list
        for i in range(len_list):
            if x > list[i][0]:
                left = i
            else:
                break
        for i in reversed(range(len_list)):
            if x < list[i][0]:
                right = i
            else:
                break
        # ## print(list[left], list[right])
        if left == -1:
            assert right == 0
            left += 1;
            right += 1
        if right == len_list:
            assert left == len_list - 1
            left -= 1;
            right -= 1
        assert list[right][0] != list[left][0]
        return (x - list[left][0]) / (list[right][0] - list[left][0]) * (list[right][1] - list[left][1]) + list[left][1]

    def init_d_list(self):
        for i, speed in enumerate(self.speed_list):
            if speed >= 999.0: return

            self.D_list[i] = self.interpolant(self.speed_list[i], self.speed2ddis)

    def init_speed_list(self, speed):
        self.speed_list[0] = speed
        pointer = 1
        while True:
            self.speed_list[pointer] = self.speed_list[pointer - 1] + 98.066000
            if self.speed_list[pointer] >= 1000:
                self.speed_list[pointer] = 1000
                return
            pointer += 1
            if pointer > 10:
                return

    def update_info(self, data):
        self.alive = (data['Availability'] != 0)
        self.data = data
        if hasattr(self, 'ID'):
            assert data['ID'] == self.ID
        for key in data:
            setattr(self, key, data[key])

        self.pos3d = np.array([self.X, self.Y, self.Alt], dtype=float)
        self.pos2d = np.array([self.X, self.Y], dtype=float)
        if self.previous_pos3d is not None:
            self.flying_dis += np.linalg.norm(self.pos3d - self.previous_pos3d)

        self.is_op_ms = hasattr(self.target, 'OpLeftWeapon')
        if self.target is None: 
            self.is_op_ms = (not hasattr(self.host, 'OpLeftWeapon'))
            return  # 对应的敌方（我方）目标已经死了

        self.delta_traj.append(self.target.pos3d - self.pos3d)
        self.distance.append(np.linalg.norm(self.target.pos3d - self.pos3d))
        self.ms_at_terminal = (self.distance[-1] < 20e3)    # 当与目标的距离小于20km时，进入末制导
        min_distance = min(self.distance)   # 导弹飞行历程中，距离目标的最小距离
        # 是否打空，变成无效导弹
        self.tracking_target = False if self.distance[-1] > (min_distance + 1e3) else True
        if not self.tracking_target: self.ms_at_terminal = True # 如果已经变成无效导弹，也设置成末制导状态，毕竟不需要再管它了
        self.target_speed.append(self.target.Speed)
        self.target_speed.append(self.Speed)
        self.h_angle = 90 - self.Heading * 180 / np.pi
        if self.Speed >= 1000:
            self.speed_peak = True
        if self.flying_time == 0:
            self.init_speed_list(self.Speed)
            self.init_d_list()
        starget_dir = dir2rad(self.target.pos2d - self.pos2d) * 180 / np.pi
        starget_dir = reg_deg_at(starget_dir, ref=self.h_angle)

        if len(self.distance) >= 2:
            self.time.append(self.flying_time)
            self.delta_dis.append(self.distance[-1] - self.distance[-2])
            self.ms_speed.append(self.Speed)
        self.ter_dis_est, self.ter_ms_speed = self.estimate_terminal_dis()

        ### print亮红('ID', self.ID, 'impact dst warning! ', self.ter_dis_est)
        # if (not hasattr(self.target,'OpLeftWeapon')) and self.tracking_target: # 我方是目标
        #     with open('./log/%s'%str(self.ID), 'a+') as f:
        #         f.write('导弹速度 %.2f, 目标距离 %.2f, T估计 %.2f, %s \n'%(self.Speed, self.distance[-1], self.ter_dis_est, str(self.dis_arr)))
        self.impact_eta = len(self.dis_arr) - 1

        self.debug_estimate_next_pos = self.pos3d.copy()
        self.debug_estimate_next_pos[2] += self.Speed * np.sin(self.Pitch)
        self.debug_estimate_next_pos[0] += self.Speed * np.cos(self.Pitch) * np.cos(self.h_angle * np.pi / 180)
        self.debug_estimate_next_pos[1] += self.Speed * np.cos(self.Pitch) * np.sin(self.h_angle * np.pi / 180)

        self.debug_estimate_uav_next_pos = self.target.pos3d.copy()
        self.debug_estimate_uav_next_pos[2] += self.target.Speed * np.sin(self.target.Pitch)
        self.debug_estimate_uav_next_pos[0] += self.target.Speed * np.cos(self.target.Pitch) * np.cos(
            self.target.h_angle * np.pi / 180)
        self.debug_estimate_uav_next_pos[1] += self.target.Speed * np.cos(self.target.Pitch) * np.sin(
            self.target.h_angle * np.pi / 180)

        d_angle = reg_deg_at(self.target.h_angle, ref=self.h_angle + 180) - (self.h_angle + 180)
        # 处于转向阶段的飞机 不适用
        # if np.abs(d_angle) < 1:
        #     if self.flying_time == 2:
        #         with open('./log/ft1.txt', 'a+') as f:
        #             f.write('speed %.2f, delta %.2f \n'%(self.previous_speed, self.distance[-1]-self.distance[-2]))
        #     elif self.flying_time>2:
        #         with open('./log/ft2.txt', 'a+') as f:
        #             f.write('speed %.2f, delta %.2f \n'%(self.previous_speed, self.distance[-1]-self.distance[-2]))

        self.previous_speed = self.Speed
        self.previous_pos3d = self.pos3d
        pass

    def step_init(self):
        self.flying_time += 1
        self.alive = False

    def end_life(self):
        if self.target is None: return  # 对应的敌方（我方）目标已经死了
        import sys, logging
        # ## print亮靛('alive:',self.target.alive)
        # if (not hasattr(self.target,'OpLeftWeapon')) and self.tracking_target: # 我方是目标
        #     with open('./log/%s'%str(self.ID), 'a+') as f:
        #         f.write('命中速度 %.2f \n'%self.Speed)
        #         f.write('存活？ %s \n'%str(self.target.alive))
        #         f.write('*************************\n')
        #     if self.target.alive:
        #         import os
        #         os.remove('./log/%s'%str(self.ID))
        # ## print("".join([ "%.3f, "%d for d in self.ms_speed]))
        return


class Baseclass(Agent):
    def __init__(self, name, config):
        super(Baseclass, self).__init__(name, config["side"])
        self.init()

    def init(self):
        self.n_uav = 4
        self.cmd_list = []
        self.state_recall = {}
        self.STATE = 'Full_Assult'
        self.individual_state = ['Assult'] * self.n_uav
        self.id_mapping = None
        self.Name_mapping = None
        self.my_planes = None
        self.op_planes = None
        self.ms = []
        self.escape_angle = 100  # self.conf['escape_angle']
        self.vip_escape_angle = 0  # self.conf['escape_angle']
        self.escape_distance = 1.3e3  # self.conf['escape_distance']
        self.prepare_escape_distance = 10e3
        self.obs_side_1 = []
        self.our_vantage = False
        self.coop_list = None
        self.Fleet_Attack = False
        # self.vip_escape_distance = 60e3  # self.conf['escape_distance']
        # self.vip_prepare_escape_distance = 75e3

        self.initial_attack_order_adjusted = False

    def find_plane_by_name(self, name):
        for p in self.my_planes + self.op_planes:
            if p.Name == name: return p
        return None

    def find_planes_by_squad(self, squad_name):
        return [p for p in self.my_planes  if p.squad_name == squad_name]

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

    # 加载观测量
    def my_process_observation_and_show(self, obs_side, sim_time):
        # need init?
        if self.my_planes is None:
            self.my_planes = [Plane(manager=self) for _ in range(self.n_uav +1)]
            self.op_planes = [Plane(manager=self) for _ in range(self.n_uav +1)]

            # register info for the first time!
            for idx, uvas_info in enumerate(obs_side['platforminfos']):
                self.my_planes[idx].update_info(uvas_info, init=True)
            for idx, uvas_info in enumerate(obs_side['trackinfos']):
                self.op_planes[idx].update_info(uvas_info, init=True)

            if len(obs_side['platforminfos']) != (self.n_uav + 1):
                print亮红('Error! 没有从起始状态执行,可能导致一系列错误!')
                self.my_planes = list(filter(lambda p: hasattr(p, 'ID'), self.my_planes))

            if len(obs_side['trackinfos']) != (self.n_uav + 1):
                print亮红('Error! 没有从起始状态执行,可能导致一系列错误!')
                self.op_planes = list(filter(lambda p: hasattr(p, 'ID'), self.op_planes))

            for obj in self.my_planes + self.op_planes + self.ms:
                obj.step_init()

            # self.init_attack_order()

        # set every plane object to step_init
        for obj in self.my_planes + self.op_planes + self.ms:
            obj.step_init()

        everything = []
        # part 1
        my_entity_infos = obs_side['platforminfos']
        enemy_entity_infos = obs_side['trackinfos']
        if len(my_entity_infos) < 1: return
        for uvas_info in (my_entity_infos + enemy_entity_infos):
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
            host = self.find_plane_by_id(host_id)
            if host is not None:
                host.my_ms.append(missile_info)
            target = self.find_plane_by_id(target_id)
            if target is not None:
                target.attacked_by_ms.append(missile_info)
            # if host_id in target.hanging_attack_by: target.hanging_attack_by.remove(host_id)    # hanging_attack_by
            ms = self.find_ms_by_id(missile_info["ID"])
            missile_info.update({ 'host':host, 'target':target })
            if ms is None:
                self.ms.append(MS(missile_info))
                if host is not None and hasattr(host, 'OpLeftWeapon'):
                    host.OpLeftWeapon -= 1

            else:
                ms.update_info(missile_info)

            ms = self.find_ms_by_id(missile_info["ID"])
            if host is not None:
                assert ms is not None
                host.fired_ms.append(ms)
            everything.append(missile_info)

        # part 3 distance matrix
        everything_pos = np.array([np.array([p['X'], p['Y'], p['Z']]) for p in everything])
        self.id2index = {p['ID']: i for i, p in enumerate(everything)}
        self.active_index = self.id2index
        self.everything_dis = distance_matrix(everything_pos)
        Plane.dis_mat = self.everything_dis
        Plane.dis_mat_id2index = self.id2index
        for m in self.ms:
            if not m.alive: m.end_life()
        '''
        if len(list(filter(lambda p: not p.alive, self.my_planes)))>0:
            ## print('飞机被命中')
            pass

        self.my_planes = list(filter(lambda p: p.alive, self.my_planes))
        self.op_planes = list(filter(lambda p: p.alive, self.op_planes))
        self.ms = list(filter(lambda m: m.alive, self.ms))
        '''
        # self.update_mapping()

        # part 4 calculate in-radar relationship
        # for p in self.my_planes:
        #     p.in_radar = self.get_in_radar_op(p)

        # for p in self.my_planes:
        #     p.in_attack_radar = self.get_in_attack_radar_op(p)

        # for p in self.my_planes:
        #     p.in_attack_deadzoo = self.get_dead_zoo(p)

        # for op in self.op_planes:
        #     op.in_target_zoo = self.get_in_target_zoo(op)



    def id2index(self, arr):
        return [self.id2index[id_] for id_ in arr]

    def get_dis(self, id1, id2):
        index1 = self.id2index[id1]
        index2 = self.id2index[id2]
        return self.everything_dis[index1, index2]

    def observe(self, sim_time, obs_side, **kwargs) -> List[dict]:
        self.cmd_list = []
        self.my_process_observation_and_show(obs_side, sim_time)

    def process_decision(self, time, obs_side):
        self.time_of_game = time
        if time == 3: self.init_pos(self.cmd_list)
        if time >= 4: self.make_decision()
        return

    # 初始化位置
    def init_pos(self, cmd_list):
        '''
            y     [-150000, +150000]
            red_x [-150000, -125000]
            blue_x [125000,  150000]
        '''
        Special.init_Y = (2*np.random.rand() - 1)*60000  # -125000 #
        leader_original_pos = {}  # 用以初始化当前方的位置
        if self.name == "red":
            # leader_original_pos = {"X": -Special.init_X, "Y": Special.init_Y, "Z": Special.init_Z}
            leader_original_pos = {"X": -Special.init_X, "Y": -Special.init_Y, "Z": Special.init_Z}
            init_dir = 90
        else:
            leader_original_pos = {"X": Special.init_X, "Y": Special.init_Y, "Z": Special.init_Z}
            init_dir = 360 - 90

        interval_distance = 20000   # 间隔 5000米排列
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
                offset = 0 if sub_index<=1 else 1e3
                if sub_index & 1 == 0:
                    cmd_list.append(
                        CmdEnv.make_entityinitinfo(p.ID,
                                                   sub_pos['X'], sub_pos['Y'] + interval_distance + offset,
                                                   sub_pos['Z'], Special.init_speed_drone, init_dir))
                else:
                    cmd_list.append(
                        CmdEnv.make_entityinitinfo(p.ID,
                                                   sub_pos['X'], sub_pos['Y'] - interval_distance - offset,
                                                   sub_pos['Z'], Special.init_speed_drone, init_dir))
                    # interval_distance *= 2 # 编号翻倍
                sub_index += 1
            else:
                assert False, ('???')

    # 初始化攻击序列，游戏开始后，会二次调整
    def init_attack_order(self):
        a_plane = self.my_planes[0]
        if "蓝" in a_plane.Name:
            color = "蓝"
            op_color = "红"
        elif "红" in a_plane.Name:
            color = "红"
            op_color = "蓝"
        else:
            assert False
        self.color = color
        self.op_color = op_color
        attack_order = {
            color + "无人机1": [
                op_color + "无人机1",
                op_color + "无人机2",
                op_color + "有人机",
            ],
            color + "无人机2": [
                op_color + "无人机1",
                op_color + "无人机2",
                op_color + "有人机",
            ],
            color + "无人机3": [
                op_color + "无人机4",
                op_color + "无人机3",
                op_color + "有人机",
            ],
            color + "无人机4": [
                op_color + "无人机4",
                op_color + "无人机3",
                op_color + "有人机",
            ],
            color + "有人机": [
                op_color + "无人机1",
                op_color + "无人机2",
                op_color + "无人机3",
                op_color + "无人机4",
                op_color + "有人机",
            ],
        }

        attack_squad = {
            color + "有人机": {
                "squad": "U1",
                "save_last_ammo": False,
                "leader": color + "无人机1",
                "formation_mate": [color + "无人机1", color + "无人机2"]
            },
            color + "无人机1": {
                "squad": "U1",
                "save_last_ammo": False,
                "leader": None,
                "formation_mate": [color + "无人机2", color + "有人机"]
            },
            color + "无人机2": {
                "squad": "U1",
                "save_last_ammo": False,
                "leader": color + "无人机1",
                "formation_mate": [color + "无人机1", color + "有人机"]
            },
            color + "无人机3": {
                "squad": "U2",
                "save_last_ammo": True,
                "leader": None,
                "formation_mate": [color + "无人机4"]
            },
            color + "无人机4": {
                "squad": "U2",
                "save_last_ammo": False,
                "leader": color + "无人机3",
                "formation_mate": [color + "无人机3"]
            },

        }
        for p in self.my_planes:
            p.attack_order = attack_order[p.Name]
            p.squad_name = attack_squad[p.Name]["squad"]
            p.squad_leader = attack_squad[p.Name]["leader"]
            p.formation_mate = attack_squad[p.Name]["formation_mate"]
            p.save_last_ammo = attack_squad[p.Name]["save_last_ammo"]
        pass

    def check_and_make_linepatrolparam(self, receiver, coord_list, cmd_speed, cmd_accmag, cmd_g, force_old_way=False):
        host = self.find_plane_by_id(receiver)
        '''
        for point in coord_list:
            if point['X'] > host.MAX_X:
                point['X'] = host.MAX_X          #; ## print红('if point[X] > host.MAX_X: point[X] = host.MAX_X;')
            if point['X'] < host.MIN_X:
                point['X'] = host.MIN_X          #; ## print红('if point[X] < host.MIN_X: point[X] = host.MIN_X;')
            if point['Y'] > host.MAX_Y: 
                point['Y'] = host.MAX_Y          #; ## print红('if point[Y] > host.MAX_Y: point[Y] = host.MAX_Y;')
            if point['Y'] < host.MIN_Y: 
                point['Y'] = host.MIN_Y          #; ## print红('if point[Y] < host.MIN_Y: point[Y] = host.MIN_Y;')
            if point['Z'] < host.MinHeight: 
                point['Z'] = host.MinHeight      #; ### print红('if point[Z] < host.MinHeight: point[Z] = host.MinHeight;')
            if point['Z'] > host.MaxHeight: 
                point['Z'] = host.MaxHeight      #; ### print红('if point[Z] > host.MaxHeight: point[Z] = host.MaxHeight;')
        '''
        def old_way(point):
            if point['X'] > host.MAX_X:
                point['X'] = host.MAX_X          #; ## print红('if point[X] > host.MAX_X: point[X] = host.MAX_X;')
            if point['X'] < host.MIN_X:
                point['X'] = host.MIN_X          #; ## print红('if point[X] < host.MIN_X: point[X] = host.MIN_X;')
            if point['Y'] > host.MAX_Y: 
                point['Y'] = host.MAX_Y          #; ## print红('if point[Y] > host.MAX_Y: point[Y] = host.MAX_Y;')
            if point['Y'] < host.MIN_Y: 
                point['Y'] = host.MIN_Y          #; ## print红('if point[Y] < host.MIN_Y: point[Y] = host.MIN_Y;')
            if point['Z'] < host.MinHeight: 
                point['Z'] = host.MinHeight      #; ### print红('if point[Z] < host.MinHeight: point[Z] = host.MinHeight;')
            if point['Z'] > host.MaxHeight: 
                point['Z'] = host.MaxHeight      #; ### print红('if point[Z] > host.MaxHeight: point[Z] = host.MaxHeight;')
            return point

        def avail_coord(point):
            if point['X'] > host.MAX_X:
                return False
            if point['X'] < host.MIN_X:
                return False
            if point['Y'] > host.MAX_Y: 
                return False
            if point['Y'] < host.MIN_Y: 
                return False
            if point['Z'] < host.MinHeight: 
                return False
            if point['Z'] > host.MaxHeight: 
                return False
            return True
        def avail_coord_np(point):
            if point[0] > host.MAX_X:
                return False
            if point[0] < host.MIN_X:
                return False
            if point[1] > host.MAX_Y: 
                return False
            if point[1] < host.MIN_Y: 
                return False
            if point[2] < host.MinHeight: 
                return False
            if point[2] > host.MaxHeight: 
                return False
            return True

        for i, point in enumerate(coord_list):
            if avail_coord(point): continue
            if force_old_way:
                coord_list[i] = old_way(point)
                continue
            # 坐标需要修正！！
            arr = np.array([point['X'],point['Y'],point['Z']])
            vec_dir_3d = arr - host.pos3d # 从host指向arr
            vec_dir_3d_unit = vec_dir_3d / (np.linalg.norm(vec_dir_3d)+1e-7)
            res_len = self.prob_len(
                starting_point=host.pos3d,
                direction=vec_dir_3d_unit,
                dx = 100,
                lamb = avail_coord_np,
                max_try = 1000, # 100km 
            )
            res_avail = res_len*vec_dir_3d_unit + host.pos3d
            if res_len < 300:
                # ???? 这种情况比较危险，采用旧的方式
                coord_list[i] = old_way(point)
            else:
                coord_list[i] = old_way({"X": res_avail[0], "Y": res_avail[1], "Z": res_avail[2]})
        return CmdEnv.make_linepatrolparam(receiver, coord_list, cmd_speed, cmd_accmag, cmd_g)


    @staticmethod
    def prob_len(starting_point, direction, dx, lamb, max_try):
        for i in range(max_try):
            dst = starting_point + direction*dx*(i+1)
            if lamb(dst):
                continue
            else:
                return dx*(i+1)
        return dx*max_try

    def has_overall_adv(self):
        def delta_oppsite_to_op(p):
            # 获取敌方飞机的相反方向
            # 修正1：不对无弹的敌方飞机做出响应
            delta_pos = np.array([p.pos2d - op.pos2d for op in self.op_planes if op.OpLeftWeapon>0])
            dis = np.array([p.get_dis(op.ID) for op in self.op_planes if op.OpLeftWeapon>0])
            delta_dir = delta_pos / repeat_at(dis, -1, 2)
            weight = np_softmax(dis.max() - dis)
            delta = (delta_dir * repeat_at(weight, -1, 2)).sum(0)
            # math_rad_dir = dir2rad(delta)
            return delta / (np.linalg.norm(delta)+1e-7)

        def get_nearest_op_with_ms(p):
            delta_pos = np.array([p.pos2d - op.pos2d for op in self.op_planes if op.OpLeftWeapon>0])
            dis = np.array([p.get_dis(op.ID) for op in self.op_planes if op.OpLeftWeapon>0])
            if len(dis)>0:
                return self.op_planes[np.argmin(dis)]
            else:
                return None

        all_op_ammo = sum([op.OpLeftWeapon for op in self.op_planes])   # 所有敌方导弹的总和
        plane_num_adv = len(self.my_planes) - len(self.op_planes)       # 我战机数量减敌战机数量
        vip_p = [p for p in self.my_planes][0]

        if plane_num_adv > 0:
            near_op = get_nearest_op_with_ms(vip_p)
            if near_op is None:
                #
                return False, None

            delta = delta_oppsite_to_op(vip_p)

            # circle_evade：局部滞回变量
            if not hasattr(vip_p, 'circle_evade'):
                vip_p.circle_evade = False
            # 滞回变量状态转换
            # how_far_to_wall = self.prob_len(
            #         starting_point=vip_p.pos2d, 
            #         direction=delta/(np.linalg.norm(delta)+1e-7), 
            #         dx=5e3, 
            #         lamb=lambda x: (np.abs(x[0])<145e3 and np.abs(x[1])<145e3), 
            #         max_try=50)
            
            if vip_p.circle_evade:
                if np.linalg.norm(vip_p.pos2d) < 110e3:
                    vip_p.circle_evade = False
            else:
                if np.linalg.norm(vip_p.pos2d) > 125e3:
                    vip_p.circle_evade = True
            # 滞回变量作用
            if vip_p.circle_evade:
                delta_vertical_01 = np.cross([delta[0], delta[1], 0], [0,0,1] )[:2]
                delta_vertical_02 = np.cross([delta[0], delta[1], 0], [0,0,-1] )[:2]
                delta_vertical_01 = delta_vertical_01/(np.linalg.norm(delta_vertical_01)+1e-7)
                delta_vertical_02 = delta_vertical_02/(np.linalg.norm(delta_vertical_02)+1e-7)
                r = self.thresh_hold_projection(np.linalg.norm(vip_p.pos2d), 
                        min_x=110e3, y_min_x=0.0, 
                        max_x=140e3, y_max_x=1.0)
                delta_01 = delta*(1-r) + delta_vertical_01*r
                delta_02 = delta*(1-r) + delta_vertical_02*r
                # 计算两种规避方式的回避余地
                delta_space_01 = self.prob_len(starting_point=vip_p.pos2d, 
                    direction=delta_vertical_01, 
                    dx=5e3, 
                    lamb=lambda x: (np.abs(x[0])<145e3 and np.abs(x[1])<145e3), 
                    max_try=50)
                delta_space_02 = self.prob_len(starting_point=vip_p.pos2d, 
                    direction=delta_vertical_02, 
                    dx=5e3, 
                    lamb=lambda x: (np.abs(x[0])<145e3 and np.abs(x[1])<145e3), 
                    max_try=50)
                if delta_space_01 > delta_space_02:
                    delta = delta_01
                else:
                    delta = delta_02



            H2 = delta * 100e3 + vip_p.pos2d


            goto_location = [ {
                    "X": H2[0],
                    "Y": H2[1],
                    "Z": vip_p.Z
                }]

            if vip_p.advantage_state == 'going_away':
                if self.get_dis(near_op.ID, vip_p.ID) > 75e3:
                    vip_p.advantage_state = 'default'
                    return False, None
                else:
                    return True, goto_location
            elif vip_p.advantage_state == 'default':
                if self.get_dis(near_op.ID, vip_p.ID) <= 55e3:
                    vip_p.advantage_state = 'going_away'
                    return True, goto_location
                return False, None
        else:
            return False, None