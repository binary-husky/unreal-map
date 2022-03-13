from typing import List
from ..agent import Agent
from ..env_cmd import CmdEnv
from .UTILS.colorful import *
from UTILS.tensor_ops import dir2rad, np_softmax, reg_rad_at, reg_rad, repeat_at, avg_rad
from .maneuver import maneuver_angle_to_ms, maneuver_angle_to_ms3d, maneuver_speed_to_ms, maneuver_vip
import copy, logging, random, time
import numpy as np
from .tools import distance_matrix
from .base import Baseclass, Special, Drone, Vip, Plane, MS
from .missile_policy import MS_policy
from .emergent import Emergent
Init_attack_order_adjust_dis = 100e3

class Attack_Adjust():
    @staticmethod
    def test_rank(center, q, phi0, alpha_deg, polar=1):
        delta = q-center
        d = np.linalg.norm(delta)
        phi = (np.arctan2(delta[1],delta[0])-phi0)*polar
        alpha = alpha_deg * (np.pi/180)
        small_first_rank = d*np.cos(phi) - d*np.sin(phi)*np.tan(alpha) 
        return small_first_rank

    def new_new_init_adjust(self, squad_name):
        all_my_plane_center = np.array([p.pos2d for p in self.my_planes])
        all_my_plane_center = all_my_plane_center.mean(axis=0)
        # squad_mem_center = np.array([p.pos2d for p in self.my_planes if p.squad_name == squad_name])
        # squad_mem_center = squad_mem_center.mean(axis=0)
        center = all_my_plane_center
        # # 找出敌方处于边缘的两架飞机
        # angle = -9999
        # op1_tmp = None
        # op2_tmp = None
        # for op1 in self.op_planes:
        #     for op2 in self.op_planes:
        #         if op1 is op2: continue
        #         delta_deg = self.get_points_angle_deg(pt1=op1.pos2d, 
        #             pt_center=center, 
        #             pt2=op2.pos2d)
        #         if delta_deg > angle:
        #             angle = delta_deg
        #             op1_tmp = op1
        #             op2_tmp = op2

        # rad1 = dir2rad(op1_tmp.pos2d-center) # 最侧翼的敌机1
        # rad2 = dir2rad(op2_tmp.pos2d-center) # 最侧翼的敌机2，另一侧
        # phi0 = avg_rad(rad1, rad2)
        op_vip = [op for op in self.op_planes if op.is_vip][0]
        phi0 = dir2rad(op_vip.pos2d-all_my_plane_center)

        for op in self.op_planes:
            if op.is_vip: continue
            v1 = op_vip.pos2d - all_my_plane_center
            v2 = op.pos2d - all_my_plane_center
            v1n = np.linalg.norm(v1)
            v2n = np.linalg.norm(v2)
            theta = self.get_points_angle_deg(pt1=op_vip.pos2d, 
                    pt_center=all_my_plane_center, 
                    pt2=op.pos2d)*np.pi/180
            alpha = np.arctan((v2n*np.cos(theta)-v1n)/(v2n*np.sin(theta)))
            print(op.Name, alpha*180/np.pi)

        if squad_name == "U1":
            small_first_rank = [self.test_rank(
                        center,
                        q = op.pos2d,
                        phi0 = phi0,
                        alpha_deg=15,
                        polar = 1 if self.name == "red" else -1
                        ) for op in self.op_planes]

        if squad_name == "U2":
            small_first_rank = [self.test_rank(
                        center,
                        q = op.pos2d,
                        phi0 = phi0,
                        alpha_deg=15,
                        polar = -1 if self.name == "red" else 1
                    ) for op in self.op_planes]

        op_list = np.array([op for op in self.op_planes], dtype=object)
        sorted_index = np.argsort(small_first_rank)   # argsort is small-first

        op_list_sorted = op_list[sorted_index]

        op_attk_by_squad = op_list_sorted

        # 一组攻击上边缘，一组攻击下边缘
        squad_mem = [p for p in self.my_planes if p.squad_name == squad_name]
        for p in squad_mem:
            p.attack_order = [op.Name for op in op_attk_by_squad]
            logging.info("'{0}' 分配战机进攻序列： '{1}'".format(p.Name, p.attack_order[0]))


    # 根据敌方每个飞机的剩余的弹药量，变动进攻序列，将无弹的目标放到最末的优先级
    # 调整的条件：1、敌方无弹，2、无人机，3、没有正在往这个敌方目标飞行的导弹
    def adjust_by_left_ammo(self):
        for p in self.my_planes:
            attack_list = [self.find_plane_by_name(op_name) for op_name in p.attack_order]
            attack_list = [op for op in attack_list if op is not None]
            new_attack_list_normal = []
            new_attack_list_mvback = []
            for i in range(len(attack_list)):
                missile_towards_op = [ms for ms in self.ms if ms.EngageTargetID==attack_list[i].ID and ms.ms_at_terminal]
                require_no_ms_guidence = (len(missile_towards_op)==0)
                if attack_list[i].OpLeftWeapon == 0 and attack_list[i].is_drone and require_no_ms_guidence:
                    new_attack_list_mvback.append(attack_list[i])
                else:
                    new_attack_list_normal.append(attack_list[i])
            new_attack_list = new_attack_list_normal + new_attack_list_mvback
            p.attack_order = [op.Name for op in new_attack_list]
        pass


    # 调整每个飞机的进攻序列
    def adjust_attack_order(self):
        '''
        # 在敌我距离缩小到阈值时，第一次调整进攻序列
        min_distance = min([self.get_dis(op.ID, p.ID) for op in self.op_planes for p in self.my_planes])
        if min_distance < Init_attack_order_adjust_dis:
            if not self.initial_attack_order_adjusted:
                self.initial_attack_order_adjusted = True
                self.init_adjust()
        
        '''

        min_distance = min([self.get_dis(op.ID, p.ID) for op in self.op_planes for p in self.my_planes])
        if min_distance > Init_attack_order_adjust_dis:
            self.new_new_init_adjust(squad_name="U1")
            self.new_new_init_adjust(squad_name="U2")
        else:
            self.new_new_init_adjust(squad_name="U1")
            self.new_new_init_adjust(squad_name="U2")
            print('over range')

        # 根据将敌方的无弹目标放到末优先级
        self.adjust_by_left_ammo()

        '''
        # 当敌方的有人机距离小队中全部成员的距离小于50， 或者距离任意成员的距离小于30，小队内全体成员前往歼灭敌方有人机
        op_vip = self.find_plane_by_name(self.op_color + "有人机")
        if op_vip is None: return

        def attack_vip_first(p):
            # ## print亮紫(p.attack_order)
            p.attack_order = [op_name for op_name in p.attack_order if "有人机" in op_name] + \
                            [op_name for op_name in p.attack_order if "有人机" not in op_name]
            # ## print亮蓝(p.attack_order)
            pass

        squads = ['U1', 'U2']
        for squad in squads:
            p_list = self.find_planes_by_squad(squad)
            if len(p_list)==0: continue
            dis2_op_vip = [self.get_dis(p.ID, op_vip.ID) for p in p_list]
            if min(dis2_op_vip) < 30e3:
                # ## print亮紫('小队全体攻击有人机')
                for p in p_list: attack_vip_first(p)
            elif max(dis2_op_vip) < 50e3:
                # ## print亮紫('小队全体攻击有人机')
                for p in p_list: attack_vip_first(p)
        '''
        return

    def adjust_attack_order_phase2(self):
        for p in self.my_planes:
            if hasattr(p, 'ms_policy_suggested_target') and p.ms_policy_suggested_target is not None:
                # 根据 ms_policy_suggested_target 调整攻击序列
                attack_list = [self.find_plane_by_name(op_name) for op_name in p.attack_order]
                attack_list = [op for op in attack_list if op is not None]
                new_attack_list_normal = []
                new_attack_list_mvback = []
                for i in range(len(attack_list)):
                    if attack_list[i].ID == p.ms_policy_suggested_target.ID:
                        new_attack_list_mvback.append(attack_list[i])
                    else:
                        new_attack_list_normal.append(attack_list[i])
                new_attack_list = [p.ms_policy_suggested_target] + new_attack_list_normal + new_attack_list_mvback
                p.attack_order = [op.Name for op in new_attack_list]
