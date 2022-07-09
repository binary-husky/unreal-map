from typing import List
from ..agent import Agent
from ..env_cmd import CmdEnv
from UTIL.colorful import *
from UTIL.tensor_ops import dir2rad, np_softmax, reg_rad_at, reg_rad
from .maneuver import maneuver_cold_to_ms, maneuver_vertical_to_ms, maneuver_angle_to_ms
import copy
import random
import numpy as np
import time
from .tools import distance_matrix
from .base import Baseclass, Special, Drone, Vip, Plane, MS
Enable_Evade = True
Fleet_Attack_Angle = 90
Fleet_Attack_Dis = 90e3
Init_attack_order_adjust_dis = 200e3


def nearby(dir, L):
    dir = dir / (np.linalg.norm(dir) + 1e-3) * L




class Yi_team(Baseclass):
    def __init__(self, name, config):
        super(Baseclass, self).__init__(name, config["side"])
        self.init()

    # 调整每个飞机的进攻序列
    def adjust_attack_order(self):

        # 只执行一次的初始化调整，敌我距离小于 Init_attack_order_adjust_dis 时执行一次
        def init_adjust():
            op_list = np.array([op for op in self.op_planes], dtype=object)
            op_dis_2_vip = [self.get_dis(
                op.ID, p.ID) for op in self.op_planes for p in self.my_planes if p.is_vip]
            assert len(op_dis_2_vip) == len(self.op_planes)
            sorted_index = np.argsort(op_dis_2_vip)
            op_list_sorted = op_list[sorted_index]
            op_attk_by_squad_1 = op_list_sorted[0::2]
            op_attk_by_squad_2 = op_list_sorted[1::2]
            op_list1_has_op_vip = any([op.is_vip for op in op_attk_by_squad_1])
            op_list2_has_op_vip = any([op.is_vip for op in op_attk_by_squad_2])
            op_vip = [op for op in self.op_planes if op.is_vip][0]
            assert (op_list1_has_op_vip and (not op_list2_has_op_vip)) or (
                op_list2_has_op_vip and (not op_list1_has_op_vip))
            if op_list2_has_op_vip:
                t = op_attk_by_squad_1
                op_attk_by_squad_1 = op_attk_by_squad_2
                op_attk_by_squad_2 = t
            squad_1_mem = [p for p in self.my_planes if p.squad_name == "U1"]
            squad_2_mem = [p for p in self.my_planes if p.squad_name == "U2"]
            for p in squad_1_mem:
                p.attack_order = [op.Name for op in op_attk_by_squad_1]
                # 如果攻击序列中没有有人机，则把有人机放到最后
                if not any(['有人机' in p_name for p_name in p.attack_order]):
                    p.attack_order.append(op_vip.Name)
                # print亮绿(p.Name, '分配战机进攻序列：', p.attack_order)
            for p in squad_2_mem:
                p.attack_order = [op.Name for op in op_attk_by_squad_2]
                # 如果攻击序列中没有有人机，则把有人机放到最后
                if not any(['有人机' in p_name for p_name in p.attack_order]):
                    p.attack_order.append(op_vip.Name)
                # print亮蓝(p.Name, '分配战机进攻序列：', p.attack_order)
            pass

        # 根据敌方每个飞机的剩余的弹药量，变动进攻序列，将无弹的目标放到最末的优先级
        def adjust_by_left_ammo():
            for p in self.my_planes:
                attack_list = [self.find_plane_by_name(op_name) for op_name in p.attack_order]
                attack_list = [op for op in attack_list if op is not None]
                new_attack_list_normal = []
                new_attack_list_mvback = []
                for i in range(len(attack_list)):
                    if attack_list[i].OpLeftWeapon == 0 and attack_list[i].is_drone:
                        new_attack_list_mvback.append(attack_list[i])
                    else:
                        new_attack_list_normal.append(attack_list[i])
                new_attack_list = new_attack_list_normal + new_attack_list_mvback
                # print('p.attack_order')
                # print亮红(p.attack_order)
                p.attack_order = [op.Name for op in new_attack_list]
            pass

        # 调用进攻序列调整函数
        min_distance = min([self.get_dis(op.ID, p.ID)
                           for op in self.op_planes for p in self.my_planes])
        if min_distance < Init_attack_order_adjust_dis:
            if not self.initial_attack_order_adjusted:
                self.initial_attack_order_adjusted = True
                init_adjust()

        adjust_by_left_ammo()


        # 当敌方的有人机距离小队中全部成员的距离小于50， 或者距离任意成员的距离小于30，小队内全体成员前往歼灭敌方有人机
        op_vip = self.find_plane_by_name(self.op_color + "有人机")
        if op_vip is None: return

        def attack_vip_first(p):
            # print亮紫(p.attack_order)
            p.attack_order = [op_name for op_name in p.attack_order if "有人机" in op_name] + \
                            [op_name for op_name in p.attack_order if "有人机" not in op_name]
            # print亮蓝(p.attack_order)
            pass

        squads = ['U1', 'U2']
        for squad in squads:
            p_list = self.find_planes_by_squad(squad)
            if len(p_list)==0: continue
            dis2_op_vip = [self.get_dis(p.ID, op_vip.ID) for p in p_list]
            if min(dis2_op_vip) < 30e3:
                # print亮紫('小队全体攻击有人机')
                for p in p_list: attack_vip_first(p)
            elif max(dis2_op_vip) < 50e3:
                # print亮紫('小队全体攻击有人机')
                for p in p_list: attack_vip_first(p)
        return
    # # 获取敌我距离
    # def get_p_op_dist(self):
    #     dist_list=np.zeros([len(self.my_planes),len(self.op_planes)])
    #     for i in range(len(self.my_planes)):
    #         for j in range(len(self.op_planes)):
    #             self.get_dis(p.ID, op.ID)

    # 获取必杀范围内的敌机
    def get_dead_zoo(self, p):
        in_radar = []
        for op in self.op_planes:
            dis = self.get_dis(op.ID, p.ID)
            if dis > p.DeadZooRadarDis:
                continue
            # 检查角度是否满足
            delta = op.pos2d - p.pos2d
            theta = 90 - p.Heading * 180 / np.pi
            theta = reg_rad(theta * np.pi / 180)
            delta2 = np.matmul(np.array([[np.cos(theta), np.sin(theta)],
                                         [-np.sin(theta), np.cos(theta)]]), delta.T)
            deg = dir2rad(delta2) * 180 / np.pi
            if deg >= -p.DeadZooRadarHorizon and deg <= p.DeadZooRadarHorizon:
                in_radar.append(op)
        return in_radar

    # 获取"深入"飞机雷达范围内的敌机
    def get_in_attack_radar_op(self, p):
        in_radar = []
        for op in self.op_planes:
            dis = self.get_dis(op.ID, p.ID)
            if dis > p.AttackRadarDis:
                continue
            # 检查角度是否满足
            delta = op.pos2d - p.pos2d
            theta = 90 - p.Heading * 180 / np.pi
            theta = reg_rad(theta * np.pi / 180)
            delta2 = np.matmul(np.array([[np.cos(theta), np.sin(theta)],
                                         [-np.sin(theta), np.cos(theta)]]), delta.T)
            deg = dir2rad(delta2) * 180 / np.pi
            if deg >= -p.AttackRadarHorizon and deg <= p.AttackRadarHorizon:
                in_radar.append(op)
        return in_radar

    # 获取进入飞机雷达范围内的敌机
    def get_in_radar_op(self, p):
        in_radar = []
        for op in self.op_planes:
            dis = self.get_dis(op.ID, p.ID)
            if dis > p.RadarDis:
                continue
            # 检查角度是否满足
            delta = op.pos2d - p.pos2d
            theta = 90 - p.Heading * 180 / np.pi
            theta = reg_rad(theta*np.pi/180)
            delta2 = np.matmul(np.array([[np.cos(theta), np.sin(theta)],
                                         [-np.sin(theta), np.cos(theta)]]), delta.T)
            deg = dir2rad(delta2) * 180 / np.pi
            if deg >= -p.RadarHorizon and deg <= p.RadarHorizon:
                in_radar.append(op)
        return in_radar

    # 获取照射追踪我方的敌方导弹的无人接
    def get_missile_op_in_radar(self, p):
        in_radar = []
        for op in self.op_planes:
            dis = self.get_dis(op.ID, p.ID)
            if dis > p.RadarDis:
                continue
            # 检查角度是否满足
            delta = op.pos2d - p.pos2d
            theta = 90 - p.Heading * 180 / np.pi
            theta = reg_rad(theta*np.pi/180)
            delta2 = np.matmul(np.array([[np.cos(theta), np.sin(theta)],
                                         [-np.sin(theta), np.cos(theta)]]), delta.T)
            deg = dir2rad(delta2) * 180 / np.pi
            if deg >= -p.RadarHorizon and deg <= p.RadarHorizon:
                in_radar.append(op)
        return in_radar

    # 检查紧急避险是否被触发，prepare_escape会将机头指向目标，emergent_escape会将机头调转80度急转弯
    def check_escape_condition(self):
        for p in self.my_planes:
            if len(p.attacked_by_ms) >= 1:
                dis = [self.get_dis(p.ID, ms['ID']) for ms in p.attacked_by_ms]
                min_ms_dis = min(dis)

                if min_ms_dis <= p.prepare_escape_distance:  # 2e3: # 2000ok 1500ok
                    p.step_state = 'prepare_escape'

                # high priority
                if min_ms_dis <= p.escape_distance:  # 2e3: # 2000ok 1500ok
                    p.step_state = 'emergent_escape'

                for ms_dict in p.attacked_by_ms:
                    ms = self.find_ms_by_id(ms_dict['ID'])
                    ft = ms.flying_time
                pass

    # 执行进攻序列，删除已经消灭的目标
    def check_follow_condition(self):
        for p in self.my_planes:
            # load plane.current_target
            while True:
                if len(p.attack_order) == 0:
                    p.current_target = p
                    # print亮红('check_follow_condition 异常')
                    break
                attack_target = self.find_plane_by_name(p.attack_order[0])
                if attack_target is None:
                    p.attack_order.pop(0)
                    continue
                else:
                    p.current_target = attack_target
                    p.bind = attack_target.ID
                    break
            p.step_state = "follow"
        pass

    # 导弹的运营
    def missile_policy(self,obs_side_1):

        # 第一步，根据进攻序列，检查是否进入射程，如果是，被打击的飞机的wish_list加一个友机
        for p in self.my_planes:
            if p.LeftWeapon == 0:
                continue
            # 有人机保存最后一枚导弹
            if p.is_vip and p.LeftWeapon == 1:
                continue
            # 特殊的无人机保存最后一枚导弹
            if p.save_last_ammo and p.LeftWeapon == 1:
                continue
            distance_to_target = self.get_dis(p.ID, p.current_target.ID)
            if distance_to_target > 55e3:
                continue
            # 距离小于 60e3
            target = p.current_target
            ms_on_its_way = target.attacked_by_ms
            # assert len(target.hanging_attack_by)==0
            n_hanging_attack_ms = 0
            n_ms = len(ms_on_its_way)
            if (n_ms + n_hanging_attack_ms) > 0:
                continue
            # 目前没有被导弹攻击
            target.wish_list_by.append(p.ID)
            pass

        # 第二步，根据飞机当前的雷达区域，判断是否有飞机深入其中但没有被攻击
        # 如果是，被打击的飞机的wish_list加一个友机
        for p in self.my_planes:
            if p.LeftWeapon == 0:
                continue
            # 如果处于规避状态，保存导弹不发射
            if p.persistent_state in ['emergent_escape_close', 'emergent_escape']:
                continue
            # 有人机保存最后一枚导弹
            if p.is_vip and p.LeftWeapon == 1:
                continue
            # 特殊的无人机保存最后一枚导弹
            if p.save_last_ammo and p.LeftWeapon == 1:
                continue
            in_attack_radar = p.in_attack_radar
            for target in in_attack_radar:
                ms_on_its_way = target.attacked_by_ms
                n_ms = len(ms_on_its_way)
                if n_ms >= 2:
                    continue
                if n_ms == 0:
                    # 闯入攻击区，但恰好没有导弹跟着
                    target.wish_list_by.append(p.ID)
                    continue
                if n_ms == 1:
                    the_ms = ms_on_its_way[0]
                    ms_distance_2target = self.get_dis(target.ID, the_ms['ID'])
                    p_distance_2target = self.get_dis(p.ID, target.ID)
                    # 1 要求飞机比已有导弹更接近敌机
                    if p_distance_2target > ms_distance_2target:
                        continue
                    # 2 要求本机的弹药？？
                    target.wish_list_by.append(p.ID)

        # 第三步，如果敌有人机在极近的必杀范围内，直接发射导弹，不再走流程
        for p in self.my_planes:
            if p.LeftWeapon == 0:
                continue
            # if p.is_vip and p.LeftWeapon == 1:
            #     continue

            in_attack_radar = p.in_attack_deadzoo
            for target in in_attack_radar:
                # 判断heading夹角
                angle = np.abs((p.Heading - reg_rad_at(target.Heading, p.Heading)) * 180 / np.pi)
                if angle > 90: angle = 180-angle
                if angle > 30: continue # 速度倾角太大
                #攻击
                # time.sleep(0.1)
                if target.is_vip: # 如果对方是vip，不考虑留弹，直接击发
                    target.wish_list_by.append(p.ID)
                elif p.is_drone:  # 否则，如果是无人机对无人机，也可
                    target.wish_list_by.append(p.ID)

        # 第四步，评估所有敌机的wish_list，对其中的我方飞机进行排序，距离最近分值最低的飞机发射导弹
        def chose_one_to_attack(op):
            score = []
            for hostid in op.wish_list_by:
                # 均衡射程和剩余弹药的影响， 剩余一个弹药顶20km
                # 选取最近的host发射
                host = self.find_plane_by_id(hostid)
                host_distance_score = self.get_dis(host.ID, op.ID)
                host_weapon_score = -host.LeftWeapon * 20e3
                host_score = host_weapon_score + host_distance_score
                score.append(host_score)
            # 分数从小到大排列, 取得分最小的
            res = np.argsort(score)
            return op.wish_list_by[res[0]]

        #返回距离最近的目标id
        def chose_a_ene_to_attack(p):
            distance_list = []
            target_id_list = []
            for op in self.op_planes:
                distance = self.get_dis(p.ID, op.ID)
                distance_list.append(distance)
                target_id_list.append(op.ID)
            return target_id_list[distance_list.index(min(distance_list))]


        for p in self.my_planes:
            p.fire_cmd_given = False # 每次决策循环，每个飞机只发射一个弹
        # 执行************************************************************************************************@@@@@@@@@@@@
        # op_vip_id = 0
        # for op in self.op_planes:
        #     if op.is_vip: op_vip_id = op.ID


        # for p in self.my_planes:
        #     # attack_target_id = chose_a_ene_to_attack(p)
        #     attack_target_id = chose_vip_to_attack()
        #     if not p.fire_cmd_given: # 每次决策循环，每个飞机只发射一个弹
        #         self.cmd_list.append(
        #             CmdEnv.make_attackparam(p.ID, attack_target_id, 1))
        #         p.fire_cmd_given = True # 每次决策循环，每个飞机只发射一个弹
        #     else:
        #         pass

        for op in self.op_planes:
            if len(op.wish_list_by) == 0:
                continue
            attack_host_id = chose_one_to_attack(op)
            cont = True
            #有人接保留2枚导弹
            for p in self.my_planes:
                if p.is_vip and attack_host_id ==p.ID and p.LeftWeapon == 2:
                    if op.is_vip:
                        #留的2发1打有人接
                        self.cmd_list.append(
                            CmdEnv.make_attackparam(attack_host_id, op.ID, 1))
                        p.fire_cmd_given = True  # 每次决策循环，每个飞机只发射一个弹
                    cont = False
                    continue

            if cont:
                p = self.find_plane_by_id(attack_host_id)
                if not p.fire_cmd_given: # 每次决策循环，每个飞机只发射一个弹
                    self.cmd_list.append(
                        CmdEnv.make_attackparam(attack_host_id, op.ID, 1))
                    p.fire_cmd_given = True # 每次决策循环，每个飞机只发射一个弹
                else:
                    pass
        return

    # 策略主函数
    def make_decision(self,obs_side):
        # 最低优先级，follow
        self.obs_side_1 = obs_side
        self.adjust_attack_order()
        # self.check_follow_condition()
        # if self.situation_eva(obs_side)==True:
        #     print('TRue')
        self.chose_vip_to_attack()
        # self.missile_policy()
        # self.change_persistent_state()

        # 最高优先级
        # state change 《change to escape mode》
        self.check_escape_condition()

        # 处理“列队追踪”指令
        def formation_follow(uav):
            # 第一步，展开队形，按照Fleet_Attack_Angle和Fleet_Attack_Dis展开小队
            n_member = len(uav.mate) + 1
            delta = uav.pos2d - uav.current_target.pos2d
            uav_mates = [self.find_plane_by_name(
                a_mate) for a_mate in uav.mate]
            delta_all = [a_mate.pos2d -
                         uav.current_target.pos2d for a_mate in uav_mates]
            delta_all.append(delta)  # 自己在最后
            # get squad center
            squad_center = uav.pos2d / n_member
            for a_mate in uav_mates:
                squad_center += a_mate.pos2d / n_member
            # rank squad member
            rad_basement = dir2rad(squad_center - uav.current_target.pos2d)
            rad_all = [reg_rad_at(dir2rad(D) - rad_basement, 0)
                       for D in delta_all]
            rank_index = np.argsort(rad_all)
            rank_at = np.where(rank_index == (n_member - 1))[0][0]  # 自己在最后
            # 第二步，根据计算得到的小队位置，计算目标点
            if n_member == 1:
                dst_now3d = uav.current_target.pos3d
            else:
                fan_theta = Fleet_Attack_Angle * (np.pi / 180)
                uav_theta = rad_basement - fan_theta / 2 + \
                    rank_at * (fan_theta / (n_member - 1))
                dst_now2d = Fleet_Attack_Dis * \
                    np.array([np.cos(uav_theta), np.sin(uav_theta)])
                # if uav.Name=="红无人机3": print亮紫('uav.current_target.pos3d', uav.current_target.pos3d)
                dst_now3d = copy.deepcopy(uav.current_target.pos3d)
                dst_now3d[:2] += dst_now2d
            # goto dst_now3d
            goto_location = [
                {"X": dst_now3d[0], "Y": dst_now3d[1], "Z": dst_now3d[2]}]

            # 第三步，根据根据小队距离目标距离是否一致，调整行进速度
            # adjust speed
            this_uav_dis = self.get_dis(uav.ID, uav.current_target.ID)
            squad_target_distance = [self.get_dis(
                a_mate.ID, uav.current_target.ID) for a_mate in uav_mates]
            squad_target_distance.append(this_uav_dis)
            # adjust speed according to distance
            default_speed = 300 * 0.75
            mean_dis = np.array(squad_target_distance).mean()
            this_uav_ideal_dis = mean_dis if uav.is_drone else mean_dis + 50e3
            dis_error = this_uav_dis - this_uav_ideal_dis
            mx = 3e3  # 误差线性调整区间
            max_ = uav.MaxSpeed
            min_ = default_speed * 0.7
            k = (max_ - min_) / mx / 2
            if dis_error > mx:
                speed = max_
            elif dis_error < -mx:
                speed = min_
            else:
                speed = min_ + k * (dis_error + mx)

            # 最后提交指令
            self.cmd_list.append(self.check_and_make_linepatrolparam(
                uav.ID,
                goto_location,
                speed,
                uav.MaxAcc,
                uav.MaxOverload
            ))

        # 处理“混战追踪”指令，追踪攻击序列中的首位敌机
        def follow_cmd(uav):
            # filter dead mate
            mates = [self.find_plane_by_name(mate) for mate in uav.mate]
            mates = [mate for mate in mates if mate is not None]
            uav.mate = [mate.Name for mate in mates if mate is not None]
            uav_and_its_mate = copy.deepcopy(mates)
            uav_and_its_mate.append(uav)
            squad_hostile_distance = [
                self.get_dis(uav.ID, op.ID) for op in self.op_planes for uav in uav_and_its_mate
            ]
            min_squad_hostile_dis = min(squad_hostile_distance)

            if min_squad_hostile_dis >= Fleet_Attack_Dis:
                formation_follow(uav)
                pass
            elif min_squad_hostile_dis < Fleet_Attack_Dis:
                # make emergent tracking
                follow = hasattr(uav, 'bind') and (uav.bind is not None)
                if not follow:
                    return
                self.cmd_list.append(
                    CmdEnv.make_followparam(
                        uav.ID,
                        uav.bind,
                        uav.FollowSpeed,
                        uav.MaxAcc,
                        uav.MaxOverload))
            return

        # 处理“紧急规避准备”指令
        def prepare_escape_cmd(uav):
            # STATE_TRANS
            goto_location = maneuver_angle_to_ms(uav, 180)
            self.cmd_list.append(self.check_and_make_linepatrolparam(
                uav.ID,
                goto_location,
                uav.MaxSpeed,
                uav.MaxAcc,
                uav.MaxOverload
            ))

        # 处理“紧急规避”指令
        def emergent_escape_cmd(uav):
            uav.persistent_state = 'emergent_escape'
            goto_location = maneuver_angle_to_ms(uav, self.escape_angle)
            self.cmd_list.append(self.check_and_make_linepatrolparam(
                uav.ID,
                goto_location,
                uav.MaxSpeed,
                uav.MaxAcc,
                uav.MaxOverload
            ))


        # 任务优先级
        for p in self.my_planes:
            if p.step_state == 'follow':
                # follow_cmd(p)
                pass
            elif p.step_state == 'prepare_escape':
                prepare_escape_cmd(p)
            elif p.step_state == 'emergent_escape':
                emergent_escape_cmd(p)

        return

    # 当追踪飞机的导弹的数量发生变化，重新计算规避方向
    def change_persistent_state(self):
        for p in self.my_planes:
            if p.ms_number_changed():  # 当追踪飞机的导弹的数量发生变化
                p.persistent_state = 'follow'
        pass

    def situation_eva(self, obs_side_1):
        # flag = 0
        our_weapon = np.array(
            [plane['LeftWeapon'] for plane in obs_side_1['platforminfos'] if plane['Availability'] == 1.0])
        our_weapon_nums = our_weapon.sum()
        our_plane_nums = our_weapon.size

        hos_plane_nums = len(
            [plane for plane in obs_side_1['trackinfos'] if plane['Availability'] == 1.0])

        Attack_hos_vip = True if hos_plane_nums - our_weapon_nums >= our_plane_nums else False
        # Protect_our_vip = True if our_plane_nums - hos_weapon_nums >= hos_plane_nums else False

        # if Attack_hos_vip : flag = 1
        # if Protect_our_vip and Attack_hos_vip == False: flag = 2

        return Attack_hos_vip

    def chose_vip_to_attack(self):
        op_vip_id =0
        cont = True
        for op in self.op_planes:
            if op.is_vip:op_vip_id = op.ID

        for p in self.my_planes:
            p.fire_cmd_given = False # 每次决策循环，每个飞机只发射一个弹

        #有人接保留2枚导弹
        for p in self.my_planes:
            if p.is_vip and p.LeftWeapon == 2:
                    #留的2发1打有人接
                self.cmd_list.append(
                    CmdEnv.make_attackparam(p.ID, op_vip_id, 1))
                p.fire_cmd_given = True  # 每次决策循环，每个飞机只发射一个弹
                cont = False
                continue
            if cont:
                if not p.fire_cmd_given: # 每次决策循环，每个飞机只发射一个弹
                    self.cmd_list.append(
                        CmdEnv.make_attackparam(p.ID, op.ID, 1))
                    p.fire_cmd_given = True # 每次决策循环，每个飞机只发射一个弹
                else:
                    pass
        return op_vip_id
