from typing import List
from agent.agent import Agent
from ..env.env_cmd import CmdEnv
from utils.utils_math import TSVector3
from .UTILS.colorful import *
from .UTILS.tensor_ops import dir2rad, np_softmax, reg_rad_at, reg_rad, repeat_at
from .maneuver import maneuver_angle_to_ms, maneuver_angle_to_ms3d, maneuver_speed_to_ms, maneuver_vip, maneuver_angle_to_op_vip
import copy
import random
import numpy as np
import time
from .tools import distance_matrix
from .base import Baseclass, Special, Drone, Vip, Plane, MS
from .missile_policy import MS_policy
from .emergent import Emergent
from .attack_seq_adjust import Attack_Adjust
# Enable_Evade = True

Fleet_Attack_Angle = 110
Fleet_Attack_Dis = 70e3

#无人机目标分配
#有人机不要太靠近

class Yi_team(Baseclass, MS_policy, Emergent, Attack_Adjust):
    def __init__(self, name, config):
        super(Baseclass, self).__init__(name, config["side"])
        self.init()
        self.times=0
        self.obs_side_1 = []


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

    #
    def get_in_target_zoo(self, op):
        in_radar = []
        for p in self.my_planes:
            dis = self.get_dis(op.ID, p.ID)
            if dis > op.Flying_to_distance:
                continue
            # 检查角度是否满足
            delta = p.pos2d - op.pos2d
            theta = 90 - op.Heading * 180 / np.pi
            theta = reg_rad(theta * np.pi / 180)
            delta2 = np.matmul(np.array([[np.cos(theta), np.sin(theta)],
                                         [-np.sin(theta), np.cos(theta)]]), delta.T)
            deg = dir2rad(delta2) * 180 / np.pi
            if deg >= -op.DeadZooRadarHorizon and deg <= op.DeadZooRadarHorizon:
                in_radar.append(p)
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

    # 获取照射追踪我方的敌方导弹的无人机
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


    # 执行进攻序列，删除已经消灭的目标
    def check_follow_condition(self):
        for p in self.my_planes:
            while True:
                # 找到下一个没死的敌方目标
                if len(p.attack_order) == 0:
                    p.current_target = p
                    break
                attack_target = self.find_plane_by_name(p.attack_order[0])
                if attack_target is None:
                    p.attack_order.pop(0)
                    continue
                else:
                    p.current_target = attack_target
                    break

        pass


    @staticmethod
    def thresh_hold_projection(x, min_x, y_min_x, max_x, y_max_x):
        assert min_x < max_x

        if x <= min_x:
            return y_min_x
        if x >= max_x:
            return y_max_x

        # 处于线性区间内
        k = (y_max_x - y_min_x)/(max_x - min_x)
        dx = x - min_x
        return dx*k + y_min_x
    
    @staticmethod
    def shrink_2d_vec(p, dst, distance):
        vec2dst = dst[:2] - p.pos2d
        dst_now2d = vec2dst / (np.linalg.norm(vec2dst) + 1e-7) * distance
        return p.pos2d + dst_now2d

    # 处理“列队追踪”指令
    def formation_follow(self, uav):
        # 第一步，展开队形，按照Fleet_Attack_Angle和Fleet_Attack_Dis展开小队
        ## print亮红(uav.Name, '--->', uav.current_target.Name)

        n_member = len(uav.formation_mate) + 1
        delta = uav.pos2d - uav.current_target.pos2d
        uav_mates = [self.find_plane_by_name(
            a_mate) for a_mate in uav.formation_mate]
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
        if n_member == 1: # or uav.is_vip:
            dst_now3d = uav.current_target.pos3d
        else:
            fan_theta = Fleet_Attack_Angle * (np.pi / 180)
            uav_theta = rad_basement - fan_theta / 2 + rank_at * (fan_theta / (n_member - 1))
            #*******************************队形*距离********************************************************
            vec2d = Fleet_Attack_Dis *np.array([np.cos(uav_theta), np.sin(uav_theta)])
            dst_now3d = copy.deepcopy(uav.current_target.pos3d)
            dst_now3d[:2] += vec2d
            ### goto dst_now3d

        # 修改， 将编队高度设为最高高度
        # # 修改， 将目标的距离缩短到50km
        if uav.is_vip:
            dst_now2d = dst_now3d[:2]
            dst_now2d = self.shrink_2d_vec(p=uav, dst=dst_now2d, distance=10e3)
            goto_location = [{"X": dst_now2d[0], "Y": dst_now2d[1], "Z": uav.MaxHeight}]
        if uav.is_drone:
            dst_now2d = dst_now3d[:2]
            dst_now2d = self.shrink_2d_vec(p=uav, dst=dst_now2d, distance=20e3)
            goto_location = [{"X": dst_now2d[0], "Y": dst_now2d[1], "Z": 8500}]
            # 根据敌方的无人机高度
            get_op_uav_avg_height = [op.Z for op in self.op_planes if op.is_drone]
            if len(get_op_uav_avg_height) > 0:
                avg_height = np.array(get_op_uav_avg_height).mean()
                goto_location = [{"X": dst_now2d[0], "Y": dst_now2d[1], "Z": avg_height}]

        # 第三步，根据根据小队距离目标距离是否一致，调整行进速度
        # adjust speed
        this_uav_dis = self.get_dis(uav.ID, uav.current_target.ID)
        squad_target_distance = [self.get_dis(
            a_mate.ID, uav.current_target.ID) for a_mate in uav_mates]
        squad_target_distance.append(this_uav_dis)
        # adjust speed according to distance

        # 以距离敌方最远的无人机作为基准
        _distance02 = [self.get_dis(a_mate.ID, uav.current_target.ID) for a_mate in uav_mates if a_mate.is_drone]
        if uav.is_drone: 
            _distance02.append(this_uav_dis)
        base_line_distance = max(_distance02)
        
        this_uav_ideal_dis = base_line_distance if uav.is_drone else base_line_distance + 15e3
        
        dis_error = this_uav_dis - this_uav_ideal_dis
        # mx = 3e3  # 误差线性调整区间

        speed = self.thresh_hold_projection(dis_error,
                    min_x=-3e3, y_min_x=uav.MaxSpeed*0.5,
                    max_x=0, y_max_x=uav.MaxSpeed)
        if uav.is_vip:
            speed = self.thresh_hold_projection(dis_error,
                    min_x=-4e3, y_min_x=200,
                    max_x=4e3, y_max_x=400)
        # 最后提交指令
        self.cmd_list.append(self.check_and_make_linepatrolparam(
            uav.ID,
            goto_location,
            speed,
            uav.MaxAcc,
            uav.MaxOverload
        ))

    def close_target_tracking(self, uav, op):
        vec3d = op.pos3d - uav.pos3d
        goto_location = maneuver_angle_to_op_vip(uav=uav, vip=op, angle=110)
        
        return goto_location

    # 处理“混战追踪”指令，追踪攻击序列中的首位敌机
    def follow_cmd(self, uav):
        # filter dead mate
        mates = [self.find_plane_by_name(mate) for mate in uav.formation_mate]
        mates = [mate for mate in mates if mate is not None]
        uav.formation_mate = [mate.Name for mate in mates if mate is not None]
        uav_and_its_mate = copy.copy(mates)
        uav_and_its_mate.append(uav)
        squad_hostile_distance = [
            self.get_dis(uav.ID, op.ID) for op in self.op_planes for uav in uav_and_its_mate
        ]
        min_squad_hostile_dis = min(squad_hostile_distance)
        # ## print(min_squad_hostile_dis)
        if min_squad_hostile_dis >= Fleet_Attack_Dis and (not self.Fleet_Attack):
            self.formation_follow(uav)
            pass
        elif min_squad_hostile_dis < Fleet_Attack_Dis:
            self.Fleet_Attack = True
            # make emergent tracking
            follow = hasattr(uav, 'current_target') and (uav.current_target is not None)
            if not follow:
                return

            没有在途的其他uav导弹 = (len([ms for ms in self.ms if (ms.host is uav)])==0)


            if uav.is_drone and uav.调试_正在被高度优势vip导弹追踪_且距离小于25km_估计撞击点预计在万米之上():
                # 由于状态机的控制，此时的飞机-导弹距离必大于12km
                # 要放弃目标吗？放弃
                # 为了更自由地控制，使用make_linepatrolparam指令替换followparam指令
                ## print亮红('触发！调试_正在被高度优势vip导弹追踪_且距离小于25km_估计撞击点预计在万米之上')
                self.abort_attack_cmd(uav, force=True)
                goto_location = uav.调试_正在被高度优势vip导弹追踪_sudden_dive_towards_goto_location()
                self.cmd_list.append(self.check_and_make_linepatrolparam(
                    uav.ID,
                    goto_location,
                    uav.MaxSpeed,
                    uav.MaxAcc,
                    uav.MaxOverload
                ))
            elif uav.LeftWeapon==0 and 没有在途的其他uav导弹 and uav.is_drone and uav.current_target.is_vip and self.get_dis(uav.ID, uav.current_target.ID)<14e3:
                speed = uav.FollowSpeed
                goto_location = self.close_target_tracking(uav, uav.current_target)
                self.cmd_list.append(self.check_and_make_linepatrolparam(
                    uav.ID,
                    goto_location,
                    uav.MaxSpeed,
                    uav.MaxAcc,
                    uav.MaxOverload
                ))
            else: # 正常情况
                speed = uav.FollowSpeed
                if uav.is_vip:
                    _, threat_distance = uav.get_nearest_threat()
                    if threat_distance < 30e3: speed = 400 # uav.MaxSpeed
                    elif threat_distance < 40e3: speed = 300
                    else: speed = 300
                    if hasattr(uav, 'last_stand'): speed = 400
                    goto_location = [{
                        "X": uav.current_target.X, "Y": uav.current_target.Y, "Z": uav.MaxHeight*0.9
                    }]
                    self.cmd_list.append(self.check_and_make_linepatrolparam(
                        uav.ID,
                        goto_location,
                        speed,
                        uav.MaxAcc,
                        uav.MaxOverload
                    ))
                else:
                    self.cmd_list.append(
                        CmdEnv.make_followparam(
                            uav.ID,
                            uav.current_target.ID,
                            uav.MaxSpeed,
                            uav.MaxAcc,
                            uav.MaxOverload))
        return

    # 当追踪飞机的导弹的数量发生变化，重新计算规避方向
    def change_persistent_state(self):
        for p in self.my_planes:
            if p.ms_number_changed():  # 当追踪飞机的导弹的数量发生变化
                p.persistent_state = 'follow'
        pass



    # 策略主函数
    def make_decision(self):
        ## print('---------')
        # 设置临时状态机的初始状态
        for p in self.my_planes: p.step_state = 'follow'

        self.change_persistent_state()      # 模块1
        self.adjust_attack_order()          # 模块2
        self.missile_policy()               # 模块3 需要调整模块2的结果
        self.adjust_attack_order_phase2()   # 模块4 在模块3之后

        self.check_follow_condition()       # 模块5 在模块2之后

        self.check_escape_condition()       # 模块6 最高优先级

        # 任务优先级
        for p in self.my_planes:
            if p.step_state == 'follow':
                self.follow_cmd(p)
            elif p.step_state == 'prepare_escape':
                self.prepare_escape_cmd(p)
            elif p.step_state == 'reverse_escape':
                self.reverse_escape_cmd(p)
            elif p.step_state == 'vip_90_escape':
                self.vip_90_escape_cmd(p)
            elif p.step_state == 'vip_180_escape':
                self.vip_0_escape_cmd(p)
            elif p.step_state== 'advantage_vip_escape':
                self.vip_goaway_when_has_advantage(p)
        
        self.previous_cmd_list = copy.deepcopy(self.cmd_list)