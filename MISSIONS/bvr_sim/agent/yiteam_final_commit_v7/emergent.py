from typing import List
from ..agent import Agent
from ..env_cmd import CmdEnv
from .UTILS.colorful import *
from .UTILS.tensor_ops import dir2rad, np_softmax, reg_rad_at, reg_rad, repeat_at
from .maneuver import maneuver_angle_to_ms, maneuver_angle_to_ms3d, maneuver_speed_to_ms, maneuver_vip
import copy
import random
import numpy as np
import time
from .tools import distance_matrix
from .base import Baseclass, Special, Drone, Vip, Plane, MS
from .missile_policy import MS_policy


class Emergent():

    # 检查紧急避险是否被触发，prepare_escape会将机头指向目标，emergent_escape会将机头调转80度急转弯
    def check_escape_condition(self):
        for p in self.my_planes:
            if p.is_vip:
                p.step_state = 'follow'
                active_ms = [self.find_ms_by_id(ms['ID']) for ms in p.attacked_by_ms]
                active_ms = list(filter(lambda ms: ms.tracking_target, active_ms))
                if len(active_ms) >= 1:
                    dis = [self.get_dis(p.ID, ms.ID) for ms in active_ms]
                    min_ms_dis = min(dis)

            
                    if min_ms_dis <= 45e3:
                        p.step_state = 'vip_180_escape'

                    if min_ms_dis <= 25e3:
                        p.step_state = 'vip_90_escape'

                    # if min_ms_dis <= p.prepare_escape_distance:  # 2e3: # 2000ok 1500ok
                    # if min_ms_dis <= 30e3:  # 2e3: # 2000ok 1500ok
                    #     p.step_state = 'vip_prepare_escape'
                    # high priority
                    # if min_ms_dis <= p.escape_distance:  # 2e3: # 2000ok 1500ok
                    #     p.step_state = 'emergent_escape'
                    # for ms in active_ms:
                    #     ft = ms.flying_time
                    # pass

                else:
                    overall_adv, _ = self.has_overall_adv()
                    if overall_adv:
                        p.step_state = 'advantage_vip_escape'

            if p.is_drone:
                active_ms = [self.find_ms_by_id(ms['ID']) for ms in p.attacked_by_ms]
                active_ms = list(filter(lambda ms: ms.tracking_target, active_ms))
                if len(active_ms) >= 1:
                    dis = [self.get_dis(p.ID, ms.ID) for ms in active_ms]
                    min_ms_dis = min(dis)
                    if min_ms_dis <= p.prepare_escape_distance:  # 2e3: # 2000ok 1500ok
                        p.step_state = 'prepare_escape'

                if len(active_ms) >= 2:
                    # 检索距离最近的两个导弹
                    min_dis = 9999e3
                    ms1_tmp = None
                    ms2_tmp = None
                    for ms1 in active_ms:
                        for ms2 in active_ms:
                            if ms1 is ms2: continue
                            distance = self.get_dis(ms1.ID, ms2.ID) #ms1_tmp = 
                            if distance < min_dis:
                                ms1_tmp = ms1
                                ms2_tmp = ms2
                                min_dis = distance
                    # 获取了距离最近的两个导弹！检查是不是距离小于3km，且属于同一个飞机发射，且flyingtime不同
                    距离小于3km = (min_dis < 3e3) and (min_dis>50)
                    同一个飞机发射 = (ms1_tmp.LauncherID==ms2_tmp.LauncherID)
                    flyingtime不同 = (ms1_tmp.flying_time != ms2_tmp.flying_time)
                    导弹initdistance = (ms1_tmp.distance[0] > 45e3) and (ms2_tmp.distance[0] > 45e3)
                    if 距离小于3km and 同一个飞机发射 and flyingtime不同 and (not 导弹initdistance):
                        print('bk')
                    if 距离小于3km and 同一个飞机发射 and flyingtime不同 and 导弹initdistance:
                        p.step_state = 'reverse_escape'

    def abort_attack_cmd(self, uav, force=False):
        def get_nearest_op_with_ms(p):
            dis = np.array([p.get_dis(op.ID) for op in self.op_planes if op.OpLeftWeapon>0])
            if len(dis)>0:
                return self.op_planes[np.argmin(dis)]
            else:
                return None
        def confirm_abort(uav, target_id):
            if uav.is_vip: 
                _, threat_distance = uav.get_nearest_threat()
                if threat_distance > 30e3:
                    return False
                else: 
                    return True
            if force: return True
            # 无人机
            op = self.find_plane_by_id(target_id)
            ms = uav.nearest_ms()
            if ms is None: return False # do not abort 
            distance_to_ms = self.get_dis(ms.ID, uav.ID)
            ## print亮红('需要决策是否放弃攻击？ 主体：', uav.Name, '攻击目标：', op.Name, ' 规避导弹名称：', ms.Name, ' 规避导弹距离：',  distance_to_ms, '是否放弃？',not ((op in uav.in_attack_radar) and (distance_to_ms>6e3)))
            if (op in uav.in_attack_radar) and (distance_to_ms>6e3):
                return False
            else:
                return True

        self.cmd_list_new = []

        for cmd in self.cmd_list:
            abort_condition = ('CmdAttackControl' in cmd) and (cmd['CmdAttackControl']['HandleID'] == uav.ID) and confirm_abort(uav, cmd['CmdAttackControl']['TgtID'])
            if abort_condition:
                pass
                ## print('warning')
            else:
                self.cmd_list_new.append(cmd)

        self.cmd_list = self.cmd_list_new

    def reverse_escape_cmd(self, uav):
        assert uav.is_drone
        ms = uav.nearest_ms()
        distance_to_ms = self.get_dis(ms.ID, uav.ID)
        if distance_to_ms>8e3:
            goto_location = maneuver_angle_to_ms(uav, 0)
        else:
            goto_location = maneuver_angle_to_ms(uav, 180)
        self.abort_attack_cmd(uav, force=True)
        self.cmd_list.append(self.check_and_make_linepatrolparam(
            uav.ID,
            goto_location,
            uav.MaxSpeed,
            uav.MaxAcc,
            uav.MaxOverload
        ))



    # 处理“紧急规避准备”指令
    def prepare_escape_cmd(self, uav):
        # STATE_TRANS
        if uav.is_vip:
            assert False
            # goto_location = maneuver_angle_to_ms(uav, 110)
            # self.abort_attack_cmd(uav)
            # self.cmd_list.append(self.check_and_make_linepatrolparam(
            #     uav.ID,
            #     goto_location,
            #     uav.MaxSpeed,
            #     uav.MaxAcc,
            #     uav.MaxOverload
            # ))
        else:
            goto_location = maneuver_angle_to_ms3d(uav, uav.escape_angle)
            speed = maneuver_speed_to_ms(uav)
            self.abort_attack_cmd(uav)
            self.cmd_list.append(self.check_and_make_linepatrolparam(
                uav.ID,
                goto_location,
                speed,
                # uav.MaxSpeed,
                uav.MaxAcc,
                uav.MaxOverload
            ))

    # # 处理“紧急规避”指令
    # def emergent_escape_cmd(self, uav):
    #     uav.persistent_state = 'emergent_escape'
    #     goto_location = maneuver_angle_to_ms(uav, self.escape_angle)
    #     self.cmd_list.append(self.check_and_make_linepatrolparam(
    #         uav.ID,
    #         goto_location,
    #         uav.MaxSpeed,
    #         uav.MaxAcc,
    #         uav.MaxOverload
    #     ))

    def vip_90_escape_cmd(self, uav):
        # def adjust_angle(uav):
        #     Angle = 0
        #     if uav.X > 145000 or uav.X < -145000 or uav.Y > 145000 or uav.Y < -145000: 
        #         Angle = 100
        #     return Angle
        # # STATE_TRANS
        # angle = adjust_angle(uav)
        goto_location = maneuver_vip(uav, 90)
        self.abort_attack_cmd(uav)
        self.cmd_list.append(self.check_and_make_linepatrolparam(
            uav.ID,
            goto_location,
            uav.MaxSpeed,
            uav.MaxAcc,
            uav.MaxOverload
        ))

    def vip_0_escape_cmd(self, uav):
        # def adjust_angle(uav):
        #     Angle = 0
        #     if uav.X > 145000 or uav.X < -145000 or uav.Y > 145000 or uav.Y < -145000: 
        #         Angle = 100
        #     return Angle
        # # STATE_TRANS
        # angle = adjust_angle(uav)
        goto_location = maneuver_angle_to_ms(uav, 0)
        self.abort_attack_cmd(uav)
        self.cmd_list.append(self.check_and_make_linepatrolparam(
            uav.ID,
            goto_location,
            uav.MaxSpeed,
            uav.MaxAcc,
            uav.MaxOverload
        ))



    def vip_goaway_when_has_advantage(self, p):
        _, goto_location = self.has_overall_adv()
        self.abort_attack_cmd(p)
        self.cmd_list.append(self.check_and_make_linepatrolparam(
            p.ID,
            goto_location,
            p.MaxSpeed,
            p.MaxAcc,
            p.MaxOverload,
            force_old_way=True
        ))
