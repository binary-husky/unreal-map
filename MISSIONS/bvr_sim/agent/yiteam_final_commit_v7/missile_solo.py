from ..agent import Agent
from ..env_cmd import CmdEnv
from UTILS.colorful import *
from UTILS.tensor_ops import dir2rad, np_softmax, reg_rad_at, reg_deg_at, reg_rad, repeat_at
import numpy as np
from .tools import distance_matrix
import uuid, logging
# class missle_client():
#     def __init__(self, op) -> None:
#         self.op = op
# 同步发射距离 = 1e3

class MS_solo():

    def assign_solo_service(self):
        # 单机打击的条件

        # 时机条件：全部飞机都挤在一个小象限内，或者只有一个飞机还有剩余弹药
        # 位置条件：距离敌机非常近，小于其易逃逸距离
        # 保持距离的需要：有人机希望保持和敌方有人机的距离，发射导弹逼迫距离的拉开

        # if self.time_of_game < 400: 
        #     return # 8分钟多一点点
        op_ms_left_sum = sum([op.OpLeftWeapon for op in self.op_planes])
        # if op_ms_left_sum > 6:
        #     return  # 敌方的导弹多余半数

        for p in self.my_planes:
            # 过滤掉没有导弹的飞机
            if p.n_avail_ms<=0: continue 

            # 谨慎发射
            careful_launch = False

            # 谨慎发射的条件
            作为副攻任务清单为空 = len(p.secondary_request)==0
            不作为主攻 = (p.mspolicy_state == 'default')
            if not 作为副攻任务清单为空: careful_launch=True
            if not 不作为主攻:  careful_launch=True

            # Bool 作为副攻任务清单为空 and 不作为主攻 and 有可用弹
            if p.is_vip:
                self.assign_vip_solo(p, careful_launch)
            else: # is a drone
                self.assign_drone_solo(p, careful_launch)

            pass
        # 抛弃弹药模式：再不打就没机会了
        pass

    def assign_drone_solo(self,p, careful_launch):
        # 将进入攻击死区的敌机排序，按距离从小到大排列
        op_距离从近到远 = sorted(p.in_attack_deadzoo, key=lambda op: self.get_dis(op.ID, p.ID))
        # 过滤掉已经被打击得目标
        op_距离从近到远_滤除已打击 = [op for op in op_距离从近到远
            if (op.under_double_attack==0) and (not op.under_solo_attack)]
        # 如果没有候选，退出
        if len(op_距离从近到远_滤除已打击)<=0: return
        # 如果有，打距离最近者
        打击敌机 = op_距离从近到远_滤除已打击[0]
        # -> confirm_solo_attack
        self.confirm_solo_attack(主机=p, 打击敌机=打击敌机, careful_launch=careful_launch)


    def assign_vip_solo(self, p, careful_launch):
        # 按距离从小到大排列
        op_距离从近到远 = sorted(self.op_planes, key=lambda op: self.get_dis(op.ID, p.ID))
        # 第1次过滤
        op_距离从近到远_滤除已打击 = [op for op in op_距离从近到远
            if (op.under_double_attack==0) and (not op.under_solo_attack)]
        # 第2次过滤
        op_距离从近到远_滤除已打击_过滤距离 = []
        for op in op_距离从近到远_滤除已打击:
            dis = self.get_dis(p.ID, op.ID)
            if op.OpLeftWeapon <= 0:
                # 没有弹药的目标
                dis_lower_bound = 0
                dis_upper_bound = 25e3
            else:
                # 仍有弹药的目标
                dis_lower_bound = 25e3
                dis_upper_bound = 40e3
            if dis>=dis_lower_bound and dis<=dis_upper_bound:
                op_距离从近到远_滤除已打击_过滤距离.append(op)

        # 加入
        if len(op_距离从近到远_滤除已打击_过滤距离)<=0: return
        打击敌机 = op_距离从近到远_滤除已打击_过滤距离[0]
        if careful_launch and (打击敌机 not in p.in_attack_deadzoo): 
            return
        self.confirm_solo_attack(主机=p, 打击敌机=打击敌机, careful_launch=careful_launch)

    def last_stand_service(self):
        只有最后一发导弹 = sum([p.LeftWeapon for p in self.my_planes])==1
        没有在途的其他我方导弹 = len([ms for ms in self.ms if not ms.is_op_ms])==0
        plane_num_adv = len(self.my_planes) - len(self.op_planes)       # 我战机数量减敌战机数量
        处于战机数量劣势 = (plane_num_adv<0)

        # 此时只有追击对方有人机才能反转
        if 只有最后一发导弹 and 没有在途的其他我方导弹 and 处于战机数量劣势:
            我方有人机 = [p for p in self.my_planes if p.is_vip][0]
            敌方有人机 = [op for op in self.op_planes if op.is_vip][0]
            我方有人机.ms_policy_suggested_target = 敌方有人机
            我方有人机.last_stand = True
            if self.time_of_game > 1010: # 没时间了
                self.confirm_solo_attack(主机=我方有人机, 打击敌机=敌方有人机, careful_launch=False)
            else:
                dis = self.get_dis(我方有人机.ID, 敌方有人机.ID)
                if dis < 8e3:
                    self.confirm_solo_attack(主机=我方有人机, 打击敌机=敌方有人机, careful_launch=False)

    def confirm_solo_attack(self, 主机, 打击敌机, careful_launch):
        solo_uuid = uuid.uuid1().hex
        self.solo_list[solo_uuid] = {
            'target_to_hit': 打击敌机,
            'host_uav': 主机,
            'solo_uuid': solo_uuid,
            'ms': None,
            'valid': True
        }
        主机.ms_to_launch.append({
            'target_ID_to_hit': 打击敌机.ID,
            'launch_state': None,
            'related_uuid': solo_uuid,
            'type': 'solo_ms',
            'careful_launch': careful_launch,
            'req_finished':False
        })
        打击敌机.under_solo_attack = True
        if 主机.is_drone: 主机.ms_policy_suggested_target = 打击敌机
        else: self.notify_drones_to_track(打击敌机, solo_uuid)
        logging.info('预案 ☆☆☆☆☆☆ 候选独奏模式, 主攻:%s, 打击目标:%s'%(主机.Name, 打击敌机.Name))
        logging.info('☆☆☆ 独奏导弹请求已添加')
        return

    def notify_drones_to_track(self, 打击敌机, solo_uuid):
        ## print亮红('尚未实现的功能')
        pass