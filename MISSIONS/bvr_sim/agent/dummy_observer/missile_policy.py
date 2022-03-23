from ..env_cmd import CmdEnv
from UTILS.colorful import *
from UTILS.tensor_ops import dir2rad, np_softmax, reg_rad_at, reg_deg_at, reg_rad, repeat_at
import numpy as np
from .tools import distance_matrix
import uuid
from .missile_exe import MS_exe
from .missile_solo import MS_solo
# class missle_client():
#     def __init__(self, op) -> None:
#         self.op = op
# 同步发射距离 = 1e3

class MS_policy(MS_exe, MS_solo):

    # 只应该在每局的第一步执行这个初始化
    def __mspolicy_init__(self):
        self.coop_list = {}
        self.solo_list = {}
        ## print亮绿('☆☆ 导弹策略初始化')
        for p in self.my_planes:
            if not hasattr(p, 'mspolicy_state'):
                p.mspolicy_state = 'default'
                p.secondary_request = []
                p.ms_to_launch = []
                p.primary_task_uuid = None
                p.ms_policy_suggested_target = None
        for op in self.op_planes: 
            if not hasattr(op, 'under_double_attack'): 
                op.under_double_attack = 0
                op.under_solo_attack = False
                op.max_angle_cross_fired = None
                op.max_angle_cross_fired_info = None

    def ms_check_unable_launch(self):
        for p in self.my_planes:
            p.on_hold_ms = 0
            if p.mspolicy_state == 'primary_waiting':
                # 处于协同进攻的等待阶段，有一个导弹onhold
                p.on_hold_ms += 1
                coop_info = self.coop_list[p.primary_task_uuid]
                if coop_info['third_ms_needed']:
                    p.on_hold_ms += 1
            for ms_todo in p.ms_to_launch:
                assert not ms_todo['req_finished']  # 假定已经完成的请求在上一个循环中已经被清理干净了
                if ms_todo['launch_state'] is None: 
                    p.on_hold_ms += 1    # 上次的发射请求可能在 None 阶段，（每个时刻能发射几个导弹？）
                if ms_todo['launch_state'] == 'given_cmd': 
                    has_ms = [
                        ms for ms in self.ms 
                        if ms.EngageTargetID==ms_todo['target_ID_to_hit'] and
                        ms.LauncherID==p.ID and
                        ms.flying_time==0
                    ]   # 筛选出刚发射的导弹
                    if len(has_ms)==0:  
                        p.on_hold_ms += 1    # 上次的发射请求可能在 given_cmd 阶段， 但没能成功执行发射
                    if len(has_ms)==2:
                        pass
                        ## print亮红('wtf??')

    # 将coop_info转换为导弹层级的指令
    def execute_ms_coop(self):
        # 先处理副攻请求，有些主攻战机可能担任副攻任务. 副攻的任务：追踪敌机，直到确认导弹成功发射
        for p in self.my_planes:
            if len(p.secondary_request)<=0: continue
            # 逐个处理secondary request， 每次一个
            req_uuid = p.secondary_request.pop(0)   # FIFO处理
            p.ms_to_launch.append({
                'target_ID_to_hit': self.coop_list[req_uuid]['target_to_hit'].ID,
                'launch_state': None,
                'related_uuid': req_uuid,
                'type': 'prior_ms',
                'req_finished':False
            })
            ## print亮绿('☆☆☆ 副攻导弹请求已添加')

        # 处理独奏. 主攻的任务 1、发射导弹，追踪敌机，2、继续追踪敌机，直到导弹进入末制导
        for uuid_ in self.solo_list:
            solo_info = self.solo_list[uuid_]
            p = solo_info['host_uav']
            敌机 = solo_info['target_to_hit']
            ms = solo_info['ms']
            if not 敌机.alive:
                solo_info['valid'] = False
            if not solo_info['valid']:
                p.ms_policy_suggested_target = None
            if ms is None: continue
            if ms.ms_at_terminal:
                p.ms_policy_suggested_target = None

        # # 再处理主攻请求. 主攻的任务 1、等待副攻的导弹到位，追踪敌机，2、发射导弹，追踪敌机，3、继续追踪敌机，直到导弹进入末制导
        # for p in self.my_planes:
        #     if 'primary' not in p.mspolicy_state: continue  # 不是主攻
        #     if p.mspolicy_state == 'primary_waiting':   # 等待导弹发射+追踪的状态
        #         # 检查是否达到了发射距离
        #         assert p.primary_task_uuid is not None
        #         coop_info = self.coop_list[p.primary_task_uuid]
        #         if coop_info['prior_ms'] is None: continue   # 先发导弹暂不存在
        #         敌机 = coop_info['target_to_hit']
        #         先发导弹 = coop_info['prior_ms']
        #         敌方与导弹的距离 = self.get_dis(先发导弹.ID, 敌机.ID)
        #         自己与敌方的距离 = self.get_dis(p.ID, 敌机.ID)
        #         deg1 = self.get_points_angle_deg(pt1=p.pos2d, pt_center=敌机.pos2d, pt2=敌机.pos2d_prediction())
        #         deg2 = self.get_points_angle_deg(pt1=先发导弹.pos2d, pt_center=敌机.pos2d, pt2=敌机.pos2d_prediction())
        #         delta_angle_deg = deg1 - deg2
        #         同步发射距离 = self.thresh_hold_projection(
        #             x=delta_angle_deg, 
        #             min_x=-90, y_min_x=-2.4e3,  # 后发先到，说明提前多，减少y提前值
        #             max_x= 90,  y_max_x= 5.5e3)   
        #         # center = （5-1.5）/2=1.75
        #         # center = （5-2）/2=1.5
        #         # center = （5-2.5）/2=1.25
        #         if (敌方与导弹的距离-自己与敌方的距离) < 同步发射距离:
        #             # 同步发射
        #             p.ms_to_launch.append({
        #                 'target_ID_to_hit': 敌机.ID,
        #                 'launch_state': None,
        #                 'related_uuid': p.primary_task_uuid,
        #                 'type': 'follow_ms',
        #                 'req_finished':False
        #             })
        #             p.mspolicy_state = 'primary_tracking'
        #             ## print亮绿('☆☆☆ 主攻导弹发射机会，敌机：',敌机.Name, '主机：', p.Name, 
        #                 'deg后发%.2f， deg先发%.2f, 差距%.2f'%(deg1,deg2, deg1 - deg2), ' 同步发射距离:', 同步发射距离)
        #             pass
        #         else: # 距离没有达到同步条件
        #             continue
        #     elif p.mspolicy_state == 'primary_tracking':
        #         coop_info = self.coop_list[p.primary_task_uuid]
        #         敌机 = self.coop_list[p.primary_task_uuid]['target_to_hit']
        #         if not 敌机.alive:
        #             coop_info['valid'] = False

        #         if not coop_info['valid']:
        #             # 处于某种原因，合作已经没有必要了，即刻终止
        #             p.mspolicy_state = 'default'
        #             p.primary_task_uuid = None
        #             coop_info['primary_uav'].ms_policy_suggested_target = None
        #             coop_info['secondary_uav'].ms_policy_suggested_target = None

        #         if coop_info['follow_ms'] is None: continue   # 后发导弹暂不存在
        #         assert coop_info['prior_ms'] is not None
        #         # 两个导弹应该都有了
        #         # 当两个导弹都进入末制导，任务结束，清理相关变量
        #         if coop_info['follow_ms'].ms_at_terminal and coop_info['prior_ms'].ms_at_terminal:
        #             ## print亮绿('☆☆☆ 两枚导弹进入末制导', coop_info['follow_ms'].Name, coop_info['prior_ms'].Name)
        #             p.mspolicy_state = 'default'
        #             p.primary_task_uuid = None
        #             coop_info['primary_uav'].ms_policy_suggested_target = None
        #             coop_info['secondary_uav'].ms_policy_suggested_target = None

###################################################################################
        # 再处理主攻请求. 主攻的任务 1、等待副攻的导弹到位，追踪敌机，2、发射导弹，追踪敌机，3、继续追踪敌机，直到导弹进入末制导
        for p in self.my_planes:
            if 'primary' not in p.mspolicy_state: continue  # 不是主攻

            assert p.mspolicy_state in ['primary_waiting', 'primary_tracking']   # 等待导弹发射+追踪的状态
            assert p.primary_task_uuid is not None

            if p.primary_task_uuid not in self.coop_list: # 因为无效，已经被清理了
                p.mspolicy_state = 'default'
                p.primary_task_uuid = None
                continue

            # 检查是否达到了发射距离
            coop_info = self.coop_list[p.primary_task_uuid]
            先发导弹 = coop_info['prior_ms']
            if 先发导弹 is None: 
                assert p.mspolicy_state == 'primary_waiting'
                assert not coop_info['follow_ms_cmd_given']
                assert not coop_info['third_ms_cmd_given']
                continue   # 先发导弹暂不存在

            敌机 = coop_info['target_to_hit']

            if not 敌机.alive: 
                coop_info['valid'] = False
            if (先发导弹 is not None) and (not coop_info['follow_ms_cmd_given']) and (self.find_ms_by_id(先发导弹.ID) is None):
                # 先发导弹不知为什么在follow ms发射之前就没了
                coop_info['valid'] = False

            if not coop_info['valid']:
                # 处于某种原因，合作已经没有必要了，即刻终止
                p.mspolicy_state = 'default'
                p.primary_task_uuid = None
                coop_info['primary_uav'].ms_policy_suggested_target = None
                coop_info['secondary_uav'].ms_policy_suggested_target = None
                continue

            if (先发导弹 is not None) and \
                    (not coop_info['follow_ms_cmd_given']) and \
                    self.can_fire_following(p, 敌机, ref_ms=先发导弹, 提前距离=0):
                p.ms_to_launch.append({
                    'target_ID_to_hit': 敌机.ID,
                    'launch_state': None,
                    'related_uuid': p.primary_task_uuid,
                    'type': 'follow_ms',
                    'req_finished':False
                })
                coop_info['follow_ms_cmd_given'] = True
                p.mspolicy_state = 'primary_tracking'

            if (coop_info['third_ms_needed']) and (coop_info['follow_ms_cmd_given']) and \
                    (not coop_info['third_ms_cmd_given']) and \
                    self.can_fire_following(p, 敌机, ref_ms=先发导弹, 提前距离=(-3e3)):
                p.ms_to_launch.append({
                    'target_ID_to_hit': 敌机.ID,
                    'launch_state': None,
                    'related_uuid': p.primary_task_uuid,
                    'type': 'third_ms',
                    'req_finished':False
                })
                coop_info['third_ms_cmd_given'] = True
                p.mspolicy_state = 'primary_tracking'

            if p.mspolicy_state == 'primary_tracking':
                if coop_info['follow_ms'] is None: continue
                if (coop_info['third_ms_needed']) and (coop_info['third_ms'] is None): continue
                assert coop_info['prior_ms'] is not None
                assert coop_info['follow_ms_cmd_given']
                # 两个导弹应该都有了
                # 当两个导弹都进入末制导，任务结束，清理相关变量，第三发导弹不再考虑范围内
                if coop_info['follow_ms'].ms_at_terminal and coop_info['prior_ms'].ms_at_terminal:
                    if (not coop_info['third_ms_needed']) or (coop_info['third_ms_needed'] and coop_info['third_ms'].ms_at_terminal):
                        ## print亮绿('☆☆☆ 两/三枚导弹进入末制导',  coop_info['follow_ms'].Name,  coop_info['prior_ms'].Name)
                        p.mspolicy_state = 'default'
                        p.primary_task_uuid = None
                        coop_info['primary_uav'].ms_policy_suggested_target = None
                        coop_info['secondary_uav'].ms_policy_suggested_target = None



    # 在 coop_list 高层调整
    def assign_primary_and_secondary(self, 敌机):
        # “不是其他敌机的主攻”，“敌机在该飞机的射程内”，“有导弹” ,“非vip” 4个条件
        my_canfire_planes = [
            p for p in self.my_planes 
            if (p.n_avail_ms > 0) and 
            ('primary' not in p.mspolicy_state) and
            p.is_drone and
            self.get_dis(敌机.ID, p.ID) < p.RadarDis
        ]
        友机列表_根据距离排序 = sorted(my_canfire_planes, key=lambda p:self.get_dis(敌机.ID, p.ID))
        if len(友机列表_根据距离排序) == 0: return # 没有适合作为主攻的友机

        tmp_coop_list_uuid = []
        # 选择最适合的主攻
        for 主攻 in 友机列表_根据距离排序:
            主攻.mspolicy_state = 'primary_waiting' # ☆☆☆☆
            res, coop_uuid = self.选择副攻(主攻=主攻, 打击目标=敌机)
            if res == 'CoopOptionGenerated': tmp_coop_list_uuid.append(coop_uuid)
            主攻.mspolicy_state = 'default' # 先恢复状态，等一会再处理

        tmp_coop_list = [self.coop_list[coop_uuid] for coop_uuid in tmp_coop_list_uuid]
        tmp_coop_list = sorted(tmp_coop_list, key=lambda coop: -coop['cross_fire_angle'])
        if len(tmp_coop_list) <= 0: return
        # 选取夹角最大的一组
        tmp_coop = tmp_coop_list[0]
        主攻 = tmp_coop['primary_uav']
        副攻 = tmp_coop['secondary_uav']
        coop_uuid = tmp_coop['coop_uid']
        cross_fire_angle = tmp_coop['cross_fire_angle']
        max_cross_fired_angle = 敌机.max_angle_cross_fired
        cross_angle_threshold = min(max_cross_fired_angle*0.6666, 30)
        # transport
        
        夹角太小_放弃此次协作 = (cross_fire_angle < cross_angle_threshold)
        if 夹角太小_放弃此次协作:
            # ## print绿('预案 ☆☆☆☆ 夹角太小_放弃此次协作, 没有合适的副攻候选')
            return

        tmp_coop['valid'] = True    # 让协作模式生效
        主攻.mspolicy_state = 'primary_waiting'  # ☆☆☆☆
        主攻.primary_task_uuid = coop_uuid
        主攻.ms_policy_suggested_target = 敌机

        副攻.secondary_request.append(coop_uuid) # ☆☆☆☆
        副攻.ms_policy_suggested_target = 敌机
        敌机.under_double_attack += 1

        # if 敌机.under_double_attack>0:
            ## print('敌机.under_double_attack', 敌机.under_double_attack)
        # assert not 敌机.under_double_attack
        ## print亮蓝('☆☆☆☆☆☆ 生成协作模式, 主攻:', 主攻.Name, ' 副攻:',副攻.Name, '打击目标:',敌机.Name)
        ## print亮蓝('☆☆☆☆☆☆ 生成协作模式完毕')



    def can_fire_following(self, p, 敌机, ref_ms, 提前距离):
        # 先发导弹 = coop_info['prior_ms']
        先发导弹 = ref_ms
        敌方与导弹的距离 = self.get_dis(先发导弹.ID, 敌机.ID)
        自己与敌方的距离 = self.get_dis(p.ID, 敌机.ID)
        deg1 = self.get_points_angle_deg(pt1=p.pos2d, pt_center=敌机.pos2d, pt2=敌机.pos2d_prediction())
        deg2 = self.get_points_angle_deg(pt1=先发导弹.pos2d, pt_center=敌机.pos2d, pt2=敌机.pos2d_prediction())
        delta_angle_deg = deg1 - deg2
        同步发射距离 = self.thresh_hold_projection(
            x=delta_angle_deg, 
            min_x=-90, y_min_x=-2.4e3,  # 后发先到，说明提前多，减少y提前值
            max_x= 90,  y_max_x= 5.5e3)   
        if (敌方与导弹的距离-自己与敌方的距离-提前距离) < 同步发射距离: 
            ## print亮绿('☆☆☆ 主攻导弹发射机会，敌机：',敌机.Name, '主机：', p.Name,  'deg后发%.2f， deg先发%.2f, 差距%.2f'%(deg1,deg2, deg1 - deg2), ' 同步发射距离:', 同步发射距离)
            return True
        else: return False



    def 清理无效的coop和solo(self):
        for uuid in self.coop_list:
            coop_info = self.coop_list[uuid]
            # assert coop_info['valid']
            pri_ms_done = (coop_info['prior_ms'] is not None)  and ((not coop_info['prior_ms'].alive)  or (not coop_info['prior_ms'].tracking_target))
            flo_ms_done = (coop_info['follow_ms'] is not None) and ((not coop_info['follow_ms'].alive) or (not coop_info['follow_ms'].tracking_target))
            if pri_ms_done and flo_ms_done:
                ## print亮绿('☆☆☆ 导弹生命周期完结，coop_list将被删除')
                if coop_info['target_to_hit'].alive:  # 打击失败
                    coop_info['target_to_hit'].under_double_attack -= 1
                    coop_info['valid'] = False
                else: # 打击成功， 敌方阵亡
                    coop_info['valid'] = False
        # 清理掉无效的, key_and_value是tuple
        self.coop_list = dict(filter(lambda key_and_value: key_and_value[1]['valid'], self.coop_list.items()))
 
        for uuid in self.solo_list:
            solo_info = self.solo_list[uuid]
            # assert solo_info['valid']
            ms_done = (solo_info['ms'] is not None) and ((not solo_info['ms'].alive) or (not solo_info['ms'].tracking_target))
            if ms_done:
                ## print亮绿('☆☆☆ 导弹生命周期完结，solo_list将被删除')
                if solo_info['target_to_hit'].alive:  # 打击失败
                    assert solo_info['target_to_hit'].under_solo_attack
                    solo_info['target_to_hit'].under_solo_attack = False
                solo_info['valid'] = False
        self.solo_list = dict(filter(lambda key_and_value: key_and_value[1]['valid'], self.solo_list.items()))
                    

    def overall_cross_fire_angle_analyse(self):
        for op in self.op_planes: 
            max_angle_cross_fired = 0
            max_angle_cross_fired_info = None
            for p1 in self.my_planes:
                for p2 in self.my_planes:
                    if p2 is p1: continue
                    delta_deg = self.get_angle_deg(p1=p1, p_center=op, p2=p2)
                    if delta_deg <= max_angle_cross_fired: continue
                    max_angle_cross_fired = delta_deg
                    max_angle_cross_fired_info = {'p1':p1,'p2':p2}
            op.max_angle_cross_fired = max_angle_cross_fired
            op.max_angle_cross_fired_info = max_angle_cross_fired_info









    
    def score副攻_(self, 主攻, 副攻, 打击目标):
        distance = self.get_dis(打击目标.ID, 副攻.ID)
        delta_deg = self.get_angle_deg(p1=副攻, p_center=打击目标, p2=主攻)
        assert delta_deg <= 180
        constant = 10e3/45  # 10km / 45deg
        return distance - (delta_deg-45)*constant


    def 选择副攻(self, 主攻, 打击目标):
        op = 打击目标
        主攻距离 = self.get_dis(op.ID, 主攻.ID)
        # 选择最适合的副攻，可以是vip
        副攻候选 = [
            p for p in self.my_planes 
            if (p.n_avail_ms > 0) and 
            ('primary' not in p.mspolicy_state) and
            self.get_dis(op.ID, p.ID) < p.RadarDis
        ]
        # 有些已经分配主攻友机恰巧也能做副攻，因为恰巧在雷达射程内
        副攻候选_2 = [
            p for p in self.my_planes 
            if (p.n_avail_ms > 0) and 
            ('primary' in p.mspolicy_state) and
            (op in p.in_radar)
        ]
        副攻候选 = 副攻候选 + 副攻候选_2
        if len(副攻候选)<=0:
            ## print绿('预案 ☆☆☆☆ 如果没有合适的副攻候选')
            return 'NoResult', None

        # 如果有合适的副攻候选
        # if len(副攻候选)>0:
        副攻候选 = [
            p for p in 副攻候选
            if self.get_dis(op.ID, p.ID) > 主攻距离
        ]
        副攻候选_按距离排序_施加角度的影响 = sorted(副攻候选, key=lambda p:self.score副攻_(主攻=主攻, 副攻=p, 打击目标=op))

        if len(副攻候选_按距离排序_施加角度的影响)<=0:
            ## print绿('预案 ☆☆☆☆ 如果没有合适的副攻候选')
            return 'NoResult', None

        # if len(副攻候选_按距离排序_施加角度的影响)>0:
        副攻 = 副攻候选_按距离排序_施加角度的影响[0]    # 最近，且夹角较大
        cross_fire_angle = self.get_angle_deg(p1=副攻, p_center=op, p2=主攻)

        # 在这里生成合作
        # 副攻.mspolicy_state = 'secondary_fire' # ☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆
        third_ms_needed = True if (op.is_vip and 副攻.n_avail_ms>=2) else False

        coop_uuid = uuid.uuid1().hex
        self.coop_list[coop_uuid] = {
            'target_to_hit': op,
            'primary_uav': 主攻,
            'secondary_uav': 副攻,
            '主攻名称': 主攻.Name,
            '副攻名称': 副攻.Name,
            '敌机名称': op.Name,
            'coop_uid': coop_uuid,
            'prior_ms': None,
            'follow_ms': None,
            'follow_ms_cmd_given': False,
            'third_ms': None,
            'third_ms_cmd_given': False,
            'third_ms_needed': third_ms_needed,
            'cross_fire_angle': cross_fire_angle,
            'valid': False
        }
        ## print蓝('预案 ☆☆☆☆☆☆ 候选协作模式, 主攻:', 主攻.Name, ' 副攻:',[p.Name for p in 副攻候选], '打击目标:',op.Name)
        ## print蓝('预案 ☆☆☆☆☆☆ 排序候选协作模式, 主攻:', 主攻.Name, ' 副攻:',[p.Name for p in 副攻候选_按距离排序_施加角度的影响], '打击目标:',op.Name)
        return 'CoopOptionGenerated', coop_uuid




    # 调整敌方接受服务的顺序
    def determine_op_rank(self, op_planes):
        threat = np.array([0 for _ in op_planes])
        # self.clients = [missle_client(op) for op in op_planes]

        # step 01 距离
        C01 = 1.0/1e3
        nearest_distance = np.array([min([self.get_dis(op.ID, p.ID) for p in self.my_planes]) for op in op_planes])
        threat = -nearest_distance*C01

        # step 02 是否已经被攻击
        C0211 = 999999
        C0212 = 30
        for i, op in enumerate(op_planes): 
            if op.under_double_attack > 0 and op.is_drone:
                threat[i] = threat[i] - C0211
            if op.is_vip:
                if op.under_double_attack == 1:
                    threat[i] = threat[i] - C0212  # can fire second roll
                if op.under_double_attack >= 2:
                    threat[i] = threat[i] - C0211  # not permitted to fired third roll
        # step 02 是否已经被攻击
        C022 = 999999
        for i, op in enumerate(op_planes): 
            if op.under_solo_attack:
                threat[i] = threat[i] - C022

        # step 03 是否 vip
        C03 = 50
        for i, op in enumerate(op_planes): 
            if op.is_vip:
                threat[i] = threat[i] + C03        
        # step 04 是否有交叉火力覆盖
        C04 = 20
        for i, op in enumerate(op_planes): 
            my_armed_planes = [p for p in self.my_planes if p.LeftWeapon >= 0]
            sorted_my_planes = sorted(my_armed_planes, key=lambda p:self.get_dis(op.ID, p.ID))
            # 从近到远排序
            distance = np.array([self.get_dis(op.ID, p.ID) for p in sorted_my_planes])
            for j in range(len(distance)-1): # j = 0/1, 1/2
                if np.abs(distance[j+1] - distance[j]) < 5e3:
                    # vec_p2op_01 = op.pos2d - sorted_my_planes[j].pos2d
                    # vec_p2op_02 = op.pos2d - sorted_my_planes[j+1].pos2d
                    # dir_01 = dir2rad(vec_p2op_01)
                    # dir_02 = dir2rad(vec_p2op_02)

                    # dir_02 = reg_rad_at(dir_02, ref=dir_01)
                    # delta_rad = np.abs(dir_02-dir_01) # 弧度制的攻击角度差
                    # delta_deg = delta_rad * 180/np.pi
                    delta_deg = self.get_angle_deg(p1=sorted_my_planes[j],p2=sorted_my_planes[j+1],p_center=op)
                    if delta_deg >=45:
                        threat[i] = threat[i] + C04
                        break
    
        # step 05 敌方有没有弹？什么时候考虑敌方是否有弹？
        # 忽略无弹目标：已经占有优势，或者优势均衡
        # 打击有弹目标：战损正好比对方多一个
        C05 = 999999
        for i, op in enumerate(op_planes): 
            if op.OpLeftWeapon <= 0:
                threat[i] = threat[i] - C05


        for i, op in enumerate(op_planes): 
            op.threat = threat[i]

        sorted_op = sorted(op_planes, key=lambda op:-op.threat)

        # delete target not suitable to attack!
        filtered_sorted_op = list(filter(lambda op: op.threat > -99999, sorted_op))

        # threat_rank = np.argsort(-threat)
        # sorted_op = [op_planes[arg_idx] for arg_idx in threat_rank]
        # ## print亮黄('威胁对方目标排序和威胁:', [(op.Name, op.threat) for op in filtered_sorted_op])

        return filtered_sorted_op


    # 调整导弹的用量
    def process_missile_usage(self):
        self.ms_check_unable_launch() # calculate p.on_hold_ms
        for p in self.my_planes:
            if '有人机' in p.Name:
                p.n_preserve_ms = 1
                p.n_avail_ms = p.LeftWeapon - p.n_preserve_ms - p.on_hold_ms
            if '无人机1' in p.Name:
                p.n_preserve_ms = 0
                p.n_avail_ms = p.LeftWeapon - p.n_preserve_ms - p.on_hold_ms
            if '无人机2' in p.Name:
                p.n_preserve_ms = 0
                p.n_avail_ms = p.LeftWeapon - p.n_preserve_ms - p.on_hold_ms
            if '无人机3' in p.Name:
                p.n_preserve_ms = 0
                p.n_avail_ms = p.LeftWeapon - p.n_preserve_ms - p.on_hold_ms
            if '无人机4' in p.Name:
                p.n_preserve_ms = 0
                p.n_avail_ms = p.LeftWeapon - p.n_preserve_ms - p.on_hold_ms

    @staticmethod
    def get_points_angle_deg(pt1, pt_center, pt2):
        vec_p2op_01 = pt_center - pt1
        vec_p2op_02 = pt_center - pt2
        dir_01 = dir2rad(vec_p2op_01)
        dir_02 = dir2rad(vec_p2op_02)

        dir_02 = reg_rad_at(dir_02, ref=dir_01)
        delta_rad = np.abs(dir_02-dir_01) # 弧度制的攻击角度差
        delta_deg = delta_rad * 180/np.pi
        return delta_deg

    @staticmethod
    def get_angle_deg(p1, p_center, p2):
        vec_p2op_01 = p_center.pos2d - p1.pos2d
        vec_p2op_02 = p_center.pos2d - p2.pos2d
        dir_01 = dir2rad(vec_p2op_01)
        dir_02 = dir2rad(vec_p2op_02)
        dir_02 = reg_rad_at(dir_02, ref=dir_01)
        delta_rad = np.abs(dir_02-dir_01) # 弧度制的攻击角度差
        delta_deg = delta_rad * 180/np.pi
        return delta_deg




    def process_client_one_by_one(self, sorted_op_list):
        for op in sorted_op_list:
            # 双重打击策略
            self.assign_primary_and_secondary(op)
        # 独立判断可以进行单机进攻的条件
        self.assign_solo_service()
        self.last_stand_service()


    # 导弹的运营
    def missile_policy(self):
        # 简单的局部持久变量初始化
        if self.coop_list is None: 
            self.__mspolicy_init__()

        # 处理我方导弹用途，从LeftWeapon出发
        self.process_missile_usage()
        self.overall_cross_fire_angle_analyse()
        sorted_op_list = self.determine_op_rank(self.op_planes)
        self.process_client_one_by_one(sorted_op_list)
        self.execute_ms_coop()
        self.ms_launch()

        self.清理无效的coop和solo()
