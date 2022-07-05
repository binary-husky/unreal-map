from scipy.optimize import linear_sum_assignment
import numpy as np
import copy
import math
import random

class RuleFoundation():
    def __init__(self, n_agent, n_thread, space, mcv=None, team=None):
        self.n_thread = n_thread
        self.n_agent = n_agent
        self.handler = [None for _  in range(self.n_thread)]
        assert n_agent == 10

    def interact_with_env(self, team_intel):
        info = team_intel['Latest-Team-Info']
        done = team_intel['Env-Suffered-Reset']
        step_cnt = team_intel['Current-Obs-Step']
        action_list = []
        for thread in range(self.n_thread):
            act_dict = {'detector_act':None, 'fighter_act':None}
            if done[thread]:
                self.handler[thread] = RuleAgent()  
                self.handler[thread].set_map_info(1000, 1000, 0, 10)
            act_dict['detector_act'], act_dict['fighter_act'] = self.handler[thread].get_action(obs_dict=info[thread], step_cnt=step_cnt[thread])
            action_list.append(act_dict)
            pass
        # $n_thread.${}
        return action_list, None




class RuleAgent():
    def __init__(self):  # 初始化接口
        self.obs_ind = 'raw'  # 状态信息形式
        self.tar = 0
        self.N = 0
        self.angle=0
        self.color_flag=True
        self.formation_flag=4
        self.star_back=True
        self.missile_long = [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]
        self.missile_short = [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]
        # missile_long[i][0]记录的是敌方第i+1个单元身上的远程炮弹数量
        # missile_long[i][k]记录的是敌方第i+1个单元身上的第k个远程炮的计时器
        self.beyond_flag = False
        self.leader_id = 4
        self.tar_pos = np.full((4,2,2), 0)
        # red
        self.tar_pos[0][0][0] = 36
        self.tar_pos[0][1][0] = 400
        self.tar_pos[1][0][0] = 100
        self.tar_pos[1][1][0] = 400
        self.tar_pos[2][0][0]= 700
        self.tar_pos[2][1][0]= 500
        self.tar_pos[3][0][0] = 500
        self.tar_pos[3][1][0] = 700
        # blue
        self.tar_pos[0][0][1] = 964
        self.tar_pos[0][1][1] = 400
        self.tar_pos[1][0][1] = 900
        self.tar_pos[1][1][1] = 400
        self.tar_pos[2][0][1]= 300
        self.tar_pos[2][1][1]= 500
        self.tar_pos[3][0][1] = 500
        self.tar_pos[3][1][1] = 700
        # type_data(攻击距离，发起攻击时要给敌方的索引加几，炮弹类型在self_info[j,？]中的索引)
        self.long_data = (120, 1, 1)
        self.short_data = (50, 11, 2)

    def init_param(self):
        self.tar = 0
        self.N = 0
        self.angle=0
        self.color_flag=True
        self.formation_flag=4
        self.star_back=True
        self.missile_long = [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]
        self.missile_short = [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]
        # missile_long[i][0]记录的是敌方第i+1个单元身上的远程炮弹数量
        # missile_long[i][k]记录的是敌方第i+1个单元身上的第k个远程炮的计时器
        self.beyond_flag = False
        self.leader_id = 4

    def dist(self, obs_dict, i, j):
        adv_obs = obs_dict['fighter'][0]['adv_obs']
        # 计算距离
        x = adv_obs[i][2 * j]
        y = adv_obs[i][2 * j + 1]
        distance = x ** 2 + y ** 2
        distance = math.sqrt(distance)
        return distance

    def set_map_info(self, size_x, size_y, detector_num, fighter_num):  # 读取地图信息
        self.size_x = size_x
        self.size_y = size_y
        self.detector_num = detector_num
        self.fighter_num = fighter_num     # 根据需要自行选择函数实现形式
        self.leader_id = 4

    def _bipartite_min_dists(self, dists):
        ri, ci = linear_sum_assignment(dists)
        return ri, ci

    def tar_judge(self,adv_obs):
        tar_exist = False
        for i in range(self.fighter_num):
            for j in range(self.fighter_num):
                if adv_obs[i][j*2]!=-1000 and adv_obs[i][j*2+1]!=-1000:
                    tar_exist = True
                    break
                    break
        return tar_exist

    def sum_alive(self,alive_status):
        alive_num = 0
        for i in range(self.fighter_num):
            if alive_status[i]:
                alive_num+=1
        return alive_num

    def tar_assign(self,alive_status,adv_obs):
        fighter_action = np.full((self.fighter_num,4),0)
        for i in range(self.fighter_num):
            # 判断该攻击单元是否存活
            if not alive_status[i]:
                continue

            min_dis = 1000 ** 2 + 1000 ** 2
            for j in range(self.fighter_num):  # 记录离我方单元最近的敌方单位id及pos
                x = adv_obs[i][2 * j]
                y = adv_obs[i][2 * j + 1]
                dis = x ** 2 + y ** 2
                if dis < min_dis:
                    min_dis = dis
                    min_id = j
            theta_start = np.arctan2(adv_obs[i][2*min_id+1], adv_obs[i][2*min_id])
            if theta_start < 0:
                theta_start += 2 * np.pi
            course = (int)((theta_start / (2 * np.pi)) * 360)
            fighter_action[i][0] = course
        return fighter_action

    def formation(self,alive_status,self_pos,self_info,step_cnt,formation_flag):     # 编队
        fighter_action = np.full((self.fighter_num, 4), 0)
        if self.color == 'red':
            if step_cnt % 8 == 0 or step_cnt % 9 == 0:
                for i in range(self.fighter_num):
                    fighter_action[i][0] = (self_info[i][0] + 120) % 360
                return fighter_action
        else:
            if step_cnt % 8 == 0 or step_cnt % 9 == 0:
                for i in range(self.fighter_num):
                    if self_info[i][0]-120<0:
                        fighter_action[i][0] = self_info[i][0] -120 + 360
                    else:
                        fighter_action[i][0] = self_info[i][0] - 120
                return fighter_action
        # 挑选领航者
        if not alive_status[self.leader_id]:  # 领航者死亡
            for i in range(self.fighter_num):
                # 挑选存活单元作为领航者
                if alive_status[i]:
                    self.leader_id = i
                    break


        # 设置默认航向
        if self.color == 'red':
            default_course = 0
        else:
            default_course = 180
        start_offset = 100  # 半圆大小3

        # 确定领航者的航迹
        for y in range(self.fighter_num):
            if not alive_status[y]:
                continue
            if y == self.leader_id:
                if self.star_back:
                    if self.color == 'red':
                        if self_pos[self.leader_id][0] > 50 :
                            fighter_action[self.leader_id][0] = default_course+180
                        else :
                            fighter_action[self.leader_id][0] = default_course
                            self.star_back=False
                    else :
                        if self_pos[self.leader_id][0] < 950:
                            fighter_action[self.leader_id][0] = default_course - 180
                        else:
                            self.star_back = False
                            fighter_action[self.leader_id][0] = default_course
                else :
                    if self.color=='red' :
                        # 领航者位置到达目标位置
                        if self_pos[self.leader_id][0] == self.tar_pos[self.tar][0][0] and self_pos[self.leader_id][1] == self.tar_pos[self.tar][1][0]:
                            self.tar = self.tar + 1
                        else:
                            theta_leader = np.arctan2(self.tar_pos[self.tar][1][0] - self_pos[self.leader_id][1],self.tar_pos[self.tar][0][0] - self_pos[self.leader_id][0])
                            if theta_leader < 0:
                                theta_leader += 2 * np.pi
                            course = (theta_leader / (2 * np.pi)) * 360
                            if 90 < course < 180 or 270 < course < 360:
                                course = math.floor(course)
                            else:
                                course = math.ceil(course)
                            fighter_action[self.leader_id][0] = course
                        if self.tar == 4:
                            self.tar = 0
                    else :
                        if self_pos[self.leader_id][0] == self.tar_pos[self.tar][0][1] and self_pos[self.leader_id][1] == self.tar_pos[self.tar][1][1]:
                            self.tar = self.tar + 1
                        else:
                            theta_leader = np.arctan2(self.tar_pos[self.tar][1][1] - self_pos[self.leader_id][1],self.tar_pos[self.tar][0][1] - self_pos[self.leader_id][0])
                            if theta_leader < 0:
                                theta_leader += 2 * np.pi
                            course = (theta_leader / (2 * np.pi)) * 360
                            if 90 < course < 180 or 270 < course < 360:
                                course = math.floor(course)
                            else:
                                course = math.ceil(course)

                            fighter_action[self.leader_id][0] = course
                        if self.tar == 4 :
                            self.tar = 0
                    #print(course)
            # 确定跟随者的航迹
            else:
                if formation_flag == 1:  ##圆形编队
                        fighter_live_num_list = []
                        for fighter_live_num in range(self.fighter_num):
                            if alive_status[fighter_live_num]:
                                fighter_live_num_list.append(fighter_live_num)

                        angle = (int)(360 / (len(fighter_live_num_list) - 1))

                        expected_poses_patrol = []
                        leader_position_patrol = np.array([self_pos[self.leader_id][0], self_pos[self.leader_id][1]])   # 领航者的位置
                        for i in range(len(fighter_live_num_list)):
                            if fighter_live_num_list[i] != self.leader_id:
                                if fighter_live_num_list[i] > self.leader_id:
                                    expected_poses_patrol.append(np.array([leader_position_patrol + start_offset * np.array([np.cos(angle * (i - 1) * np.pi / 180),np.sin(angle * (i - 1) * np.pi / 180)])]))
                                else:
                                    expected_poses_patrol.append([leader_position_patrol + start_offset * np.array([np.cos(angle * i * np.pi / 180), np.sin(angle * i * np.pi / 180)])])
                        dists_patrol = np.array([[np.linalg.norm(np.array([self_pos[i][0], self_pos[i][1]]) - pos) for pos in expected_poses_patrol] for i in range(len(fighter_live_num_list)) if fighter_live_num_list[i] != self.leader_id])
                        ri, ci = self._bipartite_min_dists(dists_patrol)

                        for i in range(len(fighter_live_num_list)):
                            if y == fighter_live_num_list[i]:
                                if y > self.leader_id:
                                    expected_poses_for_it = expected_poses_patrol[ci[i - 1]]
                                else:
                                    expected_poses_for_it = expected_poses_patrol[ci[i]]
                                break
                        relative_value_patrol = expected_poses_for_it - np.array([self_pos[y][0], self_pos[y][1]])
                        theta_patrol = np.arctan2(relative_value_patrol[0][1], relative_value_patrol[0][0])
                        if theta_patrol < 0:
                            theta_patrol += 2 * np.pi
                        course = (int)((theta_patrol / (2 * np.pi)) * 360)
                        fighter_action[y][0] = course

                elif formation_flag == 2:  ##半圆编队
                    y_width = 60.0
                    y_offset = 120
                    if self.color == 'red':
                        x_offset = -120.0
                    else:
                        x_offset = 120.0

                    ##确定期望位置 这个很关键
                    expected_poses = []
                    leader_position = np.array([self_pos[self.leader_id][0],self_pos[self.leader_id][1]])

                    for i in range(self.fighter_num - 1):
                        if i == 0:
                            temp_position = [leader_position + np.array([0.0, y_width])]
                            expected_poses.append(temp_position)
                        elif i == 1:
                            temp_position = [leader_position + np.array([0.0, 2 * y_width])]
                            expected_poses.append(temp_position)
                        elif i == 2:
                            temp_position = [leader_position + np.array([0.0, -y_width])]
                            expected_poses.append(temp_position)
                        elif i == 3:
                            temp_position = [leader_position + np.array([0.0, -2 * y_width])]
                            expected_poses.append(temp_position)
                        elif i == 4:
                            temp_position = [leader_position + np.array(
                                [x_offset * np.cos(60 * np.pi / 180), -y_offset * np.sin(60 * np.pi / 180)])]
                            expected_poses.append(temp_position)
                        elif i == 5:
                            temp_position = [leader_position + np.array(
                                [x_offset * np.cos(30 * np.pi / 180), -y_offset * np.sin(30 * np.pi / 180)])]
                            expected_poses.append(temp_position)
                        elif i == 6:
                            temp_position = [leader_position + np.array(
                                [x_offset * np.cos(0 * np.pi / 180), y_offset * np.sin(0 * np.pi / 180)])]
                            expected_poses.append(temp_position)
                        elif i == 7:
                            temp_position = [leader_position + np.array(
                                [x_offset * np.cos(30 * np.pi / 180), y_offset * np.sin(30 * np.pi / 180)])]
                            expected_poses.append(temp_position)
                        elif i == 8:
                            temp_position = [leader_position + np.array(
                                [x_offset * np.cos(60 * np.pi / 180), y_offset * np.sin(60 * np.pi / 180)])]
                            expected_poses.append(temp_position)

                    dists = np.array([[np.linalg.norm(np.array([self_pos[i][0],self_pos[i][1]]) - pos) for pos in expected_poses] for i in range(self.fighter_num) if i != self.leader_id])
                    ri, ci = self._bipartite_min_dists(dists)

                    if y <= self.leader_id:
                        ci_v1 = y
                    else:
                        ci_v1 = y - 1

                    relative_value = expected_poses[ci[ci_v1]] - np.array([self_pos[y][0],self_pos[y][1]])
                    theta_start = np.arctan2(relative_value[0][1], relative_value[0][0])
                    if theta_start < 0:
                        theta_start += 2 * np.pi
                    course = (int)((theta_start / (2 * np.pi)) * 360)
                    fighter_action[y][0] = course

                elif formation_flag == 3:  ## 三角编队

                    y_width = 45.0
                    if self.color == 'red':
                        x_width = 25.0
                    else:
                        x_width = -25.0

                    ##确定期望位置 这个很关键
                    expected_poses = []
                    leader_position = np.array([self_pos[self.leader_id][0],self_pos[self.leader_id][1]])

                    for i in range(self.fighter_num - 1):
                        if i == 0:
                            temp_position = [leader_position + np.array([-x_width, 0])]
                            expected_poses.append(temp_position)
                        elif i == 1:
                            temp_position = [leader_position + np.array([-x_width, -y_width])]
                            expected_poses.append(temp_position)
                        elif i == 2:
                            temp_position = [leader_position + np.array([-x_width, -2*y_width])]
                            expected_poses.append(temp_position)
                        elif i == 3:
                            temp_position = [leader_position + np.array([-x_width, 2 * y_width])]
                            expected_poses.append(temp_position)
                        elif i == 4:
                            temp_position = [leader_position + np.array([-x_width, 3*y_width ])]
                            expected_poses.append(temp_position)
                        elif i == 5:
                            temp_position = [leader_position + np.array([0 , y_width * 3 / 2])]
                            expected_poses.append(temp_position)
                        elif i == 6:
                            temp_position = [leader_position + np.array([0, -y_width *3/ 2])]
                            expected_poses.append(temp_position)
                        elif i == 7:
                            temp_position = [leader_position + np.array([x_width, -y_width * 0.5])]
                            expected_poses.append(temp_position)
                        elif i == 8:
                            temp_position = [leader_position + np.array([x_width, -y_width * 0.5])]
                            expected_poses.append(temp_position)

                    dists = np.array([[np.linalg.norm(np.array([self_pos[i][0],self_pos[i][1]]) - pos) for pos in expected_poses] for i in range(self.fighter_num) if i != self.leader_id])
                    ri, ci = self._bipartite_min_dists(dists)

                    if y <= self.leader_id:
                        ci_v1 = y
                    else:
                        ci_v1 = y - 1

                    relative_value = expected_poses[ci[ci_v1]] - np.array([self_pos[y][0],self_pos[y][1]])
                    theta_start = np.arctan2(relative_value[0][1], relative_value[0][0])
                    if theta_start < 0:
                        theta_start += 2 * np.pi

                    course = (int)((theta_start / (2 * np.pi)) * 360)
                    fighter_action[y][0] = course

                elif formation_flag == 4:  ## 网口

                    y_width = 45.0
                    if self.color == 'red':
                        x_width = 45.0
                    else:
                        x_width = -45.0

                    ##确定期望位置 这个很关键
                    expected_poses = []
                    leader_position = np.array([self_pos[self.leader_id][0],self_pos[self.leader_id][1]])

                    for i in range(self.fighter_num - 1):
                        if i == 0:
                            temp_position = [leader_position + np.array([0, -y_width])]
                            expected_poses.append(temp_position)
                        elif i == 1:
                            temp_position = [leader_position + np.array([0.5 * x_width, -2 * y_width])]
                            expected_poses.append(temp_position)
                        elif i == 2:
                            temp_position = [leader_position + np.array([1 * x_width, -3 * y_width])]
                            expected_poses.append(temp_position)
                        elif i == 3:
                            temp_position = [leader_position + np.array([2 * x_width, -4 * y_width])]
                            expected_poses.append(temp_position)
                        elif i == 4:
                            temp_position = [leader_position + np.array([0, y_width])]
                            expected_poses.append(temp_position)
                        elif i == 5:
                            temp_position = [leader_position + np.array([0.5*x_width, 2 * y_width])]
                            expected_poses.append(temp_position)
                        elif i == 6:
                            temp_position = [leader_position + np.array([1 * x_width, 3 * y_width])]
                            expected_poses.append(temp_position)
                        elif i == 7:
                            temp_position = [leader_position + np.array([2 * x_width, 4 * y_width])]
                            expected_poses.append(temp_position)
                        elif i == 8:
                            temp_position = [leader_position + np.array([3 * x_width, 5 * y_width])]
                            expected_poses.append(temp_position)

                    dists = np.array([[np.linalg.norm(np.array([self_pos[i][0],self_pos[i][1]]) - pos) for pos in expected_poses] for i in range(self.fighter_num) if i != self.leader_id])
                    ri, ci = self._bipartite_min_dists(dists)

                    if y <= self.leader_id:
                        ci_v1 = y
                    else:
                        ci_v1 = y - 1

                    relative_value = expected_poses[ci[ci_v1]] - np.array([self_pos[y][0],self_pos[y][1]])
                    theta_start = np.arctan2(relative_value[0][1], relative_value[0][0])
                    if theta_start < 0:
                        theta_start += 2 * np.pi
                    course = (int)((theta_start / (2 * np.pi)) * 360)
                    fighter_action[y][0] = course
        return fighter_action



    def get_action(self, obs_dict, step_cnt):
        # 获取观测数据
        obs_raw_dict = obs_dict
        obs_dict = {}
        detector_obs_list = []
        fighter_obs_list = []
        detector_data_obs_list = obs_raw_dict['detector_obs_list']
        fighter_data_obs_list = obs_raw_dict['fighter_obs_list']
        joint_data_obs_dict = obs_raw_dict['joint_obs_dict']

        adv_obs = np.full((self.fighter_num, self.fighter_num * 2), -1000, dtype=np.int32)
        fri_obs = np.full((self.fighter_num, self.fighter_num * 2), -1000, dtype=np.int32)
        self_info = np.full((self.fighter_num, 3), -1, dtype=np.int32)
        self_pos = np.full((self.fighter_num, 2), -1000, dtype=np.int32)
        alive_status = np.full(self.fighter_num, True)
        enemy_alive_status = np.full(self.fighter_num,True)
        attack_list = np.full((self.fighter_num, self.fighter_num), 0, dtype=np.int32)

        for xx in range(self.fighter_num):
            if not fighter_data_obs_list[xx]['alive']:
                continue
            for x in range(self.fighter_num):  # 攻击单元探测雷达信息
                # if not alive, skip
                if not fighter_data_obs_list[x]['alive']:
                    continue
                # self detection target. target: id
                for y in range(len(fighter_data_obs_list[x]['r_visible_list'])):
                    adv_obs[xx, 2 * (fighter_data_obs_list[x]['r_visible_list'][y]['id'] - 1)] = \
                    fighter_data_obs_list[x]['r_visible_list'][y]['pos_x'] - fighter_data_obs_list[xx]['pos_x']
                    adv_obs[xx, 2 * fighter_data_obs_list[x]['r_visible_list'][y]['id'] - 1] = \
                    fighter_data_obs_list[x]['r_visible_list'][y]['pos_y'] - fighter_data_obs_list[xx]['pos_y']
        for xx in range(self.fighter_num):  # 攻击单元被动探测雷达信息
            if not fighter_data_obs_list[xx]['alive']:
                continue
            for j in range(len(joint_data_obs_dict['passive_detection_enemy_list'])):
                adv_obs[xx, 2 * (joint_data_obs_dict['passive_detection_enemy_list'][j]['id'] - 1)] = \
                joint_data_obs_dict['passive_detection_enemy_list'][j]['pos_x'] - fighter_data_obs_list[xx]['pos_x']
                adv_obs[xx, 2 * joint_data_obs_dict['passive_detection_enemy_list'][j]['id'] - 1] = \
                joint_data_obs_dict['passive_detection_enemy_list'][j]['pos_y'] - fighter_data_obs_list[xx]['pos_y']

        for xx in range(self.fighter_num):  # 我方各攻击单元相对位置
            if not fighter_data_obs_list[xx]['alive']:
                continue
            for x in range(self.fighter_num):
                if not fighter_data_obs_list[x]['alive']:
                    continue
                fri_obs[xx, 2 * x] = fighter_data_obs_list[x]['pos_x'] - fighter_data_obs_list[xx]['pos_x']
                fri_obs[xx, 2 * x + 1] = fighter_data_obs_list[x]['pos_y'] - fighter_data_obs_list[xx]['pos_y']

        for x in range(self.fighter_num):
            if fighter_data_obs_list[x]['alive']:
                self_info[x, 0] = fighter_data_obs_list[x]['course']
                self_info[x, 1] = fighter_data_obs_list[x]['l_missile_left']
                self_info[x, 2] = fighter_data_obs_list[x]['s_missile_left']
                self_pos[x, 0] = fighter_data_obs_list[x]['pos_x']
                self_pos[x, 1] = fighter_data_obs_list[x]['pos_y']
        for i in range(self.fighter_num):
            for x in range(len(fighter_data_obs_list[i]['r_visible_list'])):
                attack_list[i][x] = fighter_data_obs_list[i]['r_visible_list'][x]["id"]

        for i in range(self.fighter_num):
            enemy_alive_status[i] = joint_data_obs_dict['alive_status_enemy_list'][i]


        for x in range(self.fighter_num):
            if not fighter_data_obs_list[x]['alive']:
                alive_status[x] = False
        fighter_obs_list.append({'adv_obs': copy.deepcopy(adv_obs), 'fri_obs': copy.deepcopy(fri_obs),
                                 'self_info': copy.deepcopy(self_info), 'alive': copy.deepcopy(alive_status),
                                 'self_pos': copy.deepcopy(self_pos), 'attack_list': copy.deepcopy(attack_list),
                                 'enemy_alive':copy.deepcopy(enemy_alive_status)
                                 }
                                )
        obs_dict['fighter'] = fighter_obs_list



        detector_action = np.full((self.detector_num,2),0)
        adv_obs=obs_dict['fighter'][0]['adv_obs']
        fri_obs=obs_dict['fighter'][0]['fri_obs']
        self_info=obs_dict['fighter'][0]['self_info']
        self_pos =obs_dict['fighter'][0]['self_pos']
        alive_status=obs_dict['fighter'][0]['alive']
        enemy_alive_status=obs_dict['fighter'][0]['enemy_alive']
        attack_list = obs_dict['fighter'][0]['attack_list']
        enemy_alive_status = obs_dict['fighter'][0]['enemy_alive']

        if step_cnt == 0 or step_cnt == 1:
            self.init_param()


        # 红蓝方判别
        if self.color_flag:
            self.color_flag = False
            if self_pos[self.leader_id][0]<500:
                self.color = 'red'
            else:
                self.color = 'blue'

        ####################################################################################
        #############################航向规则############################
        ####################################################################################
        # formation_flag  1：圆形防守编队
        #                 2：半圆半防守半进攻编队
        #                 3：三角进攻编队
        #                 4：网口进攻编队
        #############################全局态势评估#########################
        tar_exist = self.tar_judge(adv_obs)   				# 是否观测到敌方目标
        enemy_alive_num = np.sum(enemy_alive_status != 0)
        self_alive_num = np.sum(alive_status != 0)			# 敌我双方力量对比
        if abs(enemy_alive_num - self_alive_num) <= 2:
            situation_flag = 1  # 势均力敌
        elif enemy_alive_num > self_alive_num:
            situation_flag = 2  # 敌强我弱
        else:
            situation_flag = 3  # 敌弱我强

        if tar_exist:
            fighter_action=self.tar_assign(alive_status, adv_obs)    # 目标存在   目标分配策略
        else:
            # 目标不存在
            if situation_flag == 1:  # 势均力敌
                if self.color == 'red':
                    if self.beyond_flag:
                        formation_flag = 1
                        fighter_action = self.formation(alive_status, self_pos, self_info, step_cnt,formation_flag)  # 防守阵型
                    else:
                        formation_flag = 4
                        fighter_action = self.formation(alive_status, self_pos, self_info, step_cnt,formation_flag)  # 进攻阵型
                        if self_pos[self.leader_id][0] > 455:
                            self.beyond_flag = True

                else:
                    if self.beyond_flag:
                        formation_flag = 1
                        fighter_action = self.formation(alive_status, self_pos, self_info, step_cnt,formation_flag)  # 防守阵型
                    else:
                        formation_flag = 4
                        fighter_action = self.formation(alive_status, self_pos, self_info, step_cnt,formation_flag)  # 进攻阵型
                        if self_pos[self.leader_id][0] < 545:
                            self.beyond_flag = True
            elif situation_flag == 2: # 敌强我弱
                formation_flag = 1
                fighter_action=self.formation(alive_status,self_pos,self_info,step_cnt,formation_flag)	# 圆形防守阵型
            else:                     # 敌弱我强
                formation_flag = 1
                fighter_action=self.formation(alive_status,self_pos,self_info,step_cnt,formation_flag)	# 三角进攻阵型


        ####################################################################################
                #############################攻击规则############################
        ####################################################################################
        alive_enemy_num = np.sum(enemy_alive_status != 0)        # 剩余敌方单元数量
        long_sum = 0
        short_sum = 0
        for i in range(self.fighter_num):
            if alive_status[i]:
                long_sum += self_info[i, 1]
                short_sum += self_info[i, 2]
        for i in range(self.fighter_num):
            if enemy_alive_status[i]:
                long_sum += self.missile_long[i][0]
                short_sum += self.missile_short[i][0]
        long_limit = round(long_sum / alive_enemy_num)    # 远程弹上限
        short_limit = round(short_sum / alive_enemy_num)    # 近程弹上限
        if 0 < (long_sum / alive_enemy_num) < 1:  # 防止有炮弹不发
            long_limit = 1
        if 0 < (short_sum / alive_enemy_num) < 1:
            short_limit = 1
        self.attack_total(fighter_action, attack_list, self_info, obs_dict,self.missile_long,
                          self.missile_short, self.long_data, self.short_data, long_limit, short_limit)   # 总的攻击函数
        ####################################################################################
        #############################变频、干扰规则############################
        ####################################################################################
        radar = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        random.shuffle(radar)
        for i in range(self.fighter_num):
            fighter_action[i][1] = radar[i]  # 探测雷达
            fighter_action[i][2] = 11  # 干扰雷达
        return detector_action, fighter_action




    def attack_total(self, fighter_action, attack_list, self_info, obs_dict,
                     missile_long, missile_short, long_data, short_data, long_limit, short_limit):
        max_limit = max(long_limit, short_limit)
        max_limit = int(max_limit)
        for limit in range(max_limit):  # 优先发动近程炮，再发远程炮
            if limit < short_limit:   # 判断当前近程炮有没有超过近程炮上限，若未超过则执行近程炮的下层函数
                self.attack(fighter_action, attack_list, self_info, obs_dict, missile_short, limit, short_data)
            if limit < long_limit:     # 判断当前远程炮有没有超过远程炮上限，若未超过则执行远程炮的下层函数
                self.attack(fighter_action, attack_list, self_info, obs_dict, missile_long, limit, long_data)
        self.missile_tracking_update()



    def attack(self, fighter_action, attack_list, self_info, obs_dict, missile_type, tracking_num, type_data):
        # 攻击函数的下层函数，主要的参数为最后三个，分别为发出的炮弹类型，当前敌方单元被几个炮弹追踪，对应炮弹类型的数据
        strike_matrix = np.full((self.fighter_num, self.fighter_num), 0, dtype=np.int32)  # 建立一个空攻击列表

        for i in range(self.fighter_num):  # i代表敌军编号
            for j in range(self.fighter_num):  # j代表我方编号
                if missile_type[i][0] == tracking_num and fighter_action[j][3] == 0:  # 敌军i当前被几个导弹锁定且j空闲
                    for x in range(len(attack_list[j])):
                        if attack_list[j][x] == i + 1:
                            if self_info[j, type_data[2]] != 0:  # j有近/远程炮
                                if self.dist(obs_dict, j, i) < type_data[0]:  # i进入j近/远程炮范围
                                    strike_matrix[i][j] = 1  # 标记j可以攻击到i

        while np.sum(strike_matrix) != 0:  # 攻击列表不为空则进行攻击判断的循环
            enemy_number = 11  # 随意定义一个较大的初始值
            row_sum = np.sum(strike_matrix, axis=1)  # 计算攻击列表的行求和
            for i in range(len(row_sum)):
                if 0 < row_sum[i] < enemy_number:  # 挑选求和最小的非零行
                    enemy_num = i
                    enemy_number = row_sum[i]  # 选择被攻击单位

            attack_num = 11  # 随意定义一个较大的初始值
            column_sum = np.sum(strike_matrix, axis=0)  # 计算攻击列表的列求和
            for j in range(len(column_sum)):
                if strike_matrix[enemy_num][j] == 1:  # 找出所有可攻击到被选中的敌方单位的列
                    if column_sum[j] < attack_num:  # 选择求和最小的列
                        attack_num = column_sum[j]
                        ally_num = j  # 选择攻击单位

            fighter_action[ally_num][3] = (enemy_num + type_data[1])  # 发炮
            missile_type[enemy_num][0] += 1                          # 该敌方单元身上的炮弹＋1
            missile_type[enemy_num].append(101)                           # 计近/远程炮时间步
            for p in range(self.fighter_num):
                strike_matrix[enemy_num][p] = 0
                strike_matrix[p][ally_num] = 0  # 更新攻击列表


    def missile_tracking_update(self):
        for i in range(self.fighter_num):
            for j in range(1, len(self.missile_long[i])):
                self.missile_long[i][j] += 1  # 所有记时位＋1
            for j in range(1, len(self.missile_short[i])):
                self.missile_short[i][j] += 1  # 计时位+1
        for i in range(self.fighter_num):
            while 111 in self.missile_long[i]:
                self.missile_long[i].remove(111)  # 超出时延则抹掉
                self.missile_long[i][0] -= 1  # 超出时延则身上的炮弹数-1
            while 105 in self.missile_short[i]:
                self.missile_short[i].remove(105)
                self.missile_short[i][0] -= 1



