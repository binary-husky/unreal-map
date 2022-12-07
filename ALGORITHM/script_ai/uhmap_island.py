from cmath import isinf, pi
from turtle import done
import numpy as np
import math
from MISSION.uhmap.actionset_v3 import strActionToDigits, ActDigitLen
from config import GlobalConfig

class DummyAlgConfig():
    reserve = ""

class DummyAlgorithmBase():
    def __init__(self, n_agent, n_thread, space, mcv=None, team=None):
        self.n_agent = n_agent
        self.n_thread = n_thread
        self.team = team
        self.ScenarioConfig = GlobalConfig.ScenarioConfig
        self.attack_order = {}
        self.team_agent_uid = GlobalConfig.ScenarioConfig.AGENT_ID_EACH_TEAM[team]
        self.demo_type = GlobalConfig.ScenarioConfig.DemoType
        if self.demo_type == 'AirShow' or 'AirAttack':
            self.phase = 1
        if self.demo_type == 'AirAttack':
            self.TargetPosition = []

    def forward(self, inp, state, mask=None):
        raise NotImplementedError

    def to(self, device):
        return self
    
    def interact_with_env(self, State_Recall):
        assert State_Recall['Latest-Obs'] is not None, ('make sure obs is ok')
        
        ENV_PAUSE = State_Recall['ENV-PAUSE']
        ENV_ACTIVE = ~ENV_PAUSE

        assert self.n_thread == len(ENV_ACTIVE), ('the number of thread is wrong?')
        
        n_active_thread = sum(ENV_ACTIVE)

        # assert len(State_Recall['Latest-Obs']) == n_active_thread, ('make sure we have the right batch of obs')

        actions = np.zeros(shape=(self.n_thread, self.n_agent, ActDigitLen))

        # set actions of in-active threads to NaN (will be done again in multi_team.py, this line is not necessary)
        actions[ENV_PAUSE] = np.nan
        
        # swap (self.n_thread, self.n_agent) -> (self.n_agent, self.n_thread) 
        actions = np.swapaxes(actions, 0, 1)
        return actions, {}



class DummyAlgorithmIdle(DummyAlgorithmBase):
    

    '''
        福建省东北角大致方位  (-17500,-19500)
        福建省西南角大致方位  (-22500,-5000)
        注意：0°对应x轴正方向,90°对应y轴正方向
        plane_rotaion = [方位角，俯仰角，翻滚角]
    '''
    def interact_with_env(self, State_Recall):
        assert State_Recall['Latest-Obs'] is not None, ('make sure obs is ok')
        ENV_PAUSE = State_Recall['ENV-PAUSE']
        ENV_ACTIVE = ~ENV_PAUSE

        assert self.n_thread == len(ENV_ACTIVE), ('the number of thread is wrong?')

        n_active_thread = sum(ENV_ACTIVE)
        # assert len(State_Recall['Latest-Obs']) == n_active_thread, ('make sure we have the right batch of obs')

        actions = np.zeros(shape=(self.n_thread, self.n_agent, ActDigitLen))
        
        for thread in range(self.n_thread):
            if ENV_PAUSE[thread]: 
                # 如果,该线程停止，不做任何处理
                continue

            # 此处代码仅做demo用
            act_dic = ActionDictionary
            if self.demo_type == "AirShow":
                '''
                飞行场景设计：
                初始位置: x/y按照索引设定, 初始高度一致为1000米, 初始俯仰角为0, 初始滚转角为0, 初始偏航角按照索引设定
                1.上升到统一高度5000m, 欧拉角全部设为0
                2.方位角先变到-90°, 后变到0°, 在此期间高度上升至10000m
                3.方位角变到180°
                4.方位角先变到-90°, 后变到180°, 在此期间高度下降至5000m
                5.方位角变到0°
                6.重复2~5动作
                '''
                
                for id in range(self.n_agent):
                    cruise_height = 15000
                    cruise_speed = 600
                    plane_location = State_Recall['Latest-Team-Info'][thread]['dataArr'][id]['agentLocationArr']
                    plane_rotaion = State_Recall['Latest-Team-Info'][thread]['dataArr'][id]['agentRotationArr']
                    
                    if self.phase == 1:
                        if np.abs(0 - plane_rotaion[1]) < 1:
                            actions[thread, id] = act_dic.select_act('PlaneAgent', 2)
                            print("Change Height!")
                        else:
                            actions[thread, id] = act_dic.select_act('PlaneAgent', 6)
                            print("Change Direction!")
                        if np.abs(5000 - plane_location[2]) < 0.1 and np.abs(0 - plane_rotaion[0]) < 1:
                            self.phase += 1
                            print("Stage1 Done")

                    elif self.phase == 2:
                        print("Stage2!")
                        if np.abs(0 - plane_rotaion[1]) < 1:
                            actions[thread, id] = act_dic.select_act('PlaneAgent', 3)
                        else:
                            actions[thread, id] = act_dic.select_act('PlaneAgent', 12)
                        if np.abs(-90 - plane_rotaion[0]) < 1:
                            self.phase += 0.5
                    elif self.phase == 2.5:
                        actions[thread, id] = act_dic.select_act('PlaneAgent', 6)
                        if np.abs(10000 - plane_location[2]) < 0.1 and np.abs(0 - plane_rotaion[0]) < 1:
                            self.phase += 0.5
                            print("Stage2 Done")


                    elif self.phase == 3:
                        actions[thread, id] = act_dic.select_act('PlaneAgent', 10)
                        if np.abs(180 - plane_rotaion[0]) < 1:
                            self.phase += 1
                            print("Stage3 Done")
                    
                    elif self.phase == 4:
                        print("Stage4!")
                        if np.abs(0 - plane_rotaion[1]) < 1:
                            actions[thread, id] = act_dic.select_act('PlaneAgent', 2)
                        else:
                            actions[thread, id] = act_dic.select_act('PlaneAgent', 12)
                        if np.abs(-90 - plane_rotaion[0]) < 1:
                            self.phase += 0.5
                    elif self.phase == 4.5:
                        print(self.phase)
                        actions[thread, id] = act_dic.select_act('PlaneAgent', 10)
                        if np.abs(5000 - plane_location[2]) < 10 and np.abs(180 - plane_rotaion[0]) < 1:
                            self.phase += 0.5
                            print("Stage4 Done")
                    
                    elif self.phase == 5:
                        actions[thread, id] = act_dic.select_act('PlaneAgent', 6)
                        if np.abs(0 - plane_rotaion[0]) < 1:
                            self.phase = 2
                            print("Stage5 Done")
                    


            elif self.demo_type == "AirAttack":
                '''
                全流程demo演示:
                1. 飞机一直跟踪至目标方位，并上升到巡航高度(暂定15000m)
                2. 在开始上升后，开始加速动作，一直到巡航速度(暂定600m/s)
                3. 到达巡航高度后，飞机进行巡航，慢慢抵达目标
                4. 接近目标后，降低高度到打击高度(暂定5000m) (预计打击半径暂定为50,000m)
                5. 到达指定高度后，始终将目标方向作为期望航迹方位角方向
                6. 到达目标后，与目标进行交互打击，若目标被摧毁则在原地盘旋
                7. 打击后汇合
  
                '''

                # 获取目标坐标及参数设置
                if self.phase <= 1:
                    for id in range(100,105):
                        self.TargetPosition.append(State_Recall['Latest-Team-Info'][thread]['dataArr'][id]['agentLocationArr'])
                cruise_height = 15000
                cruise_speed = 600
                attack_height = 5000
                ready_radius = 50000
                # 执行飞行器脚本
                for id in range(self.n_agent - 5):
                    if State_Recall['Latest-Team-Info'][thread]['dataArr'][id]['agentAlive']:
                        plane_location = State_Recall['Latest-Team-Info'][thread]['dataArr'][id]['agentLocationArr']
                        plane_rotaion = State_Recall['Latest-Team-Info'][thread]['dataArr'][id]['agentRotationArr']
                        
                        # 分配目标
                        index = math.floor(id / (self.n_agent - 5) * 5)
                        target_location = np.array((self.TargetPosition[index]))

                        delta_location =  target_location - plane_location
                        target_pitch = self.DeltaLocation2Angle(delta_location[0], delta_location[1])
                        
                        # 1.
                        if self.phase == 1:
                            if np.abs(cruise_height - plane_location[2]) < 5:
                                self.phase += 1
                                print("Stage1 Done")
                            elif np.abs(0 - plane_rotaion[1]) < 1:
                                action_num = self.TargetHeight2Action(cruise_height)
                                actions[thread, id] = act_dic.select_act('PlaneAgent', action_num)
                                print("Change Height to 15000m!")
                            else:
                                # 跟踪目标位置
                                delta_location =  target_location - plane_location
                                target_pitch = self.DeltaLocation2Angle(delta_location[0], delta_location[1]) 
                                action_num = self.TargetAngle2Action(target_pitch)
                                actions[thread, id] = act_dic.select_act('PlaneAgent', action_num)
                        
                        # 2.
                        if self.phase == 2:
                            speed = float(State_Recall['Latest-Team-Info'][thread]['dataArr'][id]['rSVD1'])
                            if np.abs(cruise_speed - speed) < 1:
                                self.phase += 1
                                print("Stage2 Done")
                            else:
                                actions[thread, id] = act_dic.select_act('PlaneAgent', 14)

                        # 3. 
                        if self.phase == 3:
                            delta_location =  target_location - plane_location
                            if np.sqrt(np.sum(np.square((delta_location[0], delta_location[1])))) <= ready_radius:
                                self.phase += 1
                                print("Stage3 Done")
                            else:
                                # 跟踪目标位置
                                delta_location =  target_location - plane_location
                                target_pitch = self.DeltaLocation2Angle(delta_location[0], delta_location[1]) 
                                action_num = self.TargetAngle2Action(target_pitch)
                                actions[thread, id] = act_dic.select_act('PlaneAgent', action_num)
                        
                        # 4.
                        if self.phase == 4:
                            if np.abs(attack_height - plane_location[2]) < 5:
                                self.phase += 1
                                print("Stage4 Done")
                            elif np.abs(0 - plane_rotaion[1]) < 1:
                                action_num = self.TargetHeight2Action(attack_height)
                                actions[thread, id] = act_dic.select_act('PlaneAgent', action_num)
                                print("Change Height to 5000m!")
                            else:
                                # 跟踪目标位置
                                delta_location =  target_location - plane_location
                                target_pitch = self.DeltaLocation2Angle(delta_location[0], delta_location[1]) 
                                action_num = self.TargetAngle2Action(target_pitch)
                                actions[thread, id] = act_dic.select_act('PlaneAgent', action_num)
                        
                        # 5.
                        if self.phase == 5:
                            # 跟踪目标位置
                            delta_location =  target_location - plane_location
                            target_pitch = self.DeltaLocation2Angle(delta_location[0], delta_location[1]) 
                            action_num = self.TargetAngle2Action(target_pitch)
                            actions[thread, id] = act_dic.select_act('PlaneAgent', action_num)


                        # 5.接近目标，对目标发射导弹
                        # dist_2D = np.sqrt(np.sum(np.square((delta_location[0], delta_location[1]))))
                        # if dist_2D < 10000:
                        #     actions[thread, id] = strActionToDigits('ActionSet3::LaunchMissile;NONE')

                        # print(target_pitch)
                        # actions[thread, :] = strActionToDigits('ActionSet3::ChangeDirection;{}'.format(target_pitch))
                        # print(State_Recall['Latest-Team-Info'][thread]['dataArr'][0]['agentRotationArr']) 


        # set actions of in-active threads to NaN (will be done again in multi_team.py, this line is not necessary)
        actions[ENV_PAUSE] = np.nan
        # swap (self.n_thread, self.n_agent) -> (self.n_agent, self.n_thread) 
        actions = np.swapaxes(actions, 0, 1)
        return actions, {}

    def DeltaLocation2Angle(self, delta_x, delta_y):
        '''
        将输入的距离差向量转换为方位角度
        此处为角度制
        '''
        # assert len(delta_location) == 2 or 3
        # delta_x = delta_location[0]
        # delta_y = delta_location[1]
        if delta_x == 0 and delta_y != 0:
            theta = 90 if delta_y > 0 else -90
        else:
            abs_theta = np.arctan(np.abs(delta_y) / np.abs(delta_x)) * 180 / pi

            if delta_x > 0 and delta_y >= 0:
                theta = abs_theta
            elif delta_x < 0 and delta_y >= 0:
                theta = 180 - abs_theta
            elif delta_x > 0 and delta_y < 0:
                theta = - abs_theta
            elif delta_x < 0 and delta_y < 0:    
                theta = abs_theta - 180
        return theta
        
    def TargetAngle2Action(self, target_yaw):
        '''
        将输入的期望角度转化为对应的离散动作
        此处为角度制
        '''
        action_yaw_set = np.array([0, 45, 90, 135, 180, -135, -90, -45])
        delta_action_yaw_set = np.abs(action_yaw_set - target_yaw)

        output_num = None
        for i, element in enumerate(delta_action_yaw_set):
            if element <= 22.5 or element >= (180+135+22.5):
                output_num = i
                break
        # if output_num is None:
        #     print('离散方位角动作设置或者程序逻辑有问题！')
        #     print(target_yaw)
        #     print(delta_action_yaw_set)

        assert output_num is not None, '离散方位角动作设置或者程序逻辑有问题！'
        return output_num + 6
    def TargetHeight2Action(self, target_height):
        '''
        将输入的期望高度转化为对应的离散动作
        '''
        action_height_set = np.array([1000, 5000, 10000, 15000, 20000])
        delta_action_height_set = np.abs(action_height_set - target_height)
        height_threshold = np.abs(action_height_set[-1] - action_height_set[-2]) / 2

        output_num = None
        for i, element in enumerate(delta_action_height_set):
            if element <= height_threshold:
                output_num = i
                break

        assert output_num is not None, '离散高度动作设置或者程序逻辑有问题！'
        return output_num + 1



class ActionDictionary():
    '''
        Height Space(5): 20000m, 15000m, 10000m, 5000m, 1000m
        Direction Space(8): 45°, 90°, 135°, 180°, -135°, -90°, -45°, 0°
        Speed Space(2): Positive, Negative
    '''
    # Direction Space(16): 22.5°, 45°, 67.5°, 90°, 112.5°, 135°, 157.5°, 180°, -157.5°, -135°, -112.5°, -90°, -67.5°, -45°, -22.5°, 0°
    # Speed Space(10): 150, 200, 250, 300, 350, 400, 450, 500, 550, 600
    
    dictionary_args = [
    'N/A;N/A',                           # 0
    
    'ChangeHeight;1000',                 # 1
    'ChangeHeight;5000',                 # 2
    'ChangeHeight;10000',                # 3
    'ChangeHeight;15000',                # 4
    'ChangeHeight;20000',                # 5

    'ChangeDirection;0',                 # 6
    'ChangeDirection;45',                # 7
    'ChangeDirection;90',                # 8
    'ChangeDirection;135',               # 9
    'ChangeDirection;180',               # 10
    'ChangeDirection;-135',              # 11
    'ChangeDirection;-90',               # 12
    'ChangeDirection;-45',               # 13

    'ChangeSpeed;Positive',               # 14
    'ChangeSpeed;Negative',               # 15
    
    ]


    @staticmethod
    def select_act(type, a):
        if type =='PlaneAgent':
            args = ActionDictionary.dictionary_args[a]
            return strActionToDigits(f'ActionSet3::{args}')

    @staticmethod     
    def get_avail_act():
        pass




