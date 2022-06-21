import numpy as np
import math
from scipy.optimize import linear_sum_assignment


def defense_combat(self_data, ally_agents_data, enemy_agents_data, key_points, blue_alive, red_alive):
    # 防守方red小车最大速度
    red_car_max_vel = 600
    # 进攻方blue小车最大速度
    blue_car_max_vel = 600
    # 进攻方blue无人机最大速度
    blue_drone_max_vel = 600
    # 进攻方无人机占领夺控点胜利时间
    time_to_win = 2.0
    # 驱离载荷作用范围
    expel_range = 1200
    # 无人车打击距离
    fire_dist = 2000

    all_enemy_agent_pos = []
    for agent_id, dict_value in enemy_agents_data.items():
        all_enemy_agent_pos.append([dict_value['X'], dict_value['Y'], dict_value['Z']])
    """
    all_enemy_agent_yaw = []
    for agent_id, dict_value in enemy_agents_data.items():
        if agent_id != '231':
            all_enemy_agent_yaw.append([dict_value['Yaw']])
    """

    if '311' in self_data.keys():
        friend_agents_data = dict(self_data, **ally_agents_data)
    if '311' in ally_agents_data.keys():
        friend_agents_data = dict(ally_agents_data, **self_data)
    all_friend_agent_pos = []
    for agent_id, dict_value in friend_agents_data.items():
        all_friend_agent_pos.append([dict_value['X'], dict_value['Y'], dict_value['Z']])
    """
    all_friend_agent_yaw = []
    for agent_id, dict_value in friend_agents_data.items():
        all_friend_agent_yaw.append([dict_value['Yaw']])
    """

    target_location = np.array(key_points)
    red_car_current_pos = np.array(all_friend_agent_pos)
    blue_car_current_pos = np.array(all_enemy_agent_pos[0:2])
    blue_drone_current_pos = np.array([all_enemy_agent_pos[-1]])
    #red_car_current_yaw = np.array(all_friend_agent_yaw)
    #blue_car_current_yaw = np.array(all_enemy_agent_yaw)
    """
    blue_alive = []
    for agent_id, dict_value in enemy_agents_data.items():
        if agent_id is not '231':
            if dict_value['Blood'] == 0:
                blue_alive.append(False)
            else:
                blue_alive.append(True)

    red_alive = []
    for agent_id, dict_value in friend_agents_data.items():
        if dict_value['Blood'] == 0:
            red_alive.append(False)
        else:
            red_alive.append(True)
    """

    blue_drone_dist_to_go = np.array([[np.linalg.norm(blue_drone_current_pos[a][:2] - target_location[b][:2]) for b in range(2)]
                                      for a in range(1)])
    blue_drone_time_to_go = blue_drone_dist_to_go / blue_drone_max_vel
    if blue_drone_time_to_go[0][0] < 0.2:
        blue_drone_time_to_go[0][0] = -1000
    if blue_drone_time_to_go[0][1] < 0.2:
        blue_drone_time_to_go[0][1] = -1000

    red_car_dist_to_go = np.array(
        [[np.linalg.norm(red_car_current_pos[a][:2] - target_location[b][:2]) - expel_range for b in range(2)]
         for a in range(2)])
    red_car_time_to_go = red_car_dist_to_go / red_car_max_vel
    if red_alive[0] is False:
        red_car_time_to_go[0] = 100000
    elif red_alive[1] is False:
        red_car_time_to_go[1] = 100000

    red_car_next_pos = red_car_current_pos
    if 0:  #np.sum(red_alive) == 0:
        target_id = np.array([1, 1])
        #red_car_next_yaw = red_car_current_yaw
        #red_car_next_fire_yaw = np.array([[0], [0]])
        #red_car_fire_flag = [False, False]
    else:
        if np.sum(red_alive) == 2 and np.sum(blue_alive) == 2:
            red_blue_car_relative_dist = np.array(
                [[np.linalg.norm(red_car_current_pos[a][:2] - blue_car_current_pos[b][:2]) for b in
                  range(2)] for a in range(2)])
            _, col_index = linear_sum_assignment(red_blue_car_relative_dist)
            target_pos = blue_car_current_pos[col_index]
            target_id = col_index
        elif np.sum(red_alive) == 1 and np.sum(blue_alive) == 2:
            if red_alive[0] is True:
                red_blue_car_relative_dist = np.array(
                    [[np.linalg.norm(red_car_current_pos[0][:2] - blue_car_current_pos[b][:2]) for b in range(2)]])
                target_pos = np.array(
                    [blue_car_current_pos[np.argmin(red_blue_car_relative_dist)], red_car_current_pos[1]])
            elif red_alive[1] is True:
                red_blue_car_relative_dist = np.array(
                    [[np.linalg.norm(red_car_current_pos[1][:2] - blue_car_current_pos[b][:2]) for b in range(2)]])
                target_pos = np.array(
                    [red_car_current_pos[0], blue_car_current_pos[np.argmin(red_blue_car_relative_dist)]])
            target_id = np.array([np.argmin(red_blue_car_relative_dist), np.argmin(red_blue_car_relative_dist)])
        elif np.sum(red_alive) == 2 and np.sum(blue_alive) == 1:
            if blue_alive[0] is True:
                target_pos = np.array([blue_car_current_pos[0], blue_car_current_pos[0]])
                target_id = np.array([0, 0])
            elif blue_alive[1] is True:
                target_pos = np.array([blue_car_current_pos[1], blue_car_current_pos[1]])
                target_id = np.array([1, 1])
        elif np.sum(red_alive) == 1 and np.sum(blue_alive) == 1:
            if red_alive[0] is True:
                if blue_alive[0] is True:
                    target_pos = np.array([blue_car_current_pos[0], red_car_current_pos[1]])
                    target_id = np.array([0, 0])
                elif blue_alive[1] is True:
                    target_pos = np.array([blue_car_current_pos[1], red_car_current_pos[1]])
                    target_id = np.array([1, 1])
            elif red_alive[1] is True:
                if blue_alive[0] is True:
                    target_pos = np.array([red_car_current_pos[0], blue_car_current_pos[0]])
                    target_id = np.array([0, 0])
                elif blue_alive[1] is True:
                    target_pos = np.array([red_car_current_pos[0], blue_car_current_pos[1]])
                    target_id = np.array([1, 1])
        else:
            red_car_next_pos = red_car_current_pos
            target_id = np.array([1, 1])

        blue_success_time = blue_drone_time_to_go + time_to_win*0.0
        if (blue_success_time - np.min(red_car_time_to_go, axis=0, keepdims=True) >= 0).all() and np.sum(
                blue_alive) > 0:
            # 'offense'
            red_car_next_pos = target_pos
            flag = 'offense'
        else:
            # 'defense'
            target_defense_index = np.argmin(
                blue_success_time - np.min(red_car_time_to_go, axis=0, keepdims=True))
            if red_alive[0] is True and red_alive[1] is True:
                #red_car_next_pos = np.array(
                #   [target_location[target_defense_index], target_location[target_defense_index]])
                red_car_next_pos[0][:2] = blue_drone_current_pos[0][0:2]
                red_car_next_pos[1][:2] = blue_drone_current_pos[0][0:2]
            elif red_alive[0] is True and red_alive[1] is False:
                #red_car_next_pos = np.array(
                #    [target_location[target_defense_index], red_car_current_pos[1]])
                red_car_next_pos[0][:2] = blue_drone_current_pos[0][0:2]
            elif red_alive[0] is False and red_alive[1] is True:
                #red_car_next_pos = np.array(
                #    [red_car_current_pos[0], target_location[target_defense_index]])
                red_car_next_pos[1][:2] = blue_drone_current_pos[0][0:2]
            flag = 'defense'
        """
        agent_yaw = [0, 0]
        fire_yaw = [0, 0]
        fire_flag = [False, False]
        for index in range(2):
            relative_dist = np.linalg.norm(red_car_current_pos[index] - target_pos[index])
            relative_yaw = math.degrees(math.atan2((target_pos[index][1] - red_car_current_pos[index][1]),
                                                   (target_pos[index][0] - red_car_current_pos[index][0])))
            if red_alive[index] is True:
                if relative_dist < fire_dist:
                    red_car_next_pos[index] = red_car_current_pos[index]

                    if ((relative_yaw < red_car_current_yaw[index] - 90) or (
                            relative_yaw > red_car_current_yaw[index] + 90)):  # 在车的基础上，旋转打击载荷
                        agent_yaw[index] = relative_yaw  # 直接旋转车朝向
                        fire_yaw[index] = 0
                    else:
                        agent_yaw[index] = red_car_current_yaw[index]
                        fire_yaw[index] = relative_yaw - red_car_current_yaw[index]
                    fire_flag[index] = True
                else:
                    fire_flag[index] = False
                    agent_yaw[index] = red_car_current_yaw[index]
                    fire_yaw[index] = 0
            else:
                fire_flag[index] = False
                agent_yaw[index] = red_car_current_yaw[index]
                fire_yaw[index] = 0

        red_car_next_yaw = np.array(agent_yaw)
        red_car_next_fire_yaw = np.array(fire_yaw)
        """



        if '311' in self_data.keys():
            return red_car_next_pos[0], target_id[0], flag
            #return [red_car_next_pos[0], red_car_next_yaw [0], red_car_next_fire_yaw[0], fire_flag[0]]
        else:
            return red_car_next_pos[1], target_id[1], flag
            #return [red_car_next_pos[1], red_car_next_yaw [1], red_car_next_fire_yaw[1], fire_flag[1]]

if __name__ == '__main__':
    # 211和221是进攻方blue小车， 231是进攻方blue无人机
    # 311和321是防守方red小车
    # 当前要决策的防守方red小车的信息
    self_data = {'321': {'X': -3.0, 'Y': 2.5, 'Z': 0, 'Yaw': 10, 'Blood': 100}}
    # 其余防守方red小车的信息
    ally_agents_data = {'311': {'X': 0.0, 'Y': 1.5, 'Z': 0, 'Yaw': 20, 'Blood': 100}}
    # 进攻方blue小车和无人机信息
    enemy_agents_data = {'211': {'X': 1.5, 'Y': -2.0, 'Z': 0, 'Yaw': 30, 'Blood': 100},
                         '221': {'X': -2.5, 'Y': -2.5, 'Z': 0, 'Yaw': 40, 'Blood': 100},
                         '231': {'X': 0.7, 'Y': 3.3, 'Z': 1.5, 'Yaw': 0}}
    # 夺控点信息
    key_points = [[0.7, 3.3, 0], [-3.0, -0.7, 0]]
    # 存活状态
    blue_alive = [True, True]
    red_alive = [True, True]
    target_position, target_id, flag = defense_combat(self_data, ally_agents_data, enemy_agents_data, key_points, blue_alive, red_alive)
    print(target_position)
    print('\r\n')
    print(target_id)
    print('\r\n')
    print(flag)






