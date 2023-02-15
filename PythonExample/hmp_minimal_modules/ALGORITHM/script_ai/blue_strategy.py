import numpy as np
import math


def offense_combat(self_data, ally_agents_data, enemy_agents_data, key_points, blue_alive, red_alive,
                   agent_type):
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
    blue_drone_current_pos = np.array([[ally_agents_data['231']['X'], ally_agents_data['231']['Y'], ally_agents_data['231']['Z']]])
    ally_agents_data.pop('231')
    if '211' in self_data.keys():
        friend_agents_data = dict(self_data, **ally_agents_data)
    if '211' in ally_agents_data.keys():
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
    red_car_current_pos = np.array(all_enemy_agent_pos)
    blue_car_current_pos = np.array(all_friend_agent_pos)
    blue_car_current_pos = blue_car_current_pos

    if agent_type == 0:
        drone_keypoint_relative_dist = np.array([[np.linalg.norm(blue_drone_current_pos[0][:2] - target_location[b][:2])
                                                  for b in range(2)]])
        if red_alive[0] is True and red_alive[1] is True:
            dist_red_car_blue_drone = np.array(
                [[np.linalg.norm(red_car_current_pos[a][:2] - blue_drone_current_pos[0][:2])]
                 for a in range(2)])
            if np.min(dist_red_car_blue_drone) < expel_range:# and np.argmin(drone_keypoint_relative_dist) < 600:
                flag = 'defense'
            else:
                flag = 'offense'
        elif red_alive[0] is True and red_alive[1] is False:
            dist_red_car_blue_drone = np.linalg.norm(red_car_current_pos[0][:2] - blue_drone_current_pos[0][:2])
            if dist_red_car_blue_drone < expel_range:# and np.argmin(drone_keypoint_relative_dist) < 600:
                flag = 'defense'
            else:
                flag = 'offense'
        elif red_alive[0] is False and red_alive[1] is True:
            dist_red_car_blue_drone = np.linalg.norm(red_car_current_pos[1][:2] - blue_drone_current_pos[0][:2])
            if dist_red_car_blue_drone < expel_range:# and np.argmin(drone_keypoint_relative_dist) < 600:
                flag = 'defense'
            else:
                flag = 'offense'
        elif red_alive[0] is False and red_alive[1] is False:
            flag = 'offense'
        if flag == 'offense':
            if blue_alive[0] is True or blue_alive[1] is True:
                # 无人车
                if blue_alive[0] is True and blue_alive[1] is True:
                    center_x = (blue_car_current_pos[0][0] + blue_car_current_pos[1][0]) / 2
                    center_y = (blue_car_current_pos[0][1] + blue_car_current_pos[1][1]) / 2
                    center_z = (blue_car_current_pos[0][2] + blue_car_current_pos[1][2]) / 2
                elif blue_alive[0] is True and blue_alive[1] is False:
                    center_x = blue_car_current_pos[0][0]
                    center_y = blue_car_current_pos[0][1]
                    center_z = blue_car_current_pos[0][2]
                elif blue_alive[0] is False and blue_alive[1] is True:
                    center_x = blue_car_current_pos[1][0]
                    center_y = blue_car_current_pos[1][1]
                    center_z = blue_car_current_pos[1][2]

                blue_center = np.array([[center_x, center_y, center_z]])

                if red_alive[0] is True and red_alive[1] is True:
                    dist = np.array([[np.linalg.norm(blue_center[0][:2] - red_car_current_pos[b][:2])
                                      for b in range(2)]])
                    target_pos = red_car_current_pos[np.argmin(dist)]
                    target_id = np.argmin(dist)
                elif red_alive[0] is True and red_alive[1] is False:
                    target_pos = red_car_current_pos[0]
                    target_id = 0
                elif red_alive[0] is False and red_alive[1] is True:
                    target_pos = red_car_current_pos[1]
                    target_id = 1
                elif red_alive[0] is False and red_alive[1] is False:
                    target_pos = target_location[0]
                    target_id = 1
            else:
                target_pos = blue_car_current_pos[0]
                target_id = 0
        else:
            target_pos = blue_drone_current_pos[0]
            target_pos[0] = target_pos[0] + (-2000 * np.sign(blue_drone_current_pos[0][0]))
            target_pos[1] = target_pos[1] + (-3000 * np.sign(blue_drone_current_pos[0][1]))
            target_pos[2] = blue_car_current_pos[0][2]
            target_id = 0
        return target_pos, target_id, flag
    elif agent_type == 1:
        # 无人机
        drone_keypoint_relative_dist = np.array([[np.linalg.norm(blue_drone_current_pos[0][:2] - target_location[b][:2])
                                                  for b in range(2)]])
        safe_target_location = []
        for index in range(2):
            if red_alive[0] is True and red_alive[1] is True:
                dist = np.array([[np.linalg.norm(target_location[index][:2] - red_car_current_pos[b][:2])
                                  for b in range(2)]])
                if np.min(dist) > expel_range:
                    safe_target_location.append(True)
                else:
                    safe_target_location.append(False)
            elif red_alive[0] is True and red_alive[1] is False:
                dist = np.linalg.norm(target_location[index][:2] - red_car_current_pos[0][:2])
                if dist > expel_range:
                    safe_target_location.append(True)
                else:
                    safe_target_location.append(False)
            elif red_alive[0] is False and red_alive[1] is True:
                dist = np.linalg.norm(target_location[index][:2] - red_car_current_pos[1][:2])
                if dist > expel_range:
                    safe_target_location.append(True)
                else:
                    safe_target_location.append(False)
            elif red_alive[0] is False and red_alive[1] is False:
                safe_target_location = [True, True]

        if safe_target_location[0] is True and safe_target_location[1] is True:
            red_car_dist = np.array(
                [[np.linalg.norm(red_car_current_pos[a][:2] - target_location[b][:2]) - expel_range for b in range(2)]
                 for a in range(2)])
            if red_alive[0] is False:
                red_car_dist[0] = 100000
            elif red_alive[1] is False:
                red_car_dist[1] = 100000

            target_defense_index = np.argmax(np.max(red_car_dist, axis=0, keepdims=True))
            blue_drone_dist_to_go = np.array(
                [[np.linalg.norm(blue_drone_current_pos[a][:2] - target_location[b][:2]) for b in range(2)]
                 for a in range(1)])
            # return target_location[np.argmin(blue_drone_dist_to_go)]
            return target_location[target_defense_index]
        elif safe_target_location[0] is True and safe_target_location[1] is False:
            return target_location[0]
        elif safe_target_location[0] is False and safe_target_location[1] is True:
            return target_location[1]
        elif safe_target_location[0] is False and safe_target_location[1] is False:
            return np.array([(target_location[0][0] + target_location[0][1]) / 2,
                             (target_location[1][0] + target_location[1][1]) / 2])


if __name__ == '__main__':
    agent_type = 0
    # 211和221是进攻方blue小车， 231是进攻方blue无人机
    # 311和321是防守方red小车
    # 当前要决策的blue小车的信息
    self_data = {'211': {'X': 1500, 'Y': -2000, 'Z': 0, 'Yaw': 30, 'Blood': 100}}
    ally_agents_data = {'221': {'X': -2500, 'Y': -2500, 'Z': 0, 'Yaw': 40, 'Blood': 100},
                        '231': {'X': 700, 'Y': 3300, 'Z': 1500, 'Yaw': 0}}
    # 进攻方blue小车和无人机信息
    enemy_agents_data = {'311': {'X': 2700.0, 'Y': 3300, 'Z': 0, 'Yaw': 20, 'Blood': 100},
                         '321': {'X': -1000, 'Y': -700, 'Z': 0, 'Yaw': 10, 'Blood': 100}}
    # 夺控点信息
    key_points = [[700, 3300, 0], [-2500, -700, 0]]
    # 存活状态
    blue_alive = [True, True]
    red_alive = [True, True]
    if agent_type == 0:
        target_position, target_id, flag = offense_combat(self_data, ally_agents_data, enemy_agents_data,
                                                          key_points, blue_alive, red_alive, agent_type)
        print(target_position)
        print('\r\n')
        print(target_id)
        print('\r\n')
        print(flag)
    elif agent_type == 1:
        target_position = offense_combat(self_data, ally_agents_data, enemy_agents_data,
                                         key_points, blue_alive, red_alive, agent_type)
        print(target_position)
