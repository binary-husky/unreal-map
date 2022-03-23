import numpy as np
from ..env_cmd import CmdEnv
from UTILS.tensor_ops import dir2rad, np_softmax, reg_rad_at, reg_rad, reg_deg_at


def maneuver_cold_to_ms(uav):
    unit_delta = uav.delta_oppsite_to_ms()
    H2 = unit_delta * 100e3 + uav.pos2d

    def check_dis(goto_location):
        d = goto_location[0]
        dis = np.linalg.norm(uav.pos3d - np.array([d['X'], d['Y'], d['Z']]))
        assert dis > 10e3

    goto_location = [
        {
            "X": H2[0], "Y": H2[1], "Z": uav.Z
        }
    ]
    check_dis(goto_location)
    return goto_location


def maneuver_vertical_to_ms(uav):
    unit_delta = uav.delta_oppsite_to_ms()
    unit_delta = np.append(unit_delta, 0)
    unit_delta = np.cross(unit_delta, np.array([0, 0, 1]))
    H2 = unit_delta[:2] * 100e3 + uav.pos2d

    def check_dis(goto_location):
        d = goto_location[0]
        dis = np.linalg.norm(uav.pos3d - np.array([d['X'], d['Y'], d['Z']]))
        assert dis > 10e3

    goto_location = [
        {
            "X": H2[0],
            "Y": H2[1],
            "Z": uav.Z
        }
    ]
    check_dis(goto_location)
    return goto_location


def check_dis(goto_location, uav):
    d = goto_location[0]
    dis = np.linalg.norm(uav.pos3d - np.array([d['X'], d['Y'], d['Z']]))
    assert dis > 10e3


# def choose_maneuver_side(uav, angle):
#     pos = +angle
#     neg = -angle
#     deg_ms_coming = dir2rad(uav.delta_oppsite_to_ms()[:2]) * 180/np.pi

def maneuver_angle_to_ms(uav, angle):
    rad = angle * np.pi / 180

    unit_delta = np.matmul(
        uav.delta_oppsite_to_ms()[:2],
        np.array([[np.cos(rad), np.sin(rad)],
                  [np.sin(-rad), np.cos(rad)]]))
    if angle != 0 and angle != 180:
        unit_delta_side2 = np.matmul(
            uav.delta_oppsite_to_ms()[:2],
            np.array([[np.cos(rad), np.sin(-rad)],
                      [np.sin(rad), np.cos(rad)]]))

        rad1 = dir2rad(unit_delta)
        rad2 = dir2rad(unit_delta_side2)
        uav_head_rad = np.pi / 2 - uav.Heading
        rad1 = reg_rad_at(rad1, uav_head_rad)
        rad2 = reg_rad_at(rad2, uav_head_rad)
        delta1 = np.abs(rad1 - uav_head_rad) * 180 / np.pi
        delta2 = np.abs(rad2 - uav_head_rad) * 180 / np.pi
        if delta2 < delta1 - 3:  # 另一侧机动
            unit_delta = unit_delta_side2

    H2 = unit_delta[:2] * 100e3 + uav.pos2d

    goto_location = [
        {
            "X": H2[0],
            "Y": H2[1],
            "Z": uav.Z
        }
    ]
    check_dis(goto_location, uav)
    return goto_location
    
def maneuver_angle_to_op_vip(uav, vip, angle):
    rad = angle * np.pi / 180
    delta_oppsite_to_vip = uav.pos3d - vip.pos3d
    unit_delta = np.matmul(
        delta_oppsite_to_vip[:2],
        np.array([[np.cos(rad), np.sin(rad)],
                  [np.sin(-rad), np.cos(rad)]]))
    if angle != 0 and angle != 180:
        unit_delta_side2 = np.matmul(
            delta_oppsite_to_vip[:2],
            np.array([[np.cos(rad), np.sin(-rad)],
                      [np.sin(rad), np.cos(rad)]]))

        rad1 = dir2rad(unit_delta)
        rad2 = dir2rad(unit_delta_side2)
        # uav_head_rad = np.pi / 2 - uav.Heading
        vip_head_rad = np.pi / 2 - vip.Heading
        rad1 = reg_rad_at(rad1, vip_head_rad)
        rad2 = reg_rad_at(rad2, vip_head_rad)
        delta1 = np.abs(rad1 - vip_head_rad) * 180 / np.pi
        delta2 = np.abs(rad2 - vip_head_rad) * 180 / np.pi
        if delta2 > delta1 - 3:  # 另一侧机动
            unit_delta = unit_delta_side2

    H2 = unit_delta[:2] * 100e3 + uav.pos2d

    goto_location = [
        {
            "X": H2[0],
            "Y": H2[1],
            "Z": uav.Z - 1000 # 因为要下高
        }
    ]
    check_dis(goto_location, uav)
    return goto_location
    


def maneuver_vip(uav, angle):
    # rad = angle * np.pi / 180
    ms = uav.nearest_ms()
    # ms.h_angle 是角度制
    angle_esc_deg01 = ms.h_angle + angle
    angle_esc_deg02 = ms.h_angle - angle

    uav_head_deg = 90 - uav.Heading*180/np.pi
    angle_esc_deg01 = reg_deg_at(angle_esc_deg01, uav_head_deg)
    angle_esc_deg02 = reg_deg_at(angle_esc_deg02, uav_head_deg)
    delta1 = np.abs(uav_head_deg - angle_esc_deg01)
    delta2 = np.abs(uav_head_deg - angle_esc_deg02)
    if delta2 < delta1:
        angle_esc_deg = angle_esc_deg02
    else:
        angle_esc_deg = angle_esc_deg01

    angle_esc_rad = angle_esc_deg * np.pi/180
    unit_delta = np.array([np.cos(angle_esc_rad), np.sin(angle_esc_rad)])
    H2 = unit_delta[:2] * 100e3 + uav.pos2d

    goto_location = [
        {
            "X": H2[0],
            "Y": H2[1],
            "Z": uav.Z
        }
    ]
    check_dis(goto_location, uav)
    return goto_location


def maneuver_angle_to_ms3d(uav, angle):
    ms = uav.nearest_ms()
    dir_ = ms.pos3d - uav.pos3d
    dir_ = dir_ / np.linalg.norm(dir_)
    H3 = dir_ * 100e3 + uav.pos3d

    goto_location = [
        {
            "X": H3[0],
            "Y": H3[1],
            "Z": H3[2]
        }
    ]
    check_dis(goto_location, uav)
    return goto_location


def maneuver_speed_to_ms(uav):
    ms = uav.nearest_ms()
    d_angle = reg_deg_at(uav.h_angle, ref=ms.h_angle + 180) - (ms.h_angle + 180)
    # 处于转向阶段的飞机 不适用
    if np.abs(d_angle) > 10:
        # with open('./log/%s' % str(ms.ID), 'a+') as f:
        #     f.write('角度过大不可调整 %.2f \n' % (d_angle))
        return uav.MaxSpeed
    
    # 开始
    # with open('./log/%s' % str(ms.ID), 'a+') as f:
    #     f.write('调整角度 %.2f \n' % (d_angle))

    def de_(uav, ms, Dv, Dvmax, rsafe):
        Dx = [19.6, 19.6 * 2, 19.6 * 3, 19.6 * 4]
        L = [19.6, 78.4, 176.4, 313.6]
        
        def choose_D(N):
            _0dx = N - 1
            center_mark = (Dv + Dvmax) / 2
            tmp = (center_mark - L[_0dx]) // Dx[_0dx]
            better = tmp if np.abs(L[_0dx] + tmp * Dx[_0dx] - center_mark) <= np.abs(
                L[_0dx] + (tmp + 1) * Dx[_0dx] - center_mark) else tmp + 1
            if (L[_0dx] + better * Dx[_0dx]) < Dvmax:
                D = better
            else:
                ## print('unexpected')
                D = 0
            return D

        def qe():
            for _0dx in range(3):
                _1dx = _0dx + 1
                if L[_0dx + 1] > Dvmax:
                    return _1dx, choose_D(N=_1dx)
            return 4, choose_D(N=4)  # 最多允许 t=4的变速

        N, D = qe()
        if not rsafe:
            D = 0
            if N > 2: N = 2
        # N = np.ceil(min(Dv/(9.8*2), 5))#1
        ms.previous_change_taking_effect = 2 * N - 1 + D
        ms.keep_shifting = N - 1 + D
        ms.current_N = N
        # with open('./log/%s' % str(ms.ID), 'a+') as f:
        #     f.write('计划调整, N=%d D=%d, rsafe=%s 目标区间[%.2f, %.2f] \n' % (
        #     N, D, str(rsafe), ms.ter_dis_est + Dv, ms.ter_dis_est + Dvmax))
        return uav.MaxSpeed - ms.current_N * (2 * 9.8)

    if ms.previous_change_taking_effect > 0:
        ms.previous_change_taking_effect -= 1
        if ms.keep_shifting > 0:
            ms.keep_shifting -= 1
            # 开始
            # with open('./log/%s' % str(ms.ID), 'a+') as f:
            #     f.write('调整作用中 %.2f \n' % (d_angle))
            return uav.MaxSpeed - ms.current_N * (2 * 9.8)
        else:
            # with open('./log/%s' % str(ms.ID), 'a+') as f:
            #     f.write('等待调整 %.2f \n' % (d_angle))
            return uav.MaxSpeed

    def get_win(speed):
        # P_list = [[125, 170, 107, 174],
        #           [263, 308, 210, 283],
        #           [395, 437, 310, 388],
        #           [528, 575, 413, 487],
        #           [650, 705, 519, 592],
        #           [784, 820, 633, 696],
        #           [912, 967, 720, 790],
        #           [1040, 1086, 830, 900],
        #           [1171, 1300, 
        #           940, 1033], ]
        P_list = [[125, 170, 107, 174],
                  [263, 308, 210, 283],
                  [395, 437, 310, 388],
                  [528, 575, 413, 487],
                  [650, 705, 519, 592],
                  [784, 820, 633, 696],
                  [912, 967, 720, 790],
                  [1040, 1086, 830, 900],
                  [1171, 1300-50, 
                  940, 1033-50], ]
        def get_range(speed, P):
            # 左上，右上，左下，右下
            y = speed
            y1 = 1000
            y2 = 733
            x1 = P[0]
            x2 = P[2]
            min_ = x1 + (y - y1) / (y2 - y1) * (x2 - x1)
            x1 = P[1]
            x2 = P[3]
            max_ = x1 + (y - y1) / (y2 - y1) * (x2 - x1)
            return [min_, max_]

        return [get_range(speed, p) for p in P_list]

    win = get_win(ms.ter_ms_speed)
    for t in range(len(win)):
        if ms.ter_dis_est < win[t][0]:
            return de_(uav, ms, Dv=(win[t][0] - ms.ter_dis_est), Dvmax=(win[t][1] - ms.ter_dis_est),
                       rsafe=(t != len(win) - 1))
        elif ms.ter_dis_est <= win[t][1]:
            # with open('./log/%s' % str(ms.ID), 'a+') as f:
            #     f.write('无需调整 %s\n' % (str(win[t])) )
            return uav.MaxSpeed
        else:
            continue
    return de_(uav, ms, Dv=(win[0][0] + (ms.ter_ms_speed + uav.MaxSpeed - ms.ter_dis_est)),
               Dvmax=(win[0][1] + (ms.ter_ms_speed + uav.MaxSpeed - ms.ter_dis_est)), rsafe=True)