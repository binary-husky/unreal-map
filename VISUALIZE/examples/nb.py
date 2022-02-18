import numpy as np
import random, time
from UTILS.tensor_ops import distance_matrix, repeat_at, delta_matrix
from VISUALIZE.mcom import mcom
colors = ['AliceBlue', 'AntiqueWhite', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque', 'Black', 'BlanchedAlmond', 'Blue', 'BlueViolet', 'Brown', 'BurlyWood', 'CadetBlue', 'Chartreuse', 'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan', 'DarkBlue', 'DarkCyan', 'DarkGoldenRod', 'DarkGray', 'DarkGrey', 'DarkGreen', 'DarkKhaki', 'DarkMagenta', 'DarkOliveGreen', 'DarkOrange', 'DarkOrchid', 'DarkRed', 'DarkSalmon', 'DarkSeaGreen', 'DarkSlateBlue', 'DarkSlateGray', 'DarkSlateGrey', 'DarkTurquoise', 'DarkViolet', 'DeepPink', 'DeepSkyBlue', 'DimGray', 'DimGrey', 'DodgerBlue', 'FireBrick', 'FloralWhite', 'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod', 'Gray', 'Grey', 'Green', 'GreenYellow', 'HoneyDew', 'HotPink', 'IndianRed', 'Indigo', 'Ivory', 'Khaki', 'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue', 'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey', 'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue', 'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime', 'LimeGreen', 'Linen', 'Magenta', 'Maroon', 'MediumAquaMarine', 'MediumBlue', 'MediumOrchid', 'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen', 'MediumTurquoise', 'MediumVioletRed', 'MidnightBlue', 'MintCream', 'MistyRose', 'Moccasin', 'NavajoWhite', 'Navy', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed', 'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed', 'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple', 'RebeccaPurple', 'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Salmon', 'SandyBrown', 'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue', 'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'Tan', 'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White', 'WhiteSmoke', 'Yellow', 'YellowGreen']
random.shuffle(colors)
def run():
    可视化桥 = mcom(path='RECYCLE/v2d_logger/', draw_mode='Threejs')
    可视化桥.初始化3D(); 可视化桥.设置样式('star')
    可视化桥.其他几何体之旋转缩放和平移('ball', 'SphereGeometry(1)',   0,0,0,  1,1,1, 0,0,0) # 球体   
    # 设置一个引力中心
    N体 = 20
    theta = 2*np.pi/N体 # 120°
    相位 = [0] + [np.random.rand()*2*np.pi for i in range(N体-1)]
    位置 = [[0,0,0]]+[[np.cos(相位[i]), np.sin(相位[i]), (np.random.rand()-0.5)/15] for i in range(N体-1)]; 位置 = np.array(位置)*20
    for i in range(N体): 位置[i] = 位置[i] * (np.random.rand()+1)
    速度 = [[0,0,0]]+[[np.cos(相位[i]+np.pi/2), np.sin(相位[i]+np.pi/2), (np.random.rand()-0.5)] for i in range(N体-1)]; 速度 = np.array(速度)*7
    质量 = [100]+[np.random.rand()+0.5 for _ in range(N体-1)]; 质量 = np.array(质量)
    G = 16 # 常量 G
    assert N体==len(位置) and N体==len(速度)
    dT = 0.01 # 时间间隔
    for step in range(100000): # F = (GMm)/距离矩阵^2
        距离矩阵 = distance_matrix(位置) # 计算距离矩阵
        距离矩阵平方 = 距离矩阵 * 距离矩阵.T
        # 计算Mm矩阵：
        质量矩阵 = repeat_at(tensor=质量, insert_dim=-1, n_times=N体)   
        Mm矩阵 = 质量矩阵 * 质量矩阵.T
        引力标量矩阵 = G*Mm矩阵/(距离矩阵平方+1e-10) # 计算引力
        for i in range(N体): 引力标量矩阵[i,i] = 0  # 对角线元素清零
        # 引力的方向，即位移的方向
        引力方向 = delta_matrix(位置) # 形状为 (N体, N体, 3)
        # 引力方向归一化，乘以引力矩阵（标量）
        引力方向 = 引力方向/(np.linalg.norm(引力方向,axis=-1, keepdims=True)+1e-10)
        引力 = 引力方向*repeat_at(tensor=引力标量矩阵, insert_dim=-1, n_times=3)
        引力合力 = 引力.sum(1)
        加速度 = 引力合力/repeat_at(tensor=质量, insert_dim=-1, n_times=3)
        # 动力学 和 运动学
        速度 = 速度 + 加速度*dT
        位置 = 位置 + 速度*dT
        if step%10==0:
            for body in range(N体):
                size = 2 if 质量[body]>2 else 质量[body]*0.42
                可视化桥.发送几何体('ball|%d|%s|%.2f'%(body, colors[body], size),  # 填入核心参量： “已声明的形状|几何体之ID标识|颜色|整体大小”
                    位置[body, 0], 位置[body, 1], 位置[body, 2], ro_x=0, ro_y=0, ro_z=0, # 6 DOF
                    opacity=1, label='',  track_n_frame=20,  track_tension=0, track_color='white')
        可视化桥.结束关键帧()
        print('\r %d/100000'%step, end='', flush=True)
if __name__ == "__main__":
    run(); time.sleep(10000)