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
    可视化桥.其他几何体之旋转缩放和平移('Plane', 'fbx=/examples/files/plane.fbx', -np.pi/2, 0, np.pi/2,  1,1,1, 0,0,0)   # 八面体

    body = 1

    for t in range(60):
        可视化桥.发送几何体('Plane|%d|%s|%.2f'%(body, colors[body], 1),  # 填入核心参量： “已声明的形状|几何体之ID标识|颜色|整体大小”
            0, 0, 0, ro_x=0, ro_y=0, ro_z=(np.pi*t/120), ro_order='ZYX', # 6 DOF
            opacity=1, label='yaw', label_color='white',  track_n_frame=20,  track_tension=0, track_color='white')

        可视化桥.结束关键帧()

    for t in range(60):
        可视化桥.发送几何体('Plane|%d|%s|%.2f'%(body, colors[body], 1),  # 填入核心参量： “已声明的形状|几何体之ID标识|颜色|整体大小”
            0, 0, 0, ro_x=0, ro_y=(np.pi*t/240), ro_z=(np.pi*60/120), ro_order='ZYX', # 6 DOF
            opacity=1, label='pitch', label_color='white', track_n_frame=20,  track_tension=0, track_color='white')

        可视化桥.结束关键帧()

    for t in range(60):
        可视化桥.发送几何体('Plane|%d|%s|%.2f'%(body, colors[body], 1),  # 填入核心参量： “已声明的形状|几何体之ID标识|颜色|整体大小”
            0, 0, 0, ro_x=(np.pi*t/120), ro_y=(np.pi*60/240), ro_z=(np.pi*60/120), ro_order='ZYX',# 6 DOF
            opacity=1, label='roll', label_color='white', track_n_frame=20,  track_tension=0, track_color='white')

        可视化桥.结束关键帧()

    '''
    for t in range(60):
        可视化桥.发送几何体('Plane|%d|%s|%.2f'%(body, colors[body], 1),  # 填入核心参量： “已声明的形状|几何体之ID标识|颜色|整体大小”
            0, 0, 0, ro_x=(np.pi*t/120), ro_y=0, ro_z=0, ro_order='XZY', # 6 DOF
            opacity=1, label='ro_x', label_color='white',  track_n_frame=20,  track_tension=0, track_color='white')

        可视化桥.结束关键帧()

    for t in range(60):
        可视化桥.发送几何体('Plane|%d|%s|%.2f'%(body, colors[body], 1),  # 填入核心参量： “已声明的形状|几何体之ID标识|颜色|整体大小”
            0, 0, 0, ro_x=(np.pi*60/120), ro_y=0, ro_z=(np.pi*t/120), ro_order='XZY', # 6 DOF
            opacity=1, label='ro_z', label_color='white', track_n_frame=20,  track_tension=0, track_color='white')

        可视化桥.结束关键帧()

    for t in range(60):
        可视化桥.发送几何体('Plane|%d|%s|%.2f'%(body, colors[body], 1),  # 填入核心参量： “已声明的形状|几何体之ID标识|颜色|整体大小”
            0, 0, 0, ro_x=(np.pi*60/120), ro_y=(np.pi*t/120), ro_z=(np.pi*60/120), ro_order='XZY', # 6 DOF
            opacity=1, label='ro_y', label_color='white', track_n_frame=20,  track_tension=0, track_color='white')

        可视化桥.结束关键帧()
    '''


    '''
    for t in range(60):
        可视化桥.发送几何体('Plane|%d|%s|%.2f'%(body, colors[body], 1),  # 填入核心参量： “已声明的形状|几何体之ID标识|颜色|整体大小”
            0, 0, 0, ro_x=(np.pi*t/120), ro_y=0, ro_z=0, ro_order='XYZ',# 6 DOF
            opacity=1, label='ro_x', label_color='white',  track_n_frame=20,  track_tension=0, track_color='white')

        可视化桥.结束关键帧()

    for t in range(60):
        可视化桥.发送几何体('Plane|%d|%s|%.2f'%(body, colors[body], 1),  # 填入核心参量： “已声明的形状|几何体之ID标识|颜色|整体大小”
            0, 0, 0, ro_x=(np.pi*60/120), ro_y=(np.pi*t/120), ro_z=0, ro_order='XYZ', # 6 DOF
            opacity=1, label='ro_y', label_color='white', track_n_frame=20,  track_tension=0, track_color='white')

        可视化桥.结束关键帧()

    for t in range(60):
        可视化桥.发送几何体('Plane|%d|%s|%.2f'%(body, colors[body], 1),  # 填入核心参量： “已声明的形状|几何体之ID标识|颜色|整体大小”
            0, 0, 0, ro_x=(np.pi*60/120), ro_y=(np.pi*60/120), ro_z=(np.pi*t/120), ro_order='XYZ',# 6 DOF
            opacity=1, label='ro_z', label_color='white', track_n_frame=20,  track_tension=0, track_color='white')

        可视化桥.结束关键帧()
    '''



if __name__ == "__main__":
    run(); time.sleep(10000)