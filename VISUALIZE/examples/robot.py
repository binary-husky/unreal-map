import numpy as np
import random, time, os
from scipy.integrate import ode
from UTIL.tensor_ops import distance_matrix, repeat_at, delta_matrix
from VISUALIZE.mcom import mcom
PI = np.pi


def gen_shape(可视化桥, index, x, y, z, rox, roy, roz):
    可视化桥.发送几何体(
        'robot-p1|%s|%s|%.2f'%(str(index)+'-1', 'Pink', 0.1),  
        x, y, z, ro_x=rox, ro_y=roy, ro_z=roz, # 6 DOF
        opacity=1, label='', label_color='white',  
    )
    可视化桥.发送几何体(
        'robot-p2|%s|%s|%.2f'%(str(index)+'-2', 'White', 0.1),  
        x, y, z, ro_x=rox, ro_y=roy, ro_z=roz, # 6 DOF
        opacity=1, label='', label_color='white',  
    )
    可视化桥.发送几何体(
        'robot-p3|%s|%s|%.2f'%(str(index)+'-3', 'White', 0.1),  
        x, y, z, ro_x=rox, ro_y=roy, ro_z=roz, # 6 DOF
        opacity=1, label='', label_color='white',  
    )
    可视化桥.发送几何体(
        'robot-p4|%s|%s|%.2f'%(str(index)+'-4', 'Blue', 0.1),  
        x, y, z, ro_x=rox, ro_y=roy, ro_z=roz, # 6 DOF
        opacity=1, label='', label_color='white',  
    )
    可视化桥.发送几何体(
        'robot-p5|%s|%s|%.2f'%(str(index)+'-5', 'Blue', 0.1),  
        x, y, z, ro_x=rox, ro_y=roy, ro_z=roz, # 6 DOF
        opacity=1, label='', label_color='white',  
    )

def deg2rad(deg):
    return deg*np.pi/180

def run():
    # 可视化界面初始化
    可视化桥 = mcom(path='TEMP/v2d_logger/', draw_mode='Threejs')
    可视化桥.初始化3D()
    # 可视化桥.设置样式('star')
    # 可视化桥.设置样式('many star')

    可视化桥.设置样式('background', color='Gray') # 注意不可以省略参数键值'color=' ！

    可视化桥.其他几何体之旋转缩放和平移(
        'robot-p1', 
        'SphereGeometry(1,32,16,0,Math.PI*2,0,1.099)', 
        0,0,0,  
        0.7,0.7,1.6, 
        0, 0, -0.473)
    可视化桥.其他几何体之旋转缩放和平移(
        'robot-p2',
        'SphereGeometry(1)',
        0,0,0,
        0.1,0.1,0.1,
        0.36,-0.15,0.80)
    可视化桥.其他几何体之旋转缩放和平移(
        'robot-p3',
        'SphereGeometry(1)',
        0,0,0,
        0.1,0.1,0.1,
        0.36,0.15,0.80)

    可视化桥.其他几何体之旋转缩放和平移(
        'robot-p4', 
        'CylinderGeometry(0.02, 0.02, 0.4, 8)',
        -deg2rad(45),0,0,  
        1, 1, 1,
        0, 0.6, 0.696,
        )
    可视化桥.其他几何体之旋转缩放和平移(
        'robot-p5',
        'CylinderGeometry(0.02, 0.02, 0.4, 8)',
        deg2rad(45),0,0,  
        1, 1, 1,
        0, -0.6, 0.696,
        )




# for i in range(20):
    # 填入核心参量： “已声明的形状|几何体之ID标识|颜色|整体大小” 
    gen_shape(可视化桥,0, 0,0,0, 0,0,0)
    可视化桥.结束关键帧()


if __name__ == "__main__":
    run()
    time.sleep(10000)