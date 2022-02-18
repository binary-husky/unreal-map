import numpy as np
import random, time
from scipy.integrate import ode
from UTILS.tensor_ops import distance_matrix, repeat_at, delta_matrix
from VISUALIZE.mcom import mcom
PI = np.pi
def run():
    # 可视化界面初始化
    可视化桥 = mcom(path='RECYCLE/v2d_logger/', draw_mode='Threejs')
    可视化桥.初始化3D(); 可视化桥.设置样式('star')
    可视化桥.其他几何体之旋转缩放和平移('ball', 'SphereGeometry(1)',   0,0,0,  1,1,1, 0,0,0) # 球体   
    # 设置初始参数
    天文单位 = 1.496e11 # m
    地球公转速度 = 2.98e4   # m/s
    地球质量 = 5.965e24     # kg
    太阳质量 = 1.989e30     # kg
    colors = ['Yellow', 'Red', 'Blue']
    size = [4, 4, 4]
    labels = ['Sun_1', 'Sun_2', 'Sun_3']
    位置 = np.stack([
        np.array([np.cos(0), np.sin(0), 0])         * 天文单位/2,
        np.array([np.cos(PI), np.sin(PI), 0])       * 天文单位/2,
        np.array([np.cos(PI/2), np.sin(PI/2), 0])   * 天文单位*2,
    ])
    速度 = np.stack([
        np.array([np.cos(0 +PI/2),  np.sin(0   +PI/2), 0])     * 地球公转速度/1.4,
        np.array([np.cos(PI +PI/2), np.sin(PI +PI/2), 0])      * 地球公转速度/1.4,
        np.array([np.cos(PI/2 +PI/2), np.sin(PI/2 +PI/2), 0])  * 地球公转速度/1.4,
    ])
    质量 = np.array([
        太阳质量,
        太阳质量,
        太阳质量,
    ])
    G = 6.67e-11 # 6.67×10^-11N·m^2/kg^2 
    N体=len(位置)
    assert N体==len(位置) and N体==len(速度)
    dT =  60*60*24 # 时间间隔

    def pack_y(位置, 速度):
        return np.concatenate((位置.reshape(-1), 速度.reshape(-1)))

    def unpack_y(y):
        位置, 速度  = y.reshape(2, N体, 3)
        return 位置, 速度

    def f(t, y):
        位置, 速度 = unpack_y(y)
        # F = (GMm)/距离矩阵^2
        距离矩阵 = distance_matrix(位置) # 计算距离矩阵
        距离矩阵平方 = 距离矩阵 * 距离矩阵.T
        # 计算Mm矩阵：
        质量矩阵 = repeat_at(tensor=质量, insert_dim=-1, n_times=N体)   
        Mm矩阵 = 质量矩阵 * 质量矩阵.T
        # 计算引力
        引力标量矩阵 = G*Mm矩阵/(距离矩阵平方+1e-10)
        # 对角线元素清零
        for i in range(N体): 引力标量矩阵[i,i] = 0
        # 引力的方向，即位移的方向
        引力方向 = delta_matrix(位置) # 形状为 (N体, N体, 3)
        # 引力方向归一化，乘以引力矩阵（标量）
        引力方向 = 引力方向/(np.linalg.norm(引力方向,axis=-1, keepdims=True)+1e-10)
        引力 = 引力方向*repeat_at(tensor=引力标量矩阵, insert_dim=-1, n_times=3)
        引力合力 = 引力.sum(1)
        # a=F/m
        加速度 = 引力合力/repeat_at(tensor=质量, insert_dim=-1, n_times=3)
        位置导数 = 速度
        速度导数 = 加速度
        y_dot = pack_y(位置导数, 速度导数)
        return y_dot

    r = ode(f).set_integrator('zvode', method='bdf')
    y0 = pack_y(位置, 速度);    r.set_initial_value(y0, 0)
    while r.successful() and r.t < 60*60*24*365*10:
        时间 = int((r.t+dT)/dT)
        位置, 速度 = unpack_y(r.integrate(r.t+dT))
        # if 时间%10==0:
        位置_实部 = 位置.real / 1.496e9
        for body in range(N体):
            
            可视化桥.发送几何体('ball|%d|%s|%.2f'%(body, colors[body], size[body]),  # 填入核心参量： “已声明的形状|几何体之ID标识|颜色|整体大小”
                位置_实部[body, 0], 位置_实部[body, 1], 位置_实部[body, 2], ro_x=0, ro_y=0, ro_z=0, # 6 DOF
                opacity=1, label=labels[body], label_color='white',  track_n_frame=100,  track_tension=0, track_color='white',    # 轨迹的颜色显示，输入js颜色名或者hex值均可
                )
        可视化桥.结束关键帧()
        print('\r %.2f'%(r.t+dT), end='', flush=True)
        
if __name__ == "__main__":
    run()
    time.sleep(10000)