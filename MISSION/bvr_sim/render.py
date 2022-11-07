import os, sys, math
import numpy as np

class RenderBridge():
    def advanced_render(self, time, win, observer):
        if not hasattr(self, '可视化桥'):
            from VISUALIZE.mcom import mcom
            self.可视化桥 = mcom(path='TEMP/v2d_logger/', draw_mode='Threejs')
            self.可视化桥.初始化3D()
            self.可视化桥.设置样式('sky')   # adjust parameters https://threejs.org/examples/?q=sky#webgl_shaders_sky


            # self.可视化桥.设置样式('skybox6side', 
            #     posx='/wget/snow_textures/posx.jpg',
            #     negx='/wget/snow_textures/negx.jpg',
            #     posy='/wget/snow_textures/negy.jpg',
            #     negy='/wget/snow_textures/posy.jpg',
            #     posz='/wget/snow_textures/posz.jpg',
            #     negz='/wget/snow_textures/negz.jpg',
            # )

            self.可视化桥.设置样式('font', fontPath='/examples/fonts/ttf/FZYTK.TTF', fontLineHeight=1500) # 注意不可以省略参数键值'fontpath=' ！！！
            self.可视化桥.其他几何体之旋转缩放和平移('BOX', 'BoxGeometry(1,1,1)',   0,0,0,  1,1,1, 0,0,0) 
            self.可视化桥.其他几何体之旋转缩放和平移('OCT', 'OctahedronGeometry(1,0)', 0,0,0,  1,1,1, 0,0,0)   # 八面体
            self.可视化桥.其他几何体之旋转缩放和平移('Plane', 'fbx=/examples/files/plane.fbx', -np.pi/2, 0, np.pi/2,  1,1,1, 0,0,0)   # 八面体
            self.可视化桥.上次结果 = ''



        if hasattr(self.可视化桥, '刚刚复位') and self.可视化桥.刚刚复位:
            self.可视化桥.刚刚复位 = False
            if win[self.player_color]: self.可视化桥.上次结果 = ' RL胜利\n'
            else: self.可视化桥.上次结果 = ' RL战败\n'
            self.可视化桥.发送几何体(
                'BOX|%d|%s|10'%(-1, 'White'),  # 填入核心参量： “已声明的形状|几何体的唯一ID标识|颜色|整体大小”
                0, 60, 0,                # 三维位置，3/6dof
                ro_x=0, ro_y=0, ro_z=0, # 欧拉旋转变换，3/6dof
                opacity=0,              # 透明度，1为不透明
                label=self.可视化桥.上次结果,               # 显示标签，空白不显示，用'\n'换行
                label_color='Green',    # 标签颜色
                fade_step = 80
            )
            for _ in range(10): 
                self.可视化桥.空指令()
                self.可视化桥.结束关键帧()
                #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

            self.可视化桥.set_env('clear_everything')

            self.可视化桥.结束关键帧()  
            #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

            self.可视化桥.发送几何体(
                'BOX|%d|%s|10'%(-1, 'White'),  # 填入核心参量： “已声明的形状|几何体的唯一ID标识|颜色|整体大小”
                0, 60, 0,                # 三维位置，3/6dof
                ro_x=0, ro_y=0, ro_z=0, # 欧拉旋转变换，3/6dof
                opacity=0,              # 透明度，1为不透明
                label=self.可视化桥.上次结果,               # 显示标签，空白不显示，用'\n'换行
                label_color='Green',    # 标签颜色
                fade_step = 80
            )
            self.可视化桥.结束关键帧()
            #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-


        # 这里走了个捷径
        red_planes = observer.my_planes
        blue_planes = observer.op_planes
        missiles = observer.ms
        z_scale = 5
        for p in red_planes:
            color_ = 'Red' if p.alive else 'Black'
            size = 3 if '有人机' in p.Name else 1.5
            pitch_ = -np.arctan(np.tan(p.Pitch)*z_scale)
            if (not p.alive) and (hasattr(p, 'previous_alive')) and (not p.previous_alive): continue
            p.previous_alive = p.alive
            self.可视化桥.发送几何体(
                'Plane|%d|%s|%.3f'%(p.ID, color_, size),  # 填入核心参量： “已声明的形状|几何体的唯一ID标识|颜色|整体大小”
                p.X/1e3, p.Y/1e3, p.Z/1e3*z_scale,                # 三维位置，3/6dof
                ro_x=p.Roll, ro_y=pitch_, ro_z=-p.Heading+np.pi/2, ro_order='ZYX',# 欧拉旋转变换，3/6dof
                opacity=1,              # 透明度，1为不透明
                label=''.join([p.Name,'\n剩余载弹 %d'%p.LeftWeapon]),               # 显示标签，空白不显示，用'\n'换行
                label_color='Crimson',    # 标签颜色
                label_offset=np.array([0,4,2]), # 标签与物体之间的相对位置，实验选项，不建议手动指定
                track_n_frame=60,        # 是否显示轨迹（0代表否），轨迹由最新的track_n_frame次位置连接而成
                track_tension=0,      # 轨迹曲线的平滑度，0为不平滑，推荐设置0不平滑
                track_color='Red',    # 轨迹的颜色显示，输入js颜色名或者hex值均可	
                fade_step = 80,
            )

        for p in blue_planes:
            color_ = 'Blue' if p.alive else 'Black'
            size = 3 if '有人机' in p.Name else 1.5
            pitch_ = -np.arctan(np.tan(p.Pitch)*z_scale)
            if (not p.alive) and (hasattr(p, 'previous_alive')) and (not p.previous_alive): continue
            p.previous_alive = p.alive
            self.可视化桥.发送几何体(
                'Plane|%d|%s|%.3f'%(p.ID, color_, size),  # 填入核心参量： “已声明的形状|几何体的唯一ID标识|颜色|整体大小”
                p.X/1e3, p.Y/1e3, p.Z/1e3*z_scale,                # 三维位置，3/6dof
                ro_x=p.Roll, ro_y=pitch_, ro_z=-p.Heading+np.pi/2, ro_order='ZYX',# 欧拉旋转变换，3/6dof
                opacity=1,              # 透明度，1为不透明
                label=''.join([p.Name,'\n剩余载弹 %d'%p.OpLeftWeapon]),               # 显示标签，空白不显示，用'\n'换行
                label_offset=np.array([0,4,2]), # 标签与物体之间的相对位置，实验选项，不建议手动指定
                label_color='DeepSkyBlue',    # 标签颜色
                track_n_frame=60,        # 是否显示轨迹（0代表否），轨迹由最新的track_n_frame次位置连接而成
                track_tension=0,      # 轨迹曲线的平滑度，0为不平滑，推荐设置0不平滑
                track_color='Blue',    # 轨迹的颜色显示，输入js颜色名或者hex值均可
                fade_step = 80,
            )            
        for p in missiles:
            color_ = 'Pink' if p.Identification=='红方' else 'BlueViolet'
            color_ = color_ if p.alive else 'Black'
            pitch_ = -np.arctan(np.tan(p.Pitch)*z_scale)
            if (not p.alive) and (hasattr(p, 'previous_alive')) and (not p.previous_alive): continue
            p.previous_alive = p.alive
            self.可视化桥.发送几何体(
                'BOX|%d|%s|0.5'%(p.ID, color_),  # 填入核心参量： “已声明的形状|几何体的唯一ID标识|颜色|整体大小”
                p.X/1e3, p.Y/1e3, p.Z/1e3*z_scale,                # 三维位置，3/6dof
                ro_x=p.Roll, ro_y=pitch_, ro_z=-p.Heading+np.pi/2, ro_order='ZYX',# 欧拉旋转变换，3/6dof
                opacity=1,              # 透明度，1为不透明
                label='飞行时间%d 距离%.2f'%(p.flying_time, p.distance[-1]) if p.alive else '',               # 显示标签，空白不显示，用'\n'换行
                label_color='Yellow',    # 标签颜色
                # label_offset=np.array([0,2,2]), # 标签与物体之间的相对位置，实验选项，不建议手动指定
                track_n_frame=120,        # 是否显示轨迹（0代表否），轨迹由最新的track_n_frame次位置连接而成
                track_tension=0,      # 轨迹曲线的平滑度，0为不平滑，推荐设置0不平滑
                track_color=color_,    # 轨迹的颜色显示，输入js颜色名或者hex值均可
                fade_step = 50,
            )






        self.可视化桥.结束关键帧()
        return
