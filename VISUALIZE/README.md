# Visual Hybrid Multi-Agent Playground (VHMAP 使用说明书)
## 面向场景和特点
面向场景：
- 科研，尤其是多智能体强化学习领域
- 3D演示
- 娱乐

应用特点：
- Python接口简化到极致
- 渲染在客户端，自动插帧，纵享丝滑帧率
- 服务端依赖少
- 占用服务端资源极少
- 基于ThreeJs，支持拖动，支持手机触屏
- 支持透视和投影两种视图的切换
- 支持回放
- 使用zlib压缩数据流，网络带宽需求小

## 安装 
```shell
pip install vhmap
```

## 20行代码-展示VHMAP的简单、丝滑
实现下图，仅需要20行python代码(含初始化)
![](md_imgs/动画x7.gif)
界面功能、操作介绍：
- 鼠标右键平移，左键旋转，滚轮缩放
- 支持触屏，如果你笔记本或手机有触控屏幕
- 左上角显示渲染刷新率
- play fps：每秒播放多少关键帧(小于渲染刷新率，则插帧；大于渲染刷新率，则超出部分无效)
- pause：暂停
- next frame：暂停并切换下一帧
- previous frame：暂停并切换上一帧
- loop to start：播放完所有数据，回到第一帧
- ppt step：以极慢极慢的速度播放一帧，方便录屏，按下后会卡顿几秒
- use orthcam：切换透视视图（物体近大远小）/投影视图（工程制图学过没），
- P.S. 第一次切换到投影视图时，需要用鼠标滚轮放大画面

```python
from VISUALIZE.mcom import mcom
import numpy as np
class TestVhmap():
    def render(self, t):
        if not hasattr(self, '可视化桥'):
            self.可视化桥 = mcom(path='RECYCLE/v2d_logger/', draw_mode='Threejs')
            self.可视化桥.初始化3D()
            self.可视化桥.设置样式('gray')
            self.可视化桥.其他几何体之旋转缩放和平移('box', 'BoxGeometry(1,1,1)',   0,0,0,  1,1,1, 0,0,0) 

        x = np.cos(t); y=np.sin(t); z= np.cos(t)*np.sin(t)  # 此帧的x,y,z坐标
        self.可视化桥.发送几何体(
            'box|2233|Red|0.1',     # 填入 ‘形状|几何体之ID标识|颜色|大小’即可
            x, y, z, ro_x=0, ro_y=0, ro_z=np.sin(t),    # 三维位置+欧拉旋转变换，六自由度
            track_n_frame=20)                           # 显示历史20帧留下的轨迹
        self.可视化桥.结束关键帧()

if __name__ == '__main__':
    x = TestVhmap()
    for step in range(1000): x.render(t=step/np.pi)
    import time; time.sleep(1000)   # 启动后打开输出的url地址即可
# 这是第21行，已经写完了 :joy: 
```

## 50行代码-演示3维N体运动(低精度定步长)
- 代码1详情请见：VISUALIZE/examples/nb.py
运行方法：
```
pip install vhmap

python -m VISUALIZE.examples.nb
```
![](md_imgs/动画9.gif)

## 90行代码-使用dop853求解常微分方程演示三体、N体运动
- 代码2详情请见：VISUALIZE/examples/nb_3body_specials.py
- 代码3详情请见：VISUALIZE/examples/nb_nbody_specials.py
```
pip install vhmap

python -m VISUALIZE.examples.nb_3body_specials
```
![](md_imgs/动画11.gif)

![](md_imgs/动画12-1.gif)

## API-中文

引入
```python
from VISUALIZE.mcom import mcom
```

初始化
```python
可视化桥 = mcom(path='RECYCLE/v2d_logger/', draw_mode='Threejs')
可视化桥.初始化3D()
```

设置样式
```python
可视化桥.设置样式('star')       # 布置星空
可视化桥.设置样式('grid')       # 布置2维网格
可视化桥.设置样式('grid3d')     # 布置3维网格
可视化桥.设置样式('earth')      # 在场景中放一个地球
可视化桥.设置样式('background', color='White') # 注意不可以省略参数键值'color=' ！！！
```

声明几何体
```python
# declare geo 'oct1', init with OctahedronGeometry, then (1)rotate & (2)scale & (3)translate
可视化桥.其他几何体之旋转缩放和平移('oct1', 'OctahedronGeometry(1,0)', 0,0,0,  1,1,1, 0,0,0)   # 八面体
# 需要换成其他几何体，请把'OctahedronGeometry(1,0)'替换，参考网址 https://threejs.org/docs/index.html?q=Geometry
可视化桥.其他几何体之旋转缩放和平移('any_name_you_want', 'TorusGeometry(10,3,16,100)',   0,0,0,  1,1,1, 0,0,0) # 甜甜圈
# declare geo 'ball'
可视化桥.其他几何体之旋转缩放和平移('ball', 'SphereGeometry(1)',   0,0,0,  1,1,1, 0,0,0) # 球体
# declare geo 'box'
可视化桥.其他几何体之旋转缩放和平移('box', 'BoxGeometry(1,1,1)',   0,0,0,  1,1,1, 0,0,0) # 长方体

```

发送几何体，可用颜色（JS颜色，支持Hex颜色）参考 https://www.w3schools.com/colors/colors_names.asp
```python
# 注意不可以省略参数键值
x=1; y=2; z=3
可视化桥.发送几何体(
    'ball|8848|MidnightBlue|0.5',  # 填入核心参量： “已声明的形状|几何体的唯一ID标识|颜色|整体大小”
    x, y, z,                # 三维位置，3/6dof
    ro_x=0, ro_y=0, ro_z=0, # 欧拉旋转变换，3/6dof
    opacity=1,              # 透明度，1为不透明
    label='',               # 显示标签，空白不显示
    label_color='White',    # 标签颜色
    track_n_frame=3,        # 是否显示轨迹（0代表否），轨迹由最新的track_n_frame次位置连接而成
    track_tension=0.1,      # 轨迹曲线的平滑度，0为不平滑，推荐不平滑
    track_color='Green',    # 轨迹的颜色显示，输入js颜色名或者hex值均可
    )
```

发送曲线
```python
# 画一条(0,0,0) -> (1,1,0) -> (2,2,0) -> (3,3,0) 的线
# 注意不可以省略参数键值!!
可视化桥.发送线条(
    'simple|3999|MidnightBlue|0.004', # 填入核心参量： “simple|线条的唯一ID标识|颜色|整体大小”
    x_arr=np.array([0, 1, 2, 3]),   # 曲线的x坐标列表
    y_arr=np.array([0, 1, 2, 3]),   # 曲线的y坐标列表
    z_arr=np.array([0, 0, 0, 0]),   # 曲线的z坐标列表
    tension=0,  # 曲线的平滑度，0为不平滑，推荐不平滑
    opacity=1,  # 透明度，1为不透明，不稳定仍然在测试中
)

# fat 型线条，支持调节宽度、虚线、透明度等，但是不稳定仍然在测试中
# 注意不可以省略参数键值!!
可视化桥.发送线条(
    'fat|3999|MidnightBlue|0.004', # 填入核心参量： “fat|线条的唯一ID标识|颜色|整体大小”
    x_arr=np.array([0, 1, 2, 3]),   # 曲线的x坐标列表
    y_arr=np.array([0, 1, 2, 3]),   # 曲线的y坐标列表
    z_arr=np.array([0, 0, 0, 0]),   # 曲线的z坐标列表
    dashScale=20,   # 此数越大，单位长度上的虚线切割越多
    dashSize=1,     # 虚线切割之实线
    gapSize=1,      # 虚线切割之实线间隔
    tension=0,      # 曲线的平滑度，0为不平滑，推荐不平滑
    opacity=1,      # 透明度，1为不透明
)
```

发射光束（从几何体src到几何体dst）
```python
# 注意不可以省略参数键值!!
可视化桥.发射光束(
    'beam',         # 有 beam 和 lightning 两种选择
    src=index_ID,   # 发射者的几何体的唯一ID标识
    dst=index_ID2,  # 接收者的几何体的唯一ID标识
    dur=0.5,        # 光束持续时间，单位秒，绝对时间，不受播放fps的影响
    size=0.03,      # 光束粗细
    color='DeepSkyBlue' # 光束颜色
)

```


终结这一帧（并开始下一帧）
```python
self.可视化桥.结束关键帧()
```

