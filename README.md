# UHMAP
基于UnrealEngine开发，文档正在完善中。Developed with Unreal Engine, documenting is in process.

此项目处于活跃开发阶段，请打星关注哦。 
** Your encouragement is extremely important to us as researchers, Please ```star``` this Github project. **

Contact us: 
- Qingxu Fu, CASIA, fuqingxu@ia.ac.cn，QQ505030475
- Tianyi Hu, CASIA, hutianyi2021@ia.ac.cn
- Yifan Xu, CASIA, xuyifan2021@ia.ac.cn
- 单位：中科院自动化研究所，yiteam群体智能课题组

# 简介

Unreal-Based Hybrid Multiagent Playground (Unreal-HMAP) 是基于虚幻引擎的新一代多智能体环境仿真环境。
该平台支持多队伍对抗，为第一个（也是目前为止唯一一个）基于虚幻引擎的多智能体+多队伍强化学习环境。
UHMAP面向对抗性异构多智能体强化学习场景。
接口部分采用Python编写，
虚幻引擎部分采用C++处理与Python的通讯接口，其他部分采用蓝图。
项目引入xtensor用于加速C++部分的数学运算。
在科研实验方面：
- 可编译为Headless的纯计算模式（即dedicated server，用于训练）
- 可实现任意倍率的仿真加速，直到跑满CPU
- 可重复性强。排除了UnrealEngine中各种会造成实验不可重复的蝴蝶效应因素
- 支持大规模。与Python端的通讯协议高度优化，避免了随智能体数量增多导致的IO卡顿
- 非常高效，极其节省CPU。Unreal引擎本身的效率远超预想。
- 跨平台。不管是Windows、Linux还是MacOs都能编译Headless模式和渲染模式的客户端
- 可跨OS连接训练中的Headless进程，甚至可以观看训练中的环境.


# 安装方法
- 第1步，必须从```源代码```安装虚幻引擎，具体方法见虚幻引擎的官方文档：https://docs.unrealengine.com/4.27/zh-CN/ProductionPipelines/DevelopmentSetup/BuildingUnrealEngine/

- 第2步，克隆本仓库。```git clone https://github.com/binary-husky/unreal-hmp.git```

- 第3步，下载github不能管理的大文件。运行```python Please_Run_This_First_To_Fetch_Big_Files.py```。

- 第4步，```右```击上一步下载的```UHMP.uproject```，选择```switch unreal engine version```，再选择```source build at xxxxx```确认。然后打开生成的```UHMP.sln```，编译即可。

- 最后，双击```UHMP.uproject```进入虚幻引擎编辑器。

注意，第1步和第4步较难，建议参考以下视频（视频中前1分46秒为第1步流程，后面为第4步流程）： https://ageasga-my.sharepoint.com/:v:/g/personal/fuqingxu_yiteam_tech/EawfqsV2jF5Nsv3KF7X1-woBH-VTvELL6FSRX4cIgUboLg?e=Vmp67E

# 环境设计方法
文档正在完善，简单demo的视频教程见```EnvDesignTutorial.pptx```（需要完成安装步骤3以下载此pptx文件）


# 项目打包方法
运行一下脚本即可。
- 
- 其中```Render/Server```代表```包含图形渲染/无界面仅计算```，后者一般用于RL训练。
- 其中```Win/linux```代表目标操作系统，注意在windows上编译linux程序需要安装```虚幻引擎交叉编译工具```。
```
python BuildlinuxRender.py
python BuildLinuxServer.py
python BuildWinRender.py
python BuildWinServer.py
```
