# UHMAP

Developed with Unreal Engine 4

# 安装方法
- 第1步，必须从```源代码```安装虚幻引擎，具体方法见虚幻引擎的官方文档：https://docs.unrealengine.com/4.27/zh-CN/ProductionPipelines/DevelopmentSetup/BuildingUnrealEngine/

- 第2步，克隆本仓库。```git clone https://github.com/binary-husky/unreal-hmp.git```

- 第3步，下载github不能管理的大文件。运行```python Please_Run_This_First_To_Fetch_Big_Files.py```。

- 第4步，```右```击上一步下载的```UHMP.uproject```，选择```switch unreal engine version```，再选择```source build at xxxxx```确认。然后打开生成的```UHMP.sln```，编译即可。

- 最后，双击```UHMP.uproject```进入虚幻引擎编辑器。

注意，第1步和第4步较难，建议参考以下视频（视频中前1分46秒为第1步流程，后面为第4步流程）： https://ageasga-my.sharepoint.com/:v:/g/personal/fuqingxu_yiteam_tech/EawfqsV2jF5Nsv3KF7X1-woBH-VTvELL6FSRX4cIgUboLg?e=Vmp67E



# 项目打包方法
运行一下脚本即可。
- 其中```Render/Server```代表```包含图形渲染/无界面仅计算```，后者一般用于RL训练。
- 其中```Win/linux```代表目标操作系统，注意在windows上编译linux程序需要安装```虚幻引擎交叉编译工具```。
```
python BuildlinuxRender.py
python BuildLinuxServer.py
python BuildWinRender.py
python BuildWinServer.py
```