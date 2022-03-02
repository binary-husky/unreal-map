"""
@FileName：demo_agent.py
@Description：
@Author：
@Time：2021/6/17 上午9:21
@Department：AIStudio研发部
@Copyright：©2011-2021 北京华如科技股份有限公司
"""
from typing import List
from agent.agent import Agent
from  env.env_cmd import CmdEnv
from utils.utils_math import TSVector3
import  copy
import  random

"""
选手需要重写继承自基类Agent中的step(self,sim_time, obs_red, **kwargs)去实现自己的策略逻辑，注意，这个方法每个步长都会被调用
"""

class DemoAgent(Agent):
    """
         自定义的Demo智能体
     @Examples:
         添加使用示例
         >>> 填写使用说明
         ··· 填写简单代码示例
     """
    def __init__(self, name, config):
        """
        初始化信息
        :param name:阵营名称
        :param config:阵营配置信息
        """
        super(DemoAgent, self).__init__(name, config["side"])
        self._init() #调用用以定义一些下文所需变量

    def _init(self):
        """对下文中使用到的变量进行定义和初始化"""
        self.my_uvas_infos = []         #该变量用以保存己方所有无人机的态势信息
        self.my_manned_info = []        #该变量用以保存己方有人机的态势信息
        self.my_allplane_infos = []     #该变量用以保存己方所有飞机的态势信息
        self.enemy_uvas_infos = []      #该变量用以保存敌方所有无人机的态势信息
        self.enemy_manned_info = []     #该变量用以保存敌方有人机的态势信息
        self.enemy_allplane_infos = []  #该变量用以保存敌方所有飞机的态势信息
        self.enemy_missile_infos = []   #该变量用以保存敌方导弹的态势信息

        self.attack_handle_enemy = {}   #该变量用于记录已经去攻击的飞机

    def reset(self, **kwargs):
        """当引擎重置会调用,选手需要重写此方法,来实现重置的逻辑"""
        self.attack_handle_enemy.clear() #重置已经去攻击的飞机的信息
        pass

    def step(self,sim_time, obs_side, **kwargs) -> List[dict]:
        """ 步长处理
        此方法继承自基类中的step(self,sim_time, obs_red, **kwargs)
        选手通过重写此方法，去实现自己的策略逻辑，注意，这个方法每个步长都会被调用
        :param sim_time: 当前想定已经运行时间
        :param obs_side:当前方所有的态势信息信息，包含了所有的当前方所需信息以及探测到的敌方信息
				obs_side 包含 platforminfos，trackinfos，missileinfos三项Key值
				obs_side['platforminfos'] 为己方飞机信息，字典列表，其中包含的字典信息如下（以下为Key值）

                        Name 			# 飞机的名称
                        Identification 	# 飞机的标识符（表示飞机是红方还是蓝方）
                        ID 				# 飞机的ID（表示飞机的唯一编号）
                        Type 			# 飞机的类型（表示飞机的类型，其中有人机类型为 1，无人机类型为2）
						Availability 	# 飞机的可用性（表示飞机的可用性，范围为0到1,为1表示飞机存活，0表示飞机阵亡）
						X 				# 飞机的当前X坐标（表示飞机的X坐标）
						Y 				# 飞机的当前Y坐标（表示飞机的Y坐标）
						Lon 			# 飞机的当前所在经度（表示飞机的所在经度）
						Lat 			# 飞机的当前所在纬度（表示飞机的所在纬度）
						Alt 			# 飞机的当前所在高度（表示飞机的所在高度）
						Heading 		# 飞机的当前朝向角度（飞机的当前朝向,范围为-180°到180° 朝向0°表示正北,逆时针方向旋转为正数0°到180°，顺时针方向为正数0°到-180°）
						Pitch 			# 飞机的当前俯仰角度（飞机的当前俯仰,范围为-90°到90°,朝向高处为正,低处为负）
						Roll 			# 飞机的当前滚转角度（飞机的当前滚转,范围为-180°到180° ）
						Speed 			# 飞机的当前速度（飞机的当前速度）
						CurTime 		# 当前时间（当前时间）
						AccMag 			# 飞机的指令加速度（飞机的指令加速度）
						NormalG 		# 飞机的指令过载（飞机的指令过载）
						IsLocked 		# 飞机是否被敌方导弹锁定（飞机是否被敌方导弹锁定）
						Status 			# 飞机的当前状态（飞机的当前状态）
						LeftWeapon 		# 飞机的当前剩余导弹数（飞机的当前剩余导弹数）

				obs_side['trackinfos'] 为敌方飞机信息，字典列表，其中包含的字典信息如下（以下为Key值）

						Name 			# 敌方飞机的名称
						Identification 	# 敌方飞机的标识符（表示敌方飞机是红方还是蓝方）
						ID 				# 敌方飞机的ID（表示飞机的唯一编号）
						Type 			# 敌方飞机的类型（表示飞机的类型，其中有人机类型为 1，无人机类型为2）
						Availability 	# 敌方飞机的可用性（表示飞机的可用性，范围为0到1,为1表示飞机存活，0表示飞机阵亡）
						X 				# 敌方飞机的当前X坐标（表示飞机的X坐标）
						Y 				# 敌方飞机的当前Y坐标（表示飞机的Y坐标）
						Lon 			# 敌方飞机的当前所在经度（表示飞机的所在经度）
						Lat 			# 敌方飞机的当前所在纬度（表示飞机的所在纬度）
						Alt 			# 敌方飞机的当前所在高度（表示飞机的所在高度）
						Heading 		# 敌方飞机的当前朝向角度（飞机的当前朝向,范围为-180°到180° 朝向0°表示正北,逆时针方向旋转为正数0°到180°，顺时针方向为正数0°到-180°）
						Pitch 			# 敌方飞机的当前俯仰角度（飞机的当前俯仰,范围为-90°到90°,朝向高处为正,低处为负）
						Roll 			# 敌方飞机的当前滚转角度（飞机的当前滚转,范围为-180°到180° ）
						Speed 			# 敌方飞机的当前速度（飞机的当前速度）
						CurTime 		# 当前时间（当前时间）
						IsLocked 		# 敌方飞机是否被敌方导弹锁定（飞机是否被己方导弹锁定）

				obs_side['missileinfos']为空间中所有未爆炸的双方导弹信息，字典列表，其中包含的字典信息如下（以下为Key值）

						Name 			# 导弹的名称
						Identification 	# 导弹的标识符（表示导弹是红方还是蓝方）
						ID 				# 导弹的ID（表示导弹的唯一编号）
						Type 			# 导弹的类型（表示导弹的类型，其中导弹类型为 3）
						Availability 	# 导弹的可用性（表示导弹的可用性，范围为0到1,为1表示飞机存活，0表示导弹已爆炸）
						X 				# 导弹的当前X坐标（表示导弹的X坐标）
						Y 				# 导弹的当前Y坐标（表示导弹的Y坐标）
						Lon 			# 导弹的当前所在经度（表示导弹的所在经度）
						Lat 			# 导弹的当前所在纬度（表示导弹的所在纬度）
						Alt 			# 导弹的当前所在高度（表示导弹的所在高度）
						Heading 		# 导弹的当前朝向角度（导弹的当前朝向,范围为-180°到180° 朝向0°表示正北,逆时针方向旋转为正数0°到180°，顺时针方向为正数0°到-180°）
						Pitch 			# 导弹的当前俯仰角度（导弹的当前俯仰,范围为-90°到90°,朝向高处为正,低处为负）
						Roll 			# 导弹的当前滚转角度（导弹的当前滚转,范围为-180°到180° ）
						Speed 			# 导弹的当前速度（导弹的当前速度）
						CurTime 		# 当前时间（当前时间）
						LauncherID 		# 导弹的发射者ID（敌方导弹的发射者ID）
						EngageTargetID 	# 导弹攻击目标的ID（敌方导弹攻击目标的ID）

        :param kwargs:保留的变量
        :return: 决策完毕的任务指令列表
        """
        cmd_list = []   # 此变量为 保存所有的决策完毕任务指令列表

        self.process_decision(sim_time, obs_side, cmd_list) # 调用决策函数进行决策判断

        return cmd_list # 返回决策完毕的任务指令列表

    def process_decision(self, sim_time, obs_side, cmd_list):
        """处理决策
        :param sim_time: 当前想定已经运行时间
        :param obs_side: 当前方所有的Observation信息，包含了所有的当前方所需信息以及探测到的敌方信息
        :param cmd_list保存所有的决策完毕任务指令列表
				可用指令有六种
					1.初始化实体指令 （初始化实体的信息，注意该指令只能在开始的前3秒有效）
						make_entityinitinfo(receiver: int,x: float,y: float,z: float,init_speed: float,init_heading: float)
						参数含义为
							:param receiver:飞机的唯一编号，即上文中飞机的ID
							:param x: 初始位置为战场x坐标
							:param y: 初始位置为战场y坐标
							:param z: 初始位置为战场z坐标
							:param init_speed: 初始速度(单位：米/秒，有人机取值范围：[150,400]，无人机取值范围：[100,300])
							:param init_heading: 初始朝向(单位：度，取值范围[0,360]，与正北方向的夹角)
					2.航线巡逻控制指令（令飞机沿航线机动）
						make_linepatrolparam(receiver: int,coord_list: List[dict],cmd_speed: float,cmd_accmag: float,cmd_g: float)
						参数含义为
							:param receiver: 飞机的唯一编号，即飞机的ID
							:param coord_list: 路径点坐标列表 -> [{"x": 500, "y": 400, "z": 2000}, {"x": 600, "y": 500, "z": 3000}]
											   区域x，y不得超过作战区域,有人机高度限制[2000,15000]，无人机高度限制[2000,10000]
							:param cmd_speed: 指令速度
							:param cmd_accmag: 指令加速度
							:param cmd_g: 指令过载
					3.区域巡逻控制指令	（令飞机沿区域巡逻）
						make_areapatrolparam(receiver: int,x: float,y: float,z: float,area_length: float,area_width: float,cmd_speed: float,cmd_accmag: float,cmd_g: float)
						    :param receiver: 飞机的唯一编号，即飞机的ID
							:param x: 区域中心坐标x坐标
							:param y: 区域中心坐标y坐标
							:param z: 区域中心坐标z坐标
							:param area_length: 区域长
							:param area_width: 区域宽
							:param cmd_speed: 指令速度
							:param cmd_accmag: 指令加速度
							:param cmd_g: 指令过载
					4.机动参数调整控制指令（调整飞机的速度、加速度和过载）
						make_motioncmdparam(receiver: int, update_motiontype: int,cmd_speed: float,cmd_accmag: float,cmd_g: float)
						    :param receiver: 飞机的唯一编号，即飞机的ID
							:param update_motiontype: 调整机动参数,其中 1为设置指令速度，2为设置指令加速度，3为设置指令速度和指令加速度
							:param cmd_speed: 指令速度
							:param cmd_accmag: 指令加速度
							:param cmd_g: 指令过载
					5.跟随目标指令 （令飞机跟随其他飞机）
						make_followparam(receiver: int,tgt_id: int,cmd_speed: float,cmd_accmag: float,cmd_g: float)
							:param receiver:飞机的唯一编号，即飞机的ID
							:param tgt_id: 目标ID,友方敌方均可
							:param cmd_speed: 指令速度
							:param cmd_accmag: 指令加速度
							:param cmd_g: 指令过载
					6.打击目标指令（令飞机使用导弹攻击其他飞机）
						make_attackparam(receiver: int,tgt_id: int,fire_range: float )
							:param receiver:飞机的唯一编号，即飞机的ID
							:param tgt_id: 目标ID
							:param fire_range: 开火范围，最大探测范围的百分比，取值范围[0, 1]

        """
        self.process_observation(obs_side)  # 获取态势信息,对态势信息进行处理

        if sim_time == 1:  # 当作战时间为1s时,初始化实体位置,注意,初始化位置的指令只能在前三秒内才会被执行
            self.init_pos(cmd_list) # 将实体放置到合适的初始化位置上

        if sim_time == 2: # 当作战时间为2s时,开始进行任务开始,并保存任务指令;
            self.mission_start(sim_time, cmd_list)# 调用任务判断以及处理，进入决策阶段

        if sim_time > 10:  # 当作战时间大于10s时,开始进行任务控制,并保存任务指令;
            self.process_attack(sim_time, cmd_list) # 处理攻击，己方使用导弹打击敌方飞机
            self.process_move(sim_time, cmd_list)   # 处理机动，己方如何机动

    def process_observation(self, obs_side):
        """
        初始化飞机态势信息
        :param obs_red: 当前方所有的态势信息信息，包含了所有的当前方所需信息以及探测到的敌方信息
        """
        my_entity_infos = obs_side['platforminfos'] # 拿到己方阵营有人机、无人机在内的所有飞机信息
        if len(my_entity_infos) < 1:
            return
        my_manned_info = [] # 用以保存当前己方有人机信息
        my_uvas_infos = []  # 用以保存当前己方无人机信息
        my_allplane_infos = []    # 用以保存当前己方所有飞机信息
        for uvas_info in my_entity_infos:
            if uvas_info['ID'] != 0 and uvas_info['Availability'] > 0.0001: # 判断飞机是否可用 飞机的ID即为飞机的唯一编号 飞机的Availability为飞机当前生命值
                uvas_info["Z"] = uvas_info["Alt"]     # 飞机的 Alt 即为飞机的当前高度
                if uvas_info['Type'] == 1:           # 所有类型为 1 的飞机是 有人机
                    my_manned_info.append(uvas_info) # 将有人机保存下来 一般情况，每方有人机只有1架
                if uvas_info['Type'] == 2:           # 所有类型为 2 的飞机是 无人机
                    my_uvas_infos.append(uvas_info)  # 将无人机保存下来 一般情况，每方无人机只有4架
                my_allplane_infos.append(uvas_info)        # 将己方所有飞机信息保存下来 一般情况，每方飞机实体总共5架

        if len(my_manned_info) < 1:       #  判断己方有人机是否被摧毁
            return

        enemy_entity_infos = obs_side['trackinfos']   # 拿到敌方阵营的飞机信息,包括敌方有人机、无人机在内的所有飞机信息
        enemy_manned_info = []  # 用以保存当前敌方有人机信息
        enemy_uvas_infos = []   # 用以保存当前敌方无人机信息
        enemy_allplane_infos = []     # 用以保存当前敌方所有飞机信息
        for uvas_info in enemy_entity_infos:
            if uvas_info['ID'] != 0 and uvas_info['Availability'] > 0.0001:  # 判断飞机是否可用 飞机的ID即为飞机的唯一编号 飞机的Availability为飞机当前生命值
                uvas_info['Z'] = uvas_info['Alt']         # 飞机的 Alt 即为飞机的当前高度
                if uvas_info['Type'] == 1:               # 所有类型为 1 的飞机是 有人机
                    enemy_manned_info.append(uvas_info)  # 将有人机保存下来 一般情况，每方有人机只有1架
                if uvas_info['Type'] == 2:               # 所有类型为 2 的飞机是 无人机
                    enemy_uvas_infos.append(uvas_info)   # 将无人机保存下来 一般情况，每方无人机只有4架
                enemy_allplane_infos.append(uvas_info)         # 将己方所有飞机信息保存下来 一般情况，每方飞机实体总共5架

        my_allplane_maps = {}
        for input_entity in my_allplane_infos:
            my_allplane_maps[int(input_entity['ID'])] = input_entity

        missile_infos = obs_side['missileinfos']  # 拿到空间中已发射且尚未爆炸的导弹信息
        enemy_missile_infos = [] #  用以保存敌方已发射且尚未爆炸的导弹信息
        for missile_info in missile_infos:
            if missile_info['LauncherID'] in my_allplane_maps:#  判断导弹是否为己方导弹 导弹的LauncherID即为导弹的发射者
                continue
            if (missile_info['ID'] != 0 and missile_info['Availability'] > 0.0001): # 判断导弹是否可用 导弹的ID即为导弹的唯一编号 导弹的Availability为导弹当前生命值
                missile_info["Z"] = missile_info["Alt"]     # 导弹的 Alt 即为导弹的当前高度
                enemy_missile_infos.append(missile_info)    # 保存敌方已发射且尚未爆炸的导弹信息

        self.my_uvas_infos = my_uvas_infos              # 保存当前己方无人机信息
        self.my_manned_info = my_manned_info            # 保存当前己方有人机信息
        self.my_allplane_infos = my_allplane_infos            # 保存当前己方所有飞机信息
        self.enemy_uvas_infos = enemy_uvas_infos        # 保存当前敌方无人机信息
        self.enemy_manned_info = enemy_manned_info      # 保存当前敌方有人机信息
        self.enemy_allplane_infos = enemy_allplane_infos      # 保存当前敌方所有飞机信息
        self.enemy_missile_infos = enemy_missile_infos  # 保存敌方已发射且尚未爆炸的导弹信息

    def init_pos(self, cmd_list):
        """
        初始化飞机部署位置
        :param cmd_list:所有的决策完毕任务指令列表
        """
        leader_original_pos = {}    # 用以初始化当前方的位置
        if self.name == "red":
            leader_original_pos = {"X": -130000, "Y": -135000, "Z": 9500}
        else :
            leader_original_pos = {"X": 130000, "Y": 135000, "Z": 9500}

        interval_distance = 5000   # 间隔 5000米排列
        for leader in self.my_manned_info: # 为己方有人机设置初始位置
            # CmdEnv.make_entityinitinfo 指令，可以将 飞机实体置于指定位置点，参数依次为 实体ID，X坐标，Y坐标，Z坐标，初始速度，初始朝向
            cmd_list.append(CmdEnv.make_entityinitinfo(leader['ID'], leader_original_pos['X'], leader_original_pos['Y'], leader_original_pos['Z'],400, 45))

        #己方无人机在有人机的y轴上分别以9500的间距进行部署
        sub_index = 0  # 编号 用以在有人机左右位置一次排序位置点
        for sub in self.my_uvas_infos: # 为己方每个无人机设置初始位置
            sub_pos = copy.deepcopy(leader_original_pos)  # 深拷贝有人机的位置点
            if sub_index & 1 == 0: # 将当前编号放在有人机的一侧
                # CmdEnv.make_entityinitinfo 指令，可以将 飞机实体置于指定位置点，参数依次为 实体ID，X坐标，Y坐标，Z坐标，初始速度，初始朝向
                cmd_list.append(CmdEnv.make_entityinitinfo(sub['ID'], sub_pos['X'], sub_pos['Y'] + interval_distance, sub_pos['Z'],300 * 0.6, 45))
            else:                   # 将当前编号放在有人机的另一侧
                # CmdEnv.make_entityinitinfo 指令，可以将 飞机实体置于指定位置点，参数依次为 实体ID，X坐标，Y坐标，Z坐标，初始速度，初始朝向
                cmd_list.append(CmdEnv.make_entityinitinfo(sub['ID'], sub_pos['X'], sub_pos['Y'] - interval_distance, sub_pos['Z'],300 * 0.6, 45))
                interval_distance *= 2 # 编号翻倍
            sub_index += 1 # 编号自增

    def process_attack(self, sim_time, cmd_list):
        """
        处理攻击，根据敌方信息进行攻击
        :param sim_time: 当前想定已经运行时间
        :param cmd_list保存所有的决策完毕任务指令列表
        """
        for plane_info in self.my_allplane_infos:  # 遍历所有的己方飞机，选择离己方最近距离的飞机进行攻击
            if plane_info['LeftWeapon'] == 0: # 判断飞机是否还有导弹  LeftWeapon 飞机的导弹数量
                continue
            close_target = self.select_target(plane_info,self.enemy_allplane_infos) # 选取一个里当前己方飞机最近的敌方飞机
            if len(close_target) < 1: # 如果没有最近的敌方飞机
                continue
            if close_target['close_distance'] <= 80000: #当最近的目标离己方飞机的间距小于60000时
                # CmdEnv.make_attackparam 指令，可以将令飞机攻击指定实体，参数依次为 实体ID，目标ID ，最大射程的百分比（0-1，达到这个距离比就会打击）
                cmd_list.append(CmdEnv.make_attackparam(plane_info['ID'], close_target['enemy_data']['ID'], 1))
                self.attack_handle_enemy[int(plane_info['ID'])]=close_target['enemy_data']['ID']#保存已经攻击过的目标

    def mission_start(self, sim_time, cmd_list):
        """
         任务开始，根据敌方信息朝敌方有人机前进
         :param sim_time: 当前想定已经运行时间
         :param cmd_list保存所有的决策完毕任务指令列表
         """
        for plane_info in self.my_uvas_infos:  # 遍历所有的己方无人飞机
            if len(self.enemy_manned_info) > 0:
                enemy_leader = self.enemy_manned_info[0]  # 拿到敌方有人机信息
                data = self.get_move_data(plane_info)  # 根据拿到飞机获取对应的机动数据
                leader_fire_route_list = [{"X": enemy_leader['X'], "Y": enemy_leader['Y'], "Z": data['area_max_alt']}, ]#获取一个路径点,将敌方有人机的所在点设置为目标点
                # EnvCmd.make_linepatrolparam 指令，可以令飞机沿给定路线机动，参数依次为 飞机ID，路线，速度，加速度，超载（与飞机转向角度有关）
                cmd_list.append(CmdEnv.make_linepatrolparam(plane_info['ID'], leader_fire_route_list, data['move_max_speed'],data['move_max_acc'],data["move_max_g"]));

    def process_move(self, sim_time, cmd_list):
        """
        机动 令己方飞机朝敌方有人机机动
        :param sim_time: 当前想定已经运行时间
        :param cmd_list保存所有的决策完毕任务指令列表
        """
        for plane in self.my_allplane_infos:# 拿到当前有人机信息
            if plane['ID'] not in self.attack_handle_enemy:# 如果当前飞机机不处于等待攻击状态
                data = self.get_move_data(plane) # 根据拿到飞机获取对应的机动数据
                if len(self.enemy_manned_info) > 0:
                    enemy_leader = self.enemy_manned_info[0]#拿到敌方有人机信息
                    if sim_time % 10 == 0:
                        speed = random.randint(int(data['move_min_speed']),int(data['move_max_speed']))
                        # CmdEnv.make_motioncmdparam 指令，可以将修改当前飞机的速度和超载，参数依次为 飞机ID，调整机动参数（1为调整速度，2调整加速度，3调整速度和加速度） ，速度，加速度，超载（与飞机转向角度有关）
                        cmd_list.append(CmdEnv.make_motioncmdparam(plane['ID'], 1, speed, data['move_max_acc'], data['move_max_g']))
                    elif plane['Type'] == 1 and sim_time > 300:#　跟随对方有人机
                        # CmdEnv.make_followparam 指令，可以令飞机跟随另一架飞机（敌我），参数依次为 飞机ID，跟随实体ID ，跟随速度，跟随加速度，跟随超载（与飞机转向角度有关）
                        cmd_list.append(CmdEnv.make_followparam(plane['ID'], enemy_leader['ID'], data['move_max_speed'], data['move_max_acc'], data['move_max_g']))
                    elif plane['Type'] == 2 :#　跟随对方有人机
                        # CmdEnv.make_followparam 指令，可以令飞机跟随另一架飞机（敌我），参数依次为 飞机ID，跟随实体ID ，跟随速度，跟随加速度，跟随超载（与飞机转向角度有关）
                        cmd_list.append(CmdEnv.make_followparam(plane['ID'], enemy_leader['ID'], data['move_max_speed'], data['move_max_acc'], data['move_max_g']))
                else:#如果对方有人机阵亡
                    data = self.get_move_data(plane)  # 根据拿到飞机获取对应的机动数据
                    # EnvCmd.make_areapatrolparam 指令，可以令飞机绕给定点巡逻，参数依次为 飞机ID，给定点X坐标，给定点Y坐标，给定点Z坐标，给定区域的长度，给定区域的宽度，速度，加速度，超载（与飞机转向角度有关）
                    cmd_list.append(
                        CmdEnv.make_areapatrolparam(plane['ID'], plane['X'], plane['Y'], data['area_max_alt'], 200, 100, data['move_max_speed'], data['move_max_acc'], data['move_max_g']))

    def select_target(self,own_plane,enemy_list)->{}:
        """
        选择离己方飞机最近的敌方飞机
        :param own_plane:己方飞机信息
        :param enemy_list:敌方飞机信息列表
        :return:最近的敌方飞机信息
        """
        min_distance = 450000 # 超过此距离的敌方飞机不在考虑之内
        close_enemy = {}
        for enemy in enemy_list:
            distance = TSVector3.distance(own_plane,enemy)# 计算空间两点距离
            if distance < min_distance: # 计算出一个离得最近的敌方实体
                min_distance = distance
                close_enemy['close_distance'] = min_distance
                close_enemy['enemy_data'] = enemy
        return close_enemy # 返回最近的敌方飞机信息

    def get_plane_by_id(self, planeID) -> {}:
        """
        根据飞机ID获取己方飞机态势信息
        :param planeID:飞机ID
        :return:己方飞机态势信息
        """
        for plane in self.my_allplane_infos:
            if plane['ID']== planeID:
                return plane
        return None # 己方飞机态势信息

    def get_move_data(self, plane) -> {}:
        """
        获取己方飞机对应的机动数据
        :param plane:己方飞机态势信息
        :return:飞机获取对应的机动数据
        """
        data = {} # 保存己方机动数据
        if plane['Type'] == 1:  # 所有类型为 1 的飞机是 有人机
            data['move_min_speed'] = 150     # 当前类型飞机的最小速度
            data['move_max_speed'] = 400     # 当前类型飞机的最大速度
            data['move_max_acc'] = 1         # 当前类型飞机的最大加速度
            data['move_max_g'] = 6           # 当前类型飞机的最大超载
            data['area_max_alt'] = 14000     # 当前类型飞机的最大高度
            data['attack_range'] = 1         # 当前类型飞机的最大导弹射程百分比
            data['launch_range'] = 80000     # 当前类型飞机的最大雷达探测范围
        else:                # 所有类型为 2 的飞机是 无人机
            data['move_min_speed'] = 100     # 当前类型飞机的最小速度
            data['move_max_speed'] = 300     # 当前类型飞机的最大速度
            data['move_max_acc'] = 2         # 当前类型飞机的最大加速度
            data['move_max_g'] = 12          # 当前类型飞机的最大超载
            data['area_max_alt'] = 10000     # 当前类型飞机的最大高度
            data['attack_range'] = 1         # 当前类型飞机的最大导弹射程百分比
            data['launch_range'] = 60000     # 当前类型飞机的最大雷达探测范围
        return data # 保存己方机动数据
