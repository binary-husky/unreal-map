import copy
import random
import numpy as np
import datetime
import time
import math

#态势评估模块
#接口输入：全局状态信息，包括进攻方无人车各种状态，防守方无人车各种状态以及地图状态

class Evaluation_module():
	def __init__(self, critical_points=[[-3000, 700, 0], [700, -3300, 0]]):
		self.R0 = 750 # 距离态势缩放因子
		self.V0 = 0.1  # 速度态势缩放因子
		self.phi0 = 1   # 俯仰角态势系数
		self.psi0 = 0  # 偏航角态势系数
		self.ammo0 = 5  # 载荷态势缩放因子(增函数用)
		self.heal0 = 5  # 血量态势缩放因子(增函数用)
		self.AMMO0 = 5  # 载荷态势缩放因子(减函数用)
		self.HEAL0 = 5  # 血量态势缩放因子(减函数用)

		# 已知的环境信息
		self.critical_points = critical_points  # 夺控点位置
	
	# 计算相对速度态势时使用的增函数，输出区间为(0,1)
	def SigmoidTen(self, x, c):
		y = np.exp(-x/c)
		return 1/(1+10*y)
	
	# 计算停留时间态势时使用的增函数，输出区间为[0,0.9)
	def SigmoidNine(self, x, c):
		y = np.exp(-x/c)
		return 1/(1+9*y) - 0.1
	
	# 计算无人车相对坐标点的态势，以备最佳点规划使用，输入为点的坐标和此无人车的信息
	# 在进行最佳点规划时，会计算待打击目标附近的几个敌方无人车的态势之和，以此来计算最佳点
	def UAV2Point(self, p_position, position, velocity, phi, psi, ammo, health):
		# 相对距离威胁
		p_position = np.array(p_position)
		position = np.array(position)
		velocity = np.array(velocity)
		r = p_position - position
		dist = np.sqrt(np.sum(np.square(r)))
		Sr = np.exp(-dist/self.R0)
		# 相对速度威胁
		V = np.dot(r, velocity) / dist  # 求速度在连线朝向上的投影
		Sv = self.SigmoidTen(V, self.V0)
		# 俯仰角威胁
		Sphi = np.exp(-np.abs(phi - self.phi0))
		# 偏航角威胁
		Spsi = np.exp(-np.abs(psi - self.psi0))
		# 载荷威胁（增函数）载荷为0时威胁为0
		# Sammo = self.SigmoidNine(ammo, self.ammo0)
		Sammo = 1
		# 强健度威胁（增函数） 血量为0时威胁为0
		Sheal = self.SigmoidNine(health, self.heal0)

		# 总态势计算 (系数之和不一定为1，每个系数直接在此处修改）
		# 载荷威胁和强健度威胁此处用乘法，算法需要 [在打击范围内] 寻找总态势最小的点作为坐标点
		S_sum = (0.6 * Sr + 0.2 * Sv + 0.2 * Sphi + 0.0 * Spsi) * Sammo * Sheal
		return S_sum

	# 计算无人车相对无人车(智能体)的态势（威胁），以作为选取打击对象的依据（选取威胁大的但是优势低的）
	# 其中a_position等表示智能体无人车的参数，即计算态势时考虑的主体的参数
	def UAV2UAV(self, identity, a_position,a_ammo,a_health, position,velocity, phi, psi, ammo, health):
		a_position = np.array(a_position)
		position = np.array(position)
		velocity = np.array(velocity)
		# 能力比例系数，如我方无人车面对敌方无人车时为1.5, 敌方无人车面对我方无人车时为0.67
		# 己方载荷及健康态势计算（增函数）
		# Mammo = self.SigmoidNine(a_ammo, self.ammo0)
		Mammo = 1
		Mhealth = self.SigmoidNine(a_health, self.heal0)
		# 对方载荷及健康优势计算（减函数）
		# Sammo = np.exp(-ammo / self.AMMO0)
		Sammo = 1
		Shealth = np.exp(-health / self.HEAL0)
		# 进攻方优势计算(选择优势最大的进行打击，若相同则选择距离更近的进行打击）
		if identity == "offensive":
			# 相对距离威胁
			r = a_position - position
			dist = np.sqrt(np.sum(np.square(r)))
			Sr = np.exp(-dist / self.R0)
			S_offensive = 10 * Mammo*Mhealth * Sammo*Shealth * Sr  # 乘了系数10以致于S不过分小
			return S_offensive
		# 防守方优势计算（优先打击距离夺控点近的无人车）
		if identity == "defensive":
			Sr_temp = 0
			for critical_point in self.critical_points:
				r = critical_point - position
				dist = np.sqrt(np.sum(np.square(r)))
				Sr = np.exp(-dist / self.R0)
				if Sr >= Sr_temp:
					Sr_temp = Sr
			S_defensive = 10 * Mammo * Mhealth * Sammo * Shealth * Sr_temp
			return S_defensive

	# 计算无人机相对夺控点的态势，以此作为防守方是否进行驱离及选择谁对谁进行驱离的依据
	def Drone2Point(self, p_position,p_ts, position, velocity):
		# 相对距离威胁
		p_position = np.array(p_position)
		position = np.array(position)
		velocity = np.array(velocity)
		r = p_position - position
		dist = np.sqrt(np.sum(np.square(r)))
		Spr = np.exp(-dist / 1)   # 此处的放缩系数采用与无人机参数相关的
		# 相对速度威胁
		V = np.dot(r, velocity) / dist  # 求速度在连线朝向上的投影
		Spv = self.SigmoidTen(V, 0.2)   # 此处的放缩系数采用与无人机参数相关的
		# 停留时间威胁
		Spt = self.SigmoidNine(p_ts, 0.5)  # 不能接受无人机停留3秒及以上
		# 计算综合态势
		Sp = Spt + 0.6 * Spr + 0.2 * Spv  # 建议驱离阈值: Sp >= 0.5
		print(Sp)
		return Sp
	
	def UAV2Point_id(self, attacker_dict, key_point):
		# 相对距离威胁
		#进攻方无人车信息
		ally_agent_pos = [attacker_dict['X'], attacker_dict['Y'], attacker_dict['Z']]
		ally_agent_blood = attacker_dict['blood']
		ally_agent_velocityx = attacker_dict['vx']
		ally_agent_velocityy = attacker_dict['vy']
		ally_agent_ammo = attacker_dict['ammo']
		ally_agent_velocity = [ally_agent_velocityx, ally_agent_velocityy, 0]
   
		p_position = np.array(key_point)
		position = np.array(ally_agent_pos)
		velocity = np.array(ally_agent_velocity)
		r = p_position - position
		
		phi = math.degrees(math.atan2((ally_agent_pos[0] - key_point[0]), (ally_agent_pos[1] - key_point[1])))
		ammo = ally_agent_ammo
		health = ally_agent_blood
		# 相对速度威胁
		dist = np.sqrt(np.sum(np.square(r)))
		Sr = np.exp(-dist/self.R0)
		# V = np.dot(r, velocity) / dist  # 求速度在连线朝向上的投影
		#Sv = self.SigmoidTen(V, self.V0)
		# 偏航角威胁
		# Sphi = np.exp(-np.abs(phi - self.phi0))
		# 俯仰角威胁
		# Spsi = np.exp(-np.abs(psi - self.psi0))
		# 载荷威胁（增函数）载荷为0时威胁为0
		# Sammo = self.SigmoidNine(ammo, self.ammo0)
		Sammo = 1
		# 强健度威胁（增函数） 血量为0时威胁为0
		Sheal = self.SigmoidNine(health, self.heal0)

		# 总态势计算 (系数之和不一定为1，每个系数直接在此处修改）
		# 载荷威胁和强健度威胁此处用乘法，算法需要 [在打击范围内] 寻找总态势最小的点作为坐标点
		# S_sum = (0.6 * Sr + 0.2 * Sv + 0.2 * Sphi + 0.0 * Spsi) * Sammo * Sheal
		# S_sum = (0.6 * Sr + 0.2 * Sv + 0.2 * Sphi) * Sammo * Sheal
		S_sum = Sr * Sammo * Sheal
		return S_sum

	def UAV2UAV_id(self, identity, attacker_dict, defender_dict):
		#进攻方无人车信息
		ally_agent_pos = [attacker_dict['X'], attacker_dict['Y'], attacker_dict['Z']]
		ally_agent_blood = attacker_dict['blood']
		ally_agent_ammo = attacker_dict['ammo']

		enemy_agent_pos = [defender_dict['X'], defender_dict['Y'], defender_dict['Z']]
		enemy_agent_blood = defender_dict['blood']
		enemy_agent_ammo = defender_dict['ammo']

		a_position = np.array(ally_agent_pos)
		position = np.array(enemy_agent_pos)
		a_ammo = ally_agent_ammo
		a_health = ally_agent_blood
		ammo = enemy_agent_ammo
		health = enemy_agent_blood

		# 进攻方优势计算(选择优势最大的进行打击，若相同则选择距离更近的进行打击）
		if identity == "offensive":
			# 相对距离威胁
			# Mammo = self.SigmoidNine(a_ammo, self.ammo0)
			Mammo = 1
			del_health = 100 - health
			Mhealth = health / 100
			# Mhealth = self.SigmoidNine(a_health, self.heal0)
			# 对方载荷及健康优势计算（减函数）
			# Sammo = np.exp(ammo / self.AMMO0)
			Sammo = 1
			# Shealth = np.exp(health / self.HEAL0) 
			del_a_health = 100 - a_health
			Shealth = a_health / 100		   
			r = a_position - position
			dist = np.sqrt(np.sum(np.square(r)))
			Sr = np.exp(-dist / self.R0)
			S_offensive =  1 * Mammo * Mhealth * Sammo * Shealth * Sr  # 乘了系数10以致于S不过分小

			return S_offensive
		# 防守方优势计算（优先打击距离夺控点近的无人车）
		if identity == "defensive":
			Sr_temp = 0
			# Mammo = self.SigmoidNine(ammo, self.ammo0)
			Mammo = 1
			# Mhealth = self.SigmoidNine(health, self.heal0)
			del_health = 100 - health
			Mhealth = health / 100
			# 对方载荷及健康优势计算（减函数）
			# Sammo = np.exp(-a_ammo / self.AMMO0)
			Sammo = 1
			del_a_health = 100 - a_health
			Shealth = a_health / 100
			for critical_point in self.critical_points:
				r = critical_point - a_position
				dist = np.sqrt(np.sum(np.square(r)))
				Sr = np.exp(-dist / self.R0)
				if Sr >= Sr_temp:
					Sr_temp = Sr
			S_defensive =  Mammo * Mhealth * Sammo * Shealth * Sr_temp
			return S_defensive

	def Drone2Point_id(self, drone_data, key_point):
		drone_pos = [drone_data['X'], drone_data['Y'], drone_data['Z']]
		drone_blood = drone_data['blood']
		drone_velocityx = drone_data['vx']
		drone_velocityy = drone_data['vy']
		drone_velocity = [drone_velocityx, drone_velocityy, 0]
		# 相对距离威胁
		p_position = np.array(key_point)
		position = np.array(drone_pos)
		velocity = np.array(drone_velocity)
		r = p_position - position
		dist = np.sqrt(np.sum(np.square(r)))
		Spr = np.exp(-dist / 1000)   # 此处的放缩系数采用与无人机参数相关的
		# 相对速度威胁
		# V = np.dot(r, velocity) / dist  # 求速度在连线朝向上的投影
		# Spv = self.SigmoidTen(V, 0.2)   # 此处的放缩系数采用与无人机参数相关的
		# 停留时间威胁
		# Spt = self.SigmoidNine(p_ts, 0.5)  # 不能接受无人机停留3秒及以上
		# 计算综合态势
		# Sp = Spt + 0.6 * Spr + 0.2 * Spv  # 建议驱离阈值: Sp >= 0.5
		Sp =  Spr 
		return Sp


	#计算防守方无人车相对于进攻方无人车的态势矩阵
	#矩阵横轴维度为进攻方无人车数量，纵轴维度为防守方无人车数量
	def defend_to_attack(self, self_data, ally_agents_data, enemy_agents_data, key_points):
		#无人车
		all_friend_agents_data = dict(self_data, **ally_agents_data)  # 进攻方所有智能体数据
		for agent_id, dict_value in all_friend_agents_data.items():
			if 'blood' not in dict_value:
				temp = agent_id
		all_friend_agents_data.pop(temp)  # 剔除无人机数据，只考虑地面无人车平台

		#进攻方无人车信息
		all_friend_agent_pos = []
		all_friend_agent_blood = []
		all_friend_agent_velocityx = []
		all_friend_agent_velocityy = []
		all_friend_agent_ammo = []
		all_friend_agent_ID = []
		all_friend_amount = 0
		for agent_id, dict_value in all_friend_agents_data.items():
			all_friend_agent_ID.append(agent_id) #编号接口，形式参照丘老师代码，正确性存疑
			# all_friend_agent_ammo.append(dict_value['ammo']) #载荷接口，形式参照丘老师代码，正确性存疑
			all_friend_agent_velocityx.append(dict_value['velocityx']) #速度接口，形式参照丘老师代码，正确性存疑
			all_friend_agent_velocityy.append(dict_value['velocityy']) #速度接口，形式参照丘老师代码，正确性存疑
			all_friend_agent_blood.append(dict_value['blood']) #血量接口，形式参照丘老师代码，正确性存疑
			all_friend_agent_pos.append([dict_value['X'], dict_value['Y'], dict_value['Z']]) #位置接口，参照丘老师代码编写
			all_friend_amount += 1
		
		#防守方无人车信息
		all_enemy_agent_pos = []
		all_enemy_agent_blood = []
		all_enemy_agent_velocityx = []
		all_enemy_agent_velocityy = []
		all_enemy_agent_ammo = []
		all_enemy_agent_ID = []
		all_enemy_amount = 0
		for agent_id, dict_value in enemy_agents_data.items():
			all_enemy_agent_ID.append(agent_id) 
			# all_enemy_agent_ammo.append(dict_value['ammo']) 
			all_enemy_agent_velocityx.append(dict_value['velocityx'])
			all_enemy_agent_velocityy.append(dict_value['velocityy']) #速度接口，形式参照丘老师代码，正确性存疑
			all_enemy_agent_blood.append(dict_value['blood'])
			all_enemy_agent_pos.append([dict_value['X'], dict_value['Y'], dict_value['Z']])
			all_enemy_amount += 1

		evaluation = np.zeros((all_friend_amount, all_enemy_amount))

		for i in range(all_friend_amount):
			for j in range(all_enemy_amount):
				yaw = math.degrees(math.atan2((all_enemy_agent_pos[j][0] - all_friend_agent_pos[i][0]), (all_enemy_agent_pos[j][1] - all_friend_agent_pos[i][1])))
				# UAV2UAV(self, identity, a_position,a_ammo,a_health, position,velocity,phi,psi,ammo,health)
				all_enemy_agent_velocity = [all_enemy_agent_velocityx[j], all_enemy_agent_velocityy[j], 0]
				evaluation[i][j] = self.UAV2UAV("offensive", all_friend_agent_pos[i], 0, all_friend_agent_blood[i],
				all_enemy_agent_pos[j], all_enemy_agent_velocity, yaw, 0, 0, all_enemy_agent_blood[j])

		return evaluation

	#计算进攻方无人车相对于防守方无人车的态势矩阵
	#矩阵横轴维度为防守方无人车数量，纵轴维度为进攻方无人车数量
	def attack_to_defend(self, self_data, ally_agents_data, enemy_agents_data, key_points):
		#无人车
		all_friend_agents_data = dict(self_data, **ally_agents_data)  # 进攻方所有智能体数据
		for agent_id, dict_value in all_friend_agents_data.items():
			if 'blood' not in dict_value:
				temp = agent_id
		all_friend_agents_data.pop(temp)  # 剔除无人机数据，只考虑地面无人车平台

		#进攻方无人车信息
		all_friend_agent_pos = []
		all_friend_agent_blood = []
		all_friend_agent_velocityx = []
		all_friend_agent_velocityy = []
		all_friend_agent_ammo = []
		all_friend_agent_ID = []
		all_friend_amount = 0
		for agent_id, dict_value in all_friend_agents_data.items():
			all_friend_agent_ID.append(agent_id) #编号接口，形式参照丘老师代码，正确性存疑
			#all_friend_agent_ammo.append(dict_value['ammo']) #载荷接口，形式参照丘老师代码，正确性存疑
			all_friend_agent_velocityx.append(dict_value['velocityx']) #速度接口，形式参照丘老师代码，正确性存疑
			all_friend_agent_velocityy.append(dict_value['velocityy']) #速度接口，形式参照丘老师代码，正确性存疑
			all_friend_agent_blood.append(dict_value['blood']) #血量接口，形式参照丘老师代码，正确性存疑
			all_friend_agent_pos.append([dict_value['X'], dict_value['Y'], dict_value['Z']]) #位置接口，参照丘老师代码编写
			all_friend_amount += 1

		#防守方无人车信息
		all_enemy_agent_pos = []
		all_enemy_agent_blood = []
		all_enemy_agent_velocityx = []
		all_enemy_agent_velocityy = []
		all_enemy_agent_ammo = []
		all_enemy_agent_ID = []
		all_enemy_amount = 0
		for agent_id, dict_value in enemy_agents_data.items():
			all_enemy_agent_ID.append(agent_id) 
			#all_enemy_agent_ammo.append(dict_value['ammo']) 
			all_enemy_agent_velocityx.append(dict_value['velocityx'])
			all_enemy_agent_velocityy.append(dict_value['velocityy'])
			all_enemy_agent_blood.append(dict_value['blood'])
			all_enemy_agent_pos.append([dict_value['X'], dict_value['Y'], dict_value['Z']])
			all_enemy_amount += 1

		evaluation = np.zeros((all_enemy_amount, all_friend_amount))

		for i in range(all_enemy_amount):
			for j in range(all_friend_amount):
				yaw = math.degrees(math.atan2((all_enemy_agent_pos[j][0] - all_friend_agent_pos[i][0]), (all_enemy_agent_pos[j][1] - all_friend_agent_pos[i][1])))
				# UAV2UAV(self, identity, a_position,a_ammo,a_health, position,velocity,phi,psi,ammo,health)
				all_friend_agent_velocity = [all_friend_agent_velocityx[j], all_friend_agent_velocityy[j], 0]
				evaluation[i][j] = self.UAV2UAV("defensive", all_enemy_agent_pos[i], 0, all_enemy_agent_blood[i], 
				all_friend_agent_pos[j], all_friend_agent_velocity, yaw, 0, 0, all_friend_agent_blood[j])

		return evaluation

	#计算无人机对于夺控点位置的态势矩阵
	#矩阵横轴代表夺控点，纵轴代表无人机
	def uav_to_defend(self, self_data, ally_agents_data, enemy_agents_data, key_points):
		all_friend_agents_data = dict(self_data, **ally_agents_data)  # 进攻方所有智能体数据
		for agent_id, dict_value in all_friend_agents_data.items():
			if 'blood' not in dict_value:
				temp1 = agent_id
				temp2 = dict_value
		
		drone_data = {}
		drone_data[temp1] = temp2

		#无人机信息
		drone_pos = []
		drone_velocityx = []
		drone_velocityy = []
		drone_ID = []
		drone_amount = 0
		for agent_id, dict_value in drone_data.items():
			drone_ID.append(agent_id) #编号接口，形式参照丘老师代码，正确性存疑
			drone_velocityx.append(dict_value['velocityx']) #速度接口，形式参照丘老师代码，正确性存疑
			drone_velocityy.append(dict_value['velocityy']) #速度接口，形式参照丘老师代码，正确性存疑
			drone_pos.append([dict_value['X'], dict_value['Y'], dict_value['Z']]) #位置接口，参照丘老师代码编写
			drone_amount += 1

		#夺控点位置
		key_point_amount = 0
		key_point_pos = []
		for key_point in key_points:
			key_point_pos.append(key_point)
			key_point_amount += 1

		evaluation = np.zeros((key_point_amount, drone_amount))

		for i in range(key_point_amount):
			for j in range(drone_amount):
				# Drone2Point(self, p_position,p_ts, position, velocity)
				# print(self.Drone2Point(key_point_pos[i], 0, drone_pos[j], drone_velocity[j]))
				drone_velocity = [drone_velocityx[j], drone_velocityy[j], 0]
				evaluation[i][j] = self.Drone2Point(key_point_pos[i], 0, drone_pos[j], drone_velocity)
				

		return evaluation
	
	'''
	#计算无人车对于周围位置点的态势评估矩阵
	#
	def attack_to_point(self_data, ally_agents_data, enemy_agents_data, key_points):
		#无人车
		all_friend_agents_data = dict(self_data, **ally_agents_data)  # 进攻方所有智能体数据
		all_friend_agents_data.pop("231")  # 剔除无人机数据，只考虑地面无人车平台

		#进攻方无人车信息
		all_friend_agent_pos = []
		all_friend_agent_blood = []
		all_friend_agent_velocity = []
		all_friend_agent_ammo = []
		all_friend_agent_ID = []
		all_friend_amount = 0
		for agent_id, dict_value in all_friend_agents_data.items():
			all_friend_agent_ID.append(dict_value['ID']) #编号接口，形式参照丘老师代码，正确性存疑
			all_friend_agent_ammo.append(dict_value['ammo']) #载荷接口，形式参照丘老师代码，正确性存疑
			all_friend_agent_velocity.append(dict_value['velocity']) #速度接口，形式参照丘老师代码，正确性存疑
			all_friend_agent_blood.append(dict_value['blood']) #血量接口，形式参照丘老师代码，正确性存疑
			all_friend_agent_pos.append([dict_value['X'], dict_value['Y'], dict_value['Z']]) #位置接口，参照丘老师代码编写
			all_friend_amount += 1

		evaluation = np.zeros((all_friend_amount, all_enemy_amount))

		for i in range(all_enemy_amount):
			for j in range(all_friend_amount):
				yaw = math.degrees(math.atan2((all_enemy_agent_pos[j][0] - all_friend_agent_pos[i][0]), (all_enemy_agent_pos[j][1] - all_friend_agent_pos[i][1])))
				#UAV2UAV(self, identity, a_position,a_ammo,a_health, position,velocity,phi,psi,ammo,health)
				evaluation[i][j] = self.UAV2UAV("defensive", all_enemy_agent_pos[i], all_enemy_agent_ammo[i], all_enemy_agent_blood[i], 
				all_friend_agent_pos[j], all_friend_agent_velocity[j], yaw, 0, all_friend_agent_ammo[j], all_friend_agent_blood[j])

		return evaluation
	'''
	#态势评估主函数
	def evaluate(self, self_data, ally_agents_data, enemy_agents_data, key_points):
		d2a = self.defend_to_attack(self_data, ally_agents_data, enemy_agents_data, key_points)
		a2d = self.attack_to_defend(self_data, ally_agents_data, enemy_agents_data, key_points)
		u2d = self.uav_to_defend(self_data, ally_agents_data, enemy_agents_data, key_points)
		return d2a, a2d, u2d




def test():
    evaluator = Evaluation_module()

    


# test
if __name__ == '__main__':
	
	test()