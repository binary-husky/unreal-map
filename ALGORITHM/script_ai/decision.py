import copy
import random
import numpy as np
import datetime
import time
from ALGORITHM.script_ai.assignment import *
from ALGORITHM.script_ai.global_params import *



class decision():
	"""docstring for decision"""
	def __init__(self, attackers, drone, defenders):
		super(decision, self).__init__()
		self.attackers = attackers
		self.drone = drone
		self.defenders = defenders

		self.assigner = TaskAssign(self.attackers, self.drone, self.defenders)


	# output actions
	def act(self, type=None):
		actions_list = {}

		self.alive_attack = len([att for att in list(self.attackers.values()) if 'dead' not in att['state']])
		self.alive_defend = len([ded for ded in list(self.defenders.values()) if 'dead' not in ded['state']])

		if not type:
			type = 'attackers'

		if type == 'attackers':
			self.attack_StateTrans()

			for ID, attr in self.attackers.items():

				if 'dead' in attr['state']:
					des_pos = [attr['X'], attr['Y'], attr['Z']]
					actions_list[ID] = des_pos
					continue

				if 'attack' in attr['state'] and attr['state'][1] is not '0':
					def_ID = attr['state'][1]
					opp = self.defenders[def_ID]
					des_pos = [opp['X']-400, opp['Y']-400, opp['Z']]
					actions_list[ID] = des_pos

				elif 'retreat' in attr['state']:
					if self.drone['state'][0] is not 'idle':
						select_key_points = key_points[self.drone['state'][1]]
						agent_pos = [attr['X'], attr['Y'], attr['Z']]
						if select_key_points[1] - agent_pos[1] > 50 and select_key_points[0] - agent_pos[0] > 50:
							k = (select_key_points[1] - agent_pos[1]) / (select_key_points[0] - agent_pos[0])
							des_pos = [agent_pos[0] - 1400, agent_pos[1] - 1400 * k, agent_pos[2]]
						else:
							des_pos = [agent_pos[0] - 1400, agent_pos[1], agent_pos[2]]
					else:
						des_pos = ATTA_RETREAT_POS
					actions_list[ID] = des_pos
					# des_pos = ATTA_RETREAT_POS
					# actions_list[ID] = des_pos

				elif 'idle' in attr['state']:
					des_pos = [attr['X'], attr['Y'], attr['Z']]
					actions_list[ID] = des_pos

			# attack target assign
			assign_attackers = [ID for ID, attr in self.attackers.items() if (len(attr['state'])>1 and attr['state'][1] == '0')]

			# attacker assignment
			if len(assign_attackers)>0:
				target_ID = self.assigner.assign_attack(assign_attackers)
				target = self.defenders[target_ID]

			for attacker_ID in assign_attackers:
				actions_list[attacker_ID] = [target['X']-300, target['Y']-300, target['Z']]
				self.attackers[attacker_ID]['state'][1] = target_ID


			# drone action
			self.drone_StateTrans()
			if 'idle' in self.drone['state']:
				des_pos = [self.drone['X'], self.drone['Y'], self.drone['Z']]
			elif 'running' in self.drone['state'] or 'hold' in self.drone['state']:
				des_pos = key_points[self.drone['state'][1]]

			actions_list['drone'] = des_pos


		# same as attackers
		elif type == 'defenders':
			self.defend_StateTrans()

			for ID, attr in self.defenders.items():

				if 'dead' in attr['state']:
					des_pos = [attr['X'], attr['Y'], attr['Z']]
					actions_list[ID] = des_pos
					continue

				if 'attack' in attr['state']:
					def_ID = attr['state'][1]
					opp = self.attackers[def_ID]
					des_pos = [opp['X'], opp['Y'], opp['Z']]

				elif 'expel' in attr['state']:
					des_pos = key_points[attr['state'][1]]

				elif 'retreat' in attr['state']:
					des_pos = DEF_RETREAT_POS

				elif 'idle' in attr['state']:
					des_pos = key_points[0]

				# else:
				# 	des_pos = [attr['X'], attr['Y'], attr['Z']]

				actions_list[ID] = des_pos


		else:
			raise ValueError('invalid type!')

		return actions_list



	# state machine
	def attack_StateTrans(self, ):

		# attackers
		for ID in list(self.attackers.keys()):
			attr = self.attackers[ID]

			# check alive/dead(blood thres: 2)
			if attr['blood'] <= 2 and 'dead' not in attr['state']:
				attr['blood'] = 0
				attr['state'] = ['dead']
				continue

			# idle2attack
			if 'idle' in attr['state']:
				if self.alive_defend > 0:
					attr['state'] = ['attack']
					def_ID = '0'
					attr['state'].append(def_ID)

				continue

			# attack2idle/retreat(blood thres: 10)
			if 'attack' in attr['state']:
				def_ID = attr['state'][1]
				is_attack = self.assigner.is_attack(ID)

				# 2idle
				if 'dead' in self.defenders[def_ID]['state']:
					attr['state'] = ['attack']
					def_ID = '0'
					attr['state'].append(def_ID)
				elif not is_attack:
					attr['state'] = ['retreat']

				continue

			# retreat2attack
			if 'retreat' in attr['state']:
				if self.alive_defend > 0 and attr['blood'] > 10:
					is_attack = self.assigner.is_attack(ID)
					if is_attack:
						attr['state'] = ['attack']
						def_ID = '0'
						attr['state'].append(def_ID)
					# dist_list = [np.linalg.norm(np.array([attr['X'], attr['Y'], attr['Z']]) - np.array([ded['X'], ded['Y'], ded['Z']])) for ded in self.defenders.values()]
					# min_dist = min(dist_list)
					# if min_dist > 1500:
					# 	attr['state'] = ['attack']
					# 	def_ID = '0'
					# 	attr['state'].append(def_ID)

		# if 'idle' in self.drone['state']:
		# self.drone['state'] = self.assigner.assign_2point()


	def drone_StateTrans(self, ):

		drone_pos = [self.drone['X'], self.drone['Y']]

		# initial assign
		if 'idle' in self.drone['state']:
			self.drone['state'] = self.assigner.assign_drone_ini()

		# # run2hold
		# elif 'running' in self.drone['state']:
		# 	cur_point_idx = self.drone['state'][1]
		# 	cur_point_pos = key_points[cur_point_idx]
		# 	if np.linalg.norm(np.array(drone_pos) - np.array(cur_point_pos)) < 10:
		# 		self.drone['state'] = ['hold', cur_point_idx]
		# 	else:
		# 		pass
		elif 'running' in self.drone['state']:
			cur_point_idx = self.drone['state'][1]
			cur_point_pos = key_points[cur_point_idx]
			if self.assigner.judge_expeled():
				self.drone['state'] = ['running', int(1 - cur_point_idx)]
				self.drone['state'] = self.assigner.assign_drone_ini()

			else:
				pass

	# defender
	def defend_StateTrans(self, ):
		for ID in list(self.defenders.keys()):
			attr = self.defenders[ID]

			# check alive/dead
			if attr['blood'] <= 2 and 'dead' not in attr['state']:
				attr['blood'] = 0
				attr['state'] = ['dead']
				continue

			# # expel nearest
			# if self.assigner.assign_expel(ID) is not None:
			# 	attr['state'] = self.assigner.assign_expel(ID)
			# 	continue

			# # expel both
			# if self.assigner.assign_expel() is not None:
			# 	attr['state'] = self.assigner.assign_expel()
			# 	continue
			#
			# # idle2attack
			# if self.alive_attack>0 and self.assigner.assign_defend() is not None:
			# 	attr['state'] = self.assigner.assign_defend()
			# 	continue
			attr['state'] = self.assigner.assign_defend(ID)

			# attack2idle/retreat(blood thres: 10)
			if 'attack' in attr['state']:
				att_ID = attr['state'][1]

				# 2retreat
				if attr['blood'] <= 10 and att_ID in self.attackers.keys() and self.attackers[att_ID]['blood'] > 5:
					attr['state'] = ['retreat']

				# 2idle
				elif 'dead' in self.attackers[att_ID]['state']:
					attr['state'] = ['idle']

			# else:
			# 	attr['state'] = self.assigner.assign_defend(ID)



def test():

	# test initial data
	# ally_agents_data={"221": {"ammo": 100, "velocity": 0.5, "X":1, "Y":1, "Z":0, "Yaw":0, 'blood':100, 'state': ['idle']}, "231": {"ammo": 100, "velocity": 0.5, "X":2, "Y":2, "Z":1.5, "Yaw":0, 'blood':100, 'state': ['idle']}}
	# enemy_agents_data={"311": {"ammo": 100, "velocity": 0.8, "X":5, "Y":5, "Z":0, "Yaw":0, 'blood':100, 'state': ['idle']}, "321": {"ammo": 100, "velocity": 0.8, "X":6, "Y":6, "Z":0, "Yaw":0, 'blood':100, 'state': ['idle']}}

	# test dead detection √
	# ally_agents_data={"221": {"ammo": 100, "velocity": 0.5, "X":1, "Y":1, "Z":0, "Yaw":0, 'blood':0, 'state': ['idle']}, "231": {"ammo": 100, "velocity": 0.5, "X":2, "Y":2, "Z":1.5, "Yaw":0, 'blood':1, 'state': ['retreat']}}
	# drone_data={"ammo": 100, "velocity": 0.5, "X":1, "Y":1, "Z":0, "Yaw":0, 'blood':0, 'state': ['idle']}
	# enemy_agents_data={"311": {"ammo": 100, "velocity": 0.8, "X":5, "Y":5, "Z":0, "Yaw":0, 'blood':1, 'state': ['attack']}, "321": {"ammo": 100, "velocity": 0.8, "X":6, "Y":6, "Z":0, "Yaw":0, 'blood':0, 'state': ['dead']}}

	# test idle2attack √
	# ally_agents_data={"221": {"ammo": 100, "velocity": 0.5, "X":1, "Y":1, "Z":0, "Yaw":0, 'blood':100, 'state': ['idle']}, "231": {"ammo": 100, "velocity": 0.5, "X":2, "Y":2, "Z":1.5, "Yaw":0, 'blood':100, 'state': ['idle']}}
	# drone_data={"ammo": 100, "velocity": 0.5, "X":1, "Y":1, "Z":0, "Yaw":0, 'blood':0, 'state': ['idle']}
	# enemy_agents_data={"311": {"ammo": 100, "velocity": 0.8, "X":5, "Y":5, "Z":0, "Yaw":0, 'blood':100, 'state': ['idle']}, "321": {"ammo": 100, "velocity": 0.8, "X":6, "Y":6, "Z":0, "Yaw":0, 'blood':100, 'state': ['idle']}}

	# test attack2idle √
	# ally_agents_data={"221": {"ammo": 100, "velocity": 0.5, "X":1, "Y":1, "Z":0, "Yaw":0, 'blood':100, 'state': ['attack', '311']}, "231": {"ammo": 100, "velocity": 0.5, "X":2, "Y":2, "Z":1.5, "Yaw":0, 'blood':100, 'state': ['attack', '311']}}
	# drone_data={"ammo": 100, "velocity": 0.5, "X":1, "Y":1, "Z":0, "Yaw":0, 'blood':0, 'state': ['idle']}
	# enemy_agents_data = {}

	# test attack2retreat √
	ally_agents_data={"221": {"ammo": 100, "velocity": 0.5, "X":1, "Y":1, "Z":0, "Yaw":0, 'blood':11, 'state': ['attack', '311']}, "231": {"ammo": 100, "velocity": 0.5, "X":2, "Y":2, "Z":1.5, "Yaw":0, 'blood':10, 'state': ['attack', '311']}}
	drone_data={"ammo": 100, "velocity": 0.5, "X":-3, "Y":0, "Z":0, "Yaw":0, 'blood':0, 'state': ['idle']}
	enemy_agents_data={"311": {"ammo": 100, "velocity": 0.8, "X":2, "Y":1, "Z":0, "Yaw":0, 'blood':6, 'state': ['idle']}, "321": {"ammo": 100, "velocity": 0.8, "X":6, "Y":6, "Z":0, "Yaw":0, 'blood':100, 'state': ['idle']}}
	# enemy_agents_data={"311":{"ammo": 100, "velocity": 0.5, "X":2, "Y":2, "Z":0, "Yaw":0, 'blood':11, 'state': ['expel']}}

	DecisionMake = decision(ally_agents_data, drone_data, enemy_agents_data)

	attackers = DecisionMake.attackers
	defenders = DecisionMake.defenders
	drone = DecisionMake.drone

	# decision module test
	attack_actions = DecisionMake.act(type='attackers')
	defend_actions = DecisionMake.act(type='defenders')

	print('attack property: ', attackers)
	print('defend property: ', defenders)

	att_states = []
	def_states = []
	for k, v in attackers.items():
		att_states.append({k:v['state']})
	for k, v in defenders.items():
		def_states.append({k:v['state']})

	drone_state = drone['state']

	print('attack states: ', att_states)
	print('defend states: ', def_states)
	print('drone states: ', drone_state)

	print('attack actions: ', attack_actions)
	print('defend actions: ', defend_actions)



if __name__ == '__main__':

	test()

	# DecisionMake = decision(ally_agents_data, enemy_agents_data)

	# while(1):

	# 	attack_actions = DecisionMake.act(type='attackers')
	# 	defend_actions = DecisionMake.act(type='defenders')

	# 	time.sleep(0.05)
