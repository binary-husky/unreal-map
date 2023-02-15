import copy
import random
import numpy as np
import datetime
import time
from ALGORITHM.script_ai.module_evaluation import *
from ALGORITHM.script_ai.global_params import *


class TaskAssign(object):
	"""docstring for TaskAssign"""
	def __init__(self, attackers, drone, defenders):
		super(TaskAssign, self).__init__()
		self.attackers = attackers
		self.drone = drone
		self.defenders = defenders

		self.evaluator = Evaluation_module()

		# params
		self.ratio_thres = 1

	# UGV attack(get defender target id)
	def assign_attack(self, attack_IDlist):
		# simple strategy -- one opponent
		max_stlist = []
		max_idlist = []
		def_idlist = []
		for ID, attr in self.defenders.items():
			if 'dead' not in attr['state']:
				def_idlist.append(ID)
		for att in attack_IDlist:
			inner_stance_list = [self.evaluator.UAV2UAV_id('offensive', self.attackers[att], defender) for defender in self.defenders.values() if 'dead' not in defender['state']]
			# print(inner_stance_list)
			inner_max_stance = max(inner_stance_list)
			max_stlist.append(inner_max_stance)
			max_idlist.append(inner_stance_list.index(inner_max_stance))

		max_stance = max(max_stlist)
		target_id = def_idlist[max_idlist[max_stlist.index(max_stance)]]
		return target_id


	# # UAV hold(hold 1)
	# def assign_drone_initial(self, ):
	# 	# check for key points
	# 	all_defender_pos = []
	# 	min_dist = []
	# 	for attr in self.defenders.values():
	# 		all_defender_pos.append([attr['X'], attr['Y']])

	# 	if len(all_defender_pos)>0:
	# 		for key_point in key_points:
	# 			min_dist.append(min([np.linalg.norm(np.array(key_point)-np.array(enemy_pos)) for enemy_pos in all_defender_pos]))

	# 		if max(min_dist) > DRIVE_AWAY_DIST:
	# 			return ['running', min_dist.index(max(min_dist))]
	# 		else:
	# 			return ['running', 0]
	# 	else:
	# 		return ['running', 0]


	# UAV hold(running back and forth)
	def assign_drone_ini(self, ):
		# check for key points
		all_defender_pos = []
		min_dist = []
		for attr in self.defenders.values():
			all_defender_pos.append([attr['X'], attr['Y']])

		if len(all_defender_pos) > 0:
			for key_point in key_points:
				min_dist.append(
					min([np.linalg.norm(np.array(key_point) - np.array(enemy_pos)) for enemy_pos in all_defender_pos]))
			return ['running', min_dist.index(max(min_dist))]
		else:
			return ['running', 0]
		# all_defender_pos = []
		# drone_dicvalue=[]
		# drone_pos=[]
		# min_dist = []
		# for attr in self.defenders.values():
		# 	all_defender_pos.append([attr['X'], attr['Y']])
		# for attr in self.drone.values():
		# 	drone_dicvalue.append(attr)
		# drone_pos.append([drone_dicvalue[2],drone_dicvalue[3]])
		# drone_pos.append([drone_dicvalue[2], drone_dicvalue[3]])
		# if len(all_defender_pos)>0:
		# 	for key_point in key_points:
		# 		drone2point=[np.linalg.norm(np.array(key_point) - np.array(pos)) for pos in drone_pos]
		# 		defender2point=[np.linalg.norm(np.array(key_point) - np.array(enemy_pos)) for enemy_pos in all_defender_pos]
		# 		min_dist.append(min([a/b for a,b in zip(drone2point,defender2point)]))
		# 	return ['running', min_dist.index(min(min_dist))]
		# else:
		# 	return ['running', 0]


	def judge_expeled(self, ):

		all_defender_pos = []
		# drone_pos = [self.drone['X'], self.drone['Y']]
		key_point_idx = self.drone['state'][1]
		key_point_pos = key_points[key_point_idx]

		for attr in self.defenders.values():
			all_defender_pos.append([attr['X'], attr['Y']])

		min_dist = min([np.linalg.norm(np.array(key_point_pos) - np.array(def_pos)) for def_pos in all_defender_pos])

		if min_dist < DRIVE_AWAY_DIST:
			return True
		else:
			return False


	# # UGV defend(negetive defend)
	# def assign_defend(self, def_ID):
	# 	defend = self.defenders[def_ID]
	# 	defend_pos = [defend['X'], defend['Y']]

	# 	all_attacker_pos = []
	# 	all_attacker_ids = []
	# 	for ID, attr in self.attackers.items():
	# 		all_attacker_pos.append([attr['X'], attr['Y']])
	# 		all_attacker_ids.append(ID)

	# 	if len(all_attacker_pos) > 0:
	# 		dist_list = [np.linalg.norm(np.array(defend_pos)-np.array(attack_pos)) for attack_pos in all_attacker_pos]
	# 		min_dist = min(dist_list)
	# 		if min_dist < DEFEND_DIST:
	# 			return ['attack', all_attacker_ids[dist_list.index(min_dist)]]
	# 		else:
	# 			return None
	# 	else:
	# 		return None


	# UGV defend(active defend)
	# def defend(self, def_ID):

		# attack_ids = [ID for ID, attr in self.attackers.items() if 'dead' not in attr['state']]
		# defend_ids = [ID for ID, attr in self.defenders.items() if 'dead' not in attr['state']]
		#
		# alive_attack = len(attack_ids)
		# alive_defend = len(defend_ids)
		#
		# if alive_attack > 0 and alive_defend > 0 and def_ID in defend_ids:
		# 	if alive_attack == alive_defend:
		# 		idx = defend_ids.index(def_ID)
		# 		return ['attack', attack_ids[idx]]
		# 	elif alive_attack < alive_defend:
		# 		return ['attack', attack_ids[0]]
		# 	else:
		# 		return ['attack', attack_ids[0]]
		# else:
		# 	return ['idle']

		# def2att_ids = [defender['state'][1] for defender in self.defenders.values() if defender['state'][0] == 'attack']
		# avail_ids = [ID for ID in attack_ids if ID not in def2att_ids]
		#
		# if len(avail_ids)>0:
		# 	return ['attack', avail_ids[0]]
		# else:
		# 	return ['attack', def2att_ids[0]]

	# # UGV expel(nearest assign)
	# def assign_expel(self, def_ID):
	# 	drone_pos = [self.drone['X'], self.drone['Y']]
	# 	drone_min_dist = []
	# 	for key_point in key_points:
	# 		drone_min_dist.append(np.linalg.norm(np.array(key_point)-np.array(drone_pos)))

	# 	if min(drone_min_dist) < DRIVE_AWAY_DIST:
	# 		min_dist = 1000
	# 		min_id = list(self.defenders.keys())[0]
	# 		key_point_idx = drone_min_dist.index(min(drone_min_dist))
	# 		for ID, attr in self.defenders.items():
	# 			expel_dist = np.linalg.norm(np.array(key_points[key_point_idx])-np.array([attr['X'], attr['Y']]))
	# 			if expel_dist < min_dist:
	# 				min_dist = expel_dist
	# 				min_id = ID
	# 		if def_ID == min_id:
	# 			return ['expel', key_point_idx]
	# 		else:
	# 			return None
	# 	else:
	# 		return None

	# UGV expel(both)
	def expel(self, ):
		drone_pos = [self.drone['X'], self.drone['Y']]
		drone_dist = []
		for key_point in key_points:
			drone_dist.append(np.linalg.norm(np.array(key_point)-np.array(drone_pos)))
		# if min(drone_dist) < DRIVE_AWAY_DIST:
		return ['expel', drone_dist.index(min(drone_dist))]
		# else:
		# 	return None

	# defender assign
	def assign_defend(self, def_ID):

		alive_attackers = [attacker for attacker in self.attackers.values() if 'dead' not in attacker['state']]
		alive_attackers_ids = [ID for ID, attr in self.attackers.items() if 'dead' not in attr['state']]
		dist = [np.linalg.norm(np.array([att['X'], att['Y']]) - np.array([self.defenders[def_ID]['X'], self.defenders[def_ID]['Y']])) for att in alive_attackers]

		if len(dist) > 0 and min(dist) < DEFEND_DIST:
			idx = dist.index(min(dist))
			return ['attack', alive_attackers_ids[idx]]
		else:
			return self.expel()

		# alive_attacker_list = [att for att in self.attackers.values() if 'dead' not in att['state']]
		#
		# if len(alive_attacker_list) > 0:
		# 	UGV_stance = [self.evaluator.UAV2UAV_id('offensive', attacker, self.defenders[def_ID]) for attacker in alive_attacker_list]
		# 	# print(inner_stance_list)
		# 	max_UGV_stance = max(UGV_stance)
		#
		# 	drone_stance = [self.evaluator.Drone2Point_id(self.drone, keyPoint) for keyPoint in key_points]
		# 	max_drone_stance = max(drone_stance)
		#
		# 	print('UGV stance: ', max_UGV_stance)
		# 	print('drone stance: ', max_drone_stance)
		# 	print('ratio: ', max_UGV_stance/max_drone_stance)
		#
		# 	ratio = max_UGV_stance/max_drone_stance
		#
		# 	if ratio > self.ratio_thres:
		# 		assigned_state = self.defend(def_ID)
		# 	else:
		# 		assigned_state = self.expel()
		#
		# else:
		# 	assigned_state = self.expel()
		#
		# return assigned_state

	# judge retreat for attackers
	def is_retreat(self, att_ID, def_ID):
		if 'dead' in self.attackers[att_ID]['state'] or 'dead' in self.defenders[def_ID]['state']:
			return False
		else:
			# # dist version
			# attacker_pos = [self.attackers[att_ID]['X'], self.attackers[att_ID]['Y']]
			# dist_list = [np.linalg.norm(np.array(attacker_pos) - np.array([defender['X'], defender['Y']])) for defender \
			# 			 in self.defenders.values()]
			# if min(dist_list) < 800:
			# 	return True
			# else:
			# 	return False

			# stance_version
			stance = self.evaluator.UAV2UAV_id('offensive', self.attackers[att_ID], self.defenders[def_ID])
			if stance > RETREAT_STANCE:
				print('retreat stance: ', stance)
				return True
			else:
				return False


	def is_attack(self, att_ID):
		if 'dead' in self.attackers[att_ID]['state']:
			return False
		else:
			# # dist version
			# attacker_pos = [self.attackers[att_ID]['X'], self.attackers[att_ID]['Y']]
			# dist_list = [np.linalg.norm(np.array(attacker_pos) - np.array([defender['X'], defender['Y']])) for defender \
			# 			 in self.defenders.values()]
			# if min(dist_list) > 1000:
			# 	return True
			# else:
			# 	return False

			# stance version
			stance_list = [self.evaluator.UAV2UAV_id('offensive', self.attackers[att_ID], self.defenders[def_ID]) for def_ID in \
						   self.defenders.keys() if 'dead' not in self.defenders[def_ID]['state']]
			if max(stance_list) < RETREAT_STANCE:
				print('attack stance: ', max(stance_list))
				return True
			else:
				print('retreat stance: ', max(stance_list))
				return False

# test
# if __name__ == '__main__':
#
# 	# assigner = TaskAssign()
# 	# attack_goals, defend_goals, avoid_goals, uav_point = align.assign_all(ally_agents_data, enemy_agents_data, key_points)