import copy
import random
import numpy as np
import datetime
import time
from ALGORITHM.script_ai.stance import *
from ALGORITHM.script_ai.global_params import *


class TaskAssign(object):
	"""docstring for TaskAssign"""
	def __init__(self, attackers, drone, defenders):
		super(TaskAssign, self).__init__()
		self.attackers = attackers
		self.drone = drone
		self.defenders = defenders

		self.evaluator = Evaluation_module()

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
	def assign_defend(self, ):

		# if alive_attackers>0:
		# defend_ids = list(self.defenders.keys())
		attack_ids = [ID for ID, attr in self.attackers.items() if 'dead' not in attr['state']]

		def2att_ids = [defender['state'][1] for defender in self.defenders.values() if defender['state'][0] == 'attack']
		avail_ids = [ID for ID in attack_ids if ID not in def2att_ids]

		if len(avail_ids)>0:
			return ['attack', avail_ids[0]]
		else:
			return ['attack', def2att_ids[0]]

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
	def assign_expel(self, ):
		drone_pos = [self.drone['X'], self.drone['Y']]
		drone_dist = []
		for key_point in key_points:
			drone_dist.append(np.linalg.norm(np.array(key_point)-np.array(drone_pos)))

		if min(drone_dist) < DRIVE_AWAY_DIST:
			return ['expel', drone_dist.index(min(drone_dist))]
		else:
			return None



# test
if __name__ == '__main__':

	assigner = TaskAssign()
	attack_goals, defend_goals, avoid_goals, uav_point = align.assign_all(ally_agents_data, enemy_agents_data, key_points)