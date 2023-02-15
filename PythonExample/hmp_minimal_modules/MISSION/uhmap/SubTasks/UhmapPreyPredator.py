import json, copy, re, os, inspect, os
import numpy as np
from UTIL.tensor_ops import my_view, repeat_at
from ...common.base_env import RawObsArray
from ..actionset_v3 import digitsToStrAction
from ..agent import Agent
from ..uhmap_env_wrapper import UhmapEnv, ScenarioConfig
from .UhmapPreyPredatorConf import SubTaskConfig
from .SubtaskCommonFn import UhmapCommonFn
from .cython_func import tear_num_arr




class UhmapPreyPredator(UhmapCommonFn, UhmapEnv):
    def __init__(self, rank) -> None:
        super().__init__(rank)
        self.observation_space = self.make_obs(get_shape=True)
        self.SubTaskConfig = SubTaskConfig
        inspect.getfile(SubTaskConfig)
        assert os.path.basename(inspect.getfile(SubTaskConfig)) == type(self).__name__+'Conf.py', \
                ('make sure you have imported the correct SubTaskConfig class')


    def init_ground(self, agent_info, pos_ro):
        N_COL = 4
        agent_class = agent_info['type']
        team = agent_info['team']
        n_team_agent = 50
        tid = agent_info['tid']
        uid = agent_info['uid']
        x = 0 + 800*(tid - n_team_agent//2) //N_COL
        y = (400* (tid%N_COL) + 2000) * (-1)**(team+1)
        x,y = np.matmul(np.array([x,y]), np.array([[np.cos(pos_ro), -np.sin(pos_ro)], [np.sin(pos_ro), np.cos(pos_ro)] ]))
        z = 500 # 500 is slightly above the ground
        yaw = 90 if team==0 else -90
        assert np.abs(x) < 15000.0 and np.abs(y) < 15000.0
        agent_property = copy.deepcopy(SubTaskConfig.AgentPropertyDefaults)
        agent_property.update({
                'DebugAgent': False,
                # max drive/fly speed
                'MaxMoveSpeed':  720          if agent_class == 'RLA_CAR_Laser' else 600,
                # also influence object mass, please change it with causion!
                'AgentScale'  : { 'x': 1,  'y': 1, 'z': 1, },
                # team belonging
                'AgentTeam': team,
                # choose ue class to init
                'ClassName': agent_class,
                # debugging
                'RSVD1': '-Ring1=2000 -Ring2=1400 -Ring3=750',
                # the rank of agent inside the team
                'IndexInTeam': tid, 
                # the unique identity of this agent in simulation system
                'UID': uid, 
                # show color
                'Color':'(R=0,G=1,B=0,A=1)' if team==0 else '(R=0,G=0,B=1,A=1)',
                # initial location
                'InitLocation': { 'x': x,  'y': y, 'z': z, },
                # initial facing direction et.al.
                'InitRotator': { 'pitch': 0,  'roll': 0, 'yaw': yaw, },
        }),
        return agent_property



    def extract_key_gameobj(self, resp):
        """
            获取非智能体的仿真物件,例如重要landmark等
        """
        keyObjArr = resp['dataGlobal']['keyObjArr']
        return keyObjArr

    def gen_reward_and_win(self, resp):
        """
            奖励的设计在此定义,
            (UE端编程死板,虽然预留了相关字段,
            但请不要在UE端提供奖励的定义。)
            建议:在UE端定义触发奖励的事件,如智能体阵亡、战术目标完成等,见parse_event
        """
        reward = [0]*self.n_teams
        events = resp['dataGlobal']['events']
        WinningResult = None
        for event in events: 
            event_parsed = self.parse_event(event)
            # if event_parsed['Event'] == 'Destroyed':
            #     team = self.find_agent_by_uid(event_parsed['UID']).team
            #     reward[team]    -= 0.05    # this team
            #     reward[1-team]  += 0.10    # opp team
            if event_parsed['Event'] == 'EndEpisode':
                # print([a.alive * a.hp for a in self.agents])
                PredatorWin = False
                PredatorRank = False
                PredatorReward = 0
                PreyWin = -1
                PreyRank = -1
                PreyReward = 0
                EndReason = event_parsed['EndReason']
                # According to MISSION\uhmap\SubTasks\UhmapPreyPredatorConf.py, team 0 is prey team, team 1 is predator team
                if EndReason == "AllPreyCaught" or EndReason == "Team_0_AllDead":
                    PredatorWin = True; PredatorRank = 0; PredatorReward = 1
                    PreyWin = False; PreyRank = 1; PreyReward = -1
                elif EndReason == "TimeMaxCntReached" or EndReason == "Team_1_AllDead":
                    PredatorWin = False; PredatorRank = 1; PredatorReward = -1
                    PreyWin = True; PreyRank = 0; PreyReward = 1
                else:
                    print('unexpected end reaon:', EndReason)
                    
                WinningResult = {"team_ranking": [PreyRank, PredatorRank], "end_reason": EndReason}

                reward = [PreyReward, PredatorReward]
        # print(reward)
        return reward, WinningResult





    @staticmethod
    def item_random_mv(src,dst,prob,rand=False):
        assert len(src.shape)==1; assert len(dst.shape)==1
        if rand: np.random.shuffle(src)
        len_src = len(src)
        n_mv = (np.random.rand(len_src) < prob).sum()
        item_mv = src[range(len_src-n_mv,len_src)]
        src = src[range(0,0+len_src-n_mv)]
        dst = np.concatenate((item_mv, dst))
        return src, dst

    @staticmethod
    def get_binary_array(n_int, n_bits=8, dtype=np.float32):
        arr = np.zeros((*n_int.shape, n_bits), dtype=dtype)
        for i in range(n_bits):
            arr[:, i] = (n_int%2==1).astype(int)
            n_int = n_int / 2
            n_int = n_int.astype(np.int8)
        return arr



    def make_obs(self, resp=None, get_shape=False):
        # CORE_DIM = 38
        CORE_DIM = 23
        assert ScenarioConfig.obs_vec_length == CORE_DIM
        if get_shape:
            return CORE_DIM

        # temporary parameters
        OBS_RANGE_PYTHON_SIDE = 15000
        MAX_NUM_OPP_OBS = 5
        MAX_NUM_ALL_OBS = 5
        
        # get and calculate distance array
        pos3d_arr = np.zeros(shape=(self.n_agents, 3), dtype=np.float32)
        for i, agent in enumerate(self.agents): pos3d_arr[i] = agent.pos3d
        # use the distance matrix calculated by unreal engine to accelerate
        # dis_mat = distance_matrix(pos3d_arr)    # dis_mat is a matrix, shape = (n_agent, n_agent)
        dis_mat = resp['dataGlobal']['distanceMat']
        alive_all = np.array([agent.alive for agent in self.agents])
        dis_mat[~alive_all,:] = +np.inf
        dis_mat[:,~alive_all] = +np.inf
        # get team list
        team_belonging = np.array([agent.team for agent in self.agents])

        # gather the obs arr of all known agents
        obs_arr = RawObsArray(key='Agent')

        if not hasattr(self, "uid_binary"):
            self.uid_binary = self.get_binary_array(np.arange(self.n_agents), 10)

        for i, agent in enumerate(self.agents):
            assert agent.location is not None
            assert agent.uid == i

            obs_arr.append(
                self.uid_binary[i]  # 0~9
            )
            obs_arr.append([
                agent.index,    # 10
                agent.team,     # 11
                agent.alive,    # 12
                agent.uid_remote, # 13
            ])
            obs_arr.append( #[14,15,16,17,18,19]
                agent.pos3d
                # tear_num_arr(agent.pos3d, n_digits=6, base=10, mv_left=0)
                # tear_num_arr(agent.pos3d, 6, ScenarioConfig.ObsBreakBase, 0) # 3 -- > 3*6 = 18 , 18-3=15, 23+15 = 38
            )
            obs_arr.append(
                agent.vel3d
            )
            obs_arr.append([
                agent.hp,
                agent.yaw,
                agent.max_speed,
            ])
        obs_ = obs_arr.get()
        new_obs = my_view(obs_, [self.n_agents, -1])

        assert CORE_DIM == new_obs.shape[-1]
        OBS_ALL_AGENTS = np.zeros(shape=(self.n_agents, MAX_NUM_OPP_OBS+MAX_NUM_ALL_OBS, CORE_DIM))

        # now arranging the individual obs
        for i, agent in enumerate(self.agents):
            if not agent.alive:
                OBS_ALL_AGENTS[i, :] = np.nan
                continue

            # if alive
            # scope <all>
            dis2all = dis_mat[i, :]
            is_ally = (team_belonging == agent.team)

            # scope <opp/hostile>
            a2h_dis = dis2all[~is_ally]
            h_alive = alive_all[~is_ally]
            h_feature = new_obs[~is_ally]
            h_iden_sort  = np.argsort(a2h_dis)[:MAX_NUM_OPP_OBS]
            a2h_dis_sorted = a2h_dis[h_iden_sort]
            h_alive_sorted = h_alive[h_iden_sort]
            h_vis_mask = (a2h_dis_sorted <= OBS_RANGE_PYTHON_SIDE) & h_alive_sorted
            
            # scope <all>
            h_vis_index = h_iden_sort[h_vis_mask]
            h_invis_index = h_iden_sort[~h_vis_mask]
            h_vis_index, h_invis_index = self.item_random_mv(src=h_vis_index, dst=h_invis_index,prob=0, rand=True)
            h_ind = np.concatenate((h_vis_index, h_invis_index))
            h_msk = np.concatenate((h_vis_index<0, h_invis_index>=0)) # "<0" project to False; ">=0" project to True
            a2h_feature_sort = h_feature[h_ind]
            a2h_feature_sort[h_msk] = 0
            if len(a2h_feature_sort)<MAX_NUM_OPP_OBS:
                a2h_feature_sort = np.concatenate((
                    a2h_feature_sort, 
                    np.ones(shape=(MAX_NUM_OPP_OBS-len(a2h_feature_sort), CORE_DIM))+np.nan), axis=0)

            # scope <ally/friend>
            a2f_dis = dis2all[is_ally]
            f_alive = alive_all[is_ally]
            f_feature = new_obs[is_ally]
            f_iden_sort  = np.argsort(a2f_dis)[:MAX_NUM_ALL_OBS]
            a2f_dis_sorted = a2f_dis[f_iden_sort]
            f_alive_sorted = f_alive[f_iden_sort]
            f_vis_mask = (a2f_dis_sorted <= OBS_RANGE_PYTHON_SIDE) & f_alive_sorted

            # scope <all>
            f_vis_index = f_iden_sort[f_vis_mask]
            self_vis_index = f_vis_index[:1] # seperate self and ally
            f_vis_index = f_vis_index[1:]    # seperate self and ally
            f_invis_index = f_iden_sort[~f_vis_mask]
            f_vis_index, f_invis_index = self.item_random_mv(src=f_vis_index, dst=f_invis_index,prob=0, rand=True)
            f_ind = np.concatenate((self_vis_index, f_vis_index, f_invis_index))
            f_msk = np.concatenate((self_vis_index<0, f_vis_index<0, f_invis_index>=0)) # "<0" project to False; ">=0" project to True
            self_ally_feature_sort = f_feature[f_ind]
            self_ally_feature_sort[f_msk] = 0
            if len(self_ally_feature_sort)<MAX_NUM_ALL_OBS:
                self_ally_feature_sort = np.concatenate((
                    self_ally_feature_sort, 
                    np.ones(shape=(MAX_NUM_ALL_OBS-len(self_ally_feature_sort), CORE_DIM))+np.nan
                ), axis=0)
            OBS_ALL_AGENTS[i,:] = np.concatenate((self_ally_feature_sort, a2h_feature_sort), axis = 0)


        # the last part of observation is the list of core game objects
        MAX_OBJ_NUM_ACCEPT = 0
        self.N_Obj = len(self.key_obj)
        if MAX_OBJ_NUM_ACCEPT!=0:
            OBJ_UID_OFFSET = 32768
            obs_arr = RawObsArray(key = 'GameObj')
            for i, obj in enumerate(self.key_obj):
                assert obj['uId'] - OBJ_UID_OFFSET == i
                obs_arr.append(
                    -self.uid_binary[i] # reverse uid binary, self.uid_binary[i]
                )
                obs_arr.append([
                    obj['uId'] - OBJ_UID_OFFSET,    #agent.index,
                    -1,                             #agent.team,
                    True,                           #agent.alive,
                    obj['uId'] - OBJ_UID_OFFSET,    #agent.uid_remote,
                ])
                # tear_num_arr(agent.pos3d, n_digits=6, base=10, mv_left=0)
                obs_arr.append(
                    [
                        obj['location']['x'], obj['location']['y'], obj['location']['z']  # agent.pos3d
                    ]
                    # tear_num_arr([
                    #     obj['location']['x'], obj['location']['y'], obj['location']['z']  # agent.pos3d
                    # ], 6, ScenarioConfig.ObsBreakBase, 0)
                )
                
                obs_arr.append([
                    obj['velocity']['x'], obj['velocity']['y'], obj['velocity']['z']  # agent.vel3d
                ]+
                [
                    -1,                         # hp
                    obj['rotation']['yaw'],     # yaw 
                    0,                          # max_speed
                ])
            OBS_GameObj = my_view(obs_arr.get(), [len(self.key_obj), -1])[:MAX_OBJ_NUM_ACCEPT, :]
            OBS_GameObj = repeat_at(OBS_GameObj, insert_dim=0, n_times=self.n_agents)
            OBS_ALL_AGENTS = np.concatenate((OBS_ALL_AGENTS, OBS_GameObj), axis=1)

        return OBS_ALL_AGENTS


