import json, copy, re, os, inspect, os
import numpy as np
from UTIL.tensor_ops import my_view, repeat_at
from ...common.base_env import RawObsArray
from ..actionset_v3 import digitsToStrAction
from ..agent import Agent
from ..uhmap_env_wrapper import UhmapEnv, ScenarioConfig
from .cython_func import tear_num_arr
from ..actset_lookup import digit2act_dictionary, AgentPropertyDefaults
from ..actset_lookup import decode_action_as_string, decode_action_as_string



class UhmapCommonFn(UhmapEnv):

    def reset(self):
        """
            Reset function, it delivers reset command to unreal engine to spawn all agents
            环境复位,每个episode的开始会执行一次此函数中会初始化所有智能体
        """
        super().reset()
        self.t = 0
        pos_ro = np.random.rand()*2*np.pi
        # spawn agents
        AgentSettingArray = []

        # count the number of agent in each team
        n_team_agent = {}
        for i, agent_info in enumerate(self.SubTaskConfig.agent_list):
            team = agent_info['team']
            if team not in n_team_agent: n_team_agent[team] = 0
            self.SubTaskConfig.agent_list[i]['uid'] = i
            self.SubTaskConfig.agent_list[i]['tid'] = n_team_agent[team]
            n_team_agent[team] += 1

        # push agent init info one by one
        for i, agent_info in enumerate(self.SubTaskConfig.agent_list):
            team = agent_info['team']
            agent_info['n_team_agent'] = n_team_agent[team]
            init_fn = getattr(self, agent_info['init_fn_name'])
            AgentSettingArray.append(init_fn(agent_info, pos_ro))

        self.agents  = [Agent(team=a['team'], team_id=a['tid'], uid=a['uid']) for a in self.SubTaskConfig.agent_list]
        
        # refer to struct.cpp, FParsedDataInput
        resp = self.client.send_and_wait_reply(json.dumps({
            'valid': True,
            'DataCmd': 'reset',
            'NumAgents' : len(self.SubTaskConfig.agent_list),
            'AgentSettingArray': AgentSettingArray,  # refer to struct.cpp, FAgentProperty
            'TimeStepMax': ScenarioConfig.MaxEpisodeStep,
            'TimeStep' : 0,
            'Actions': None,
        }))
        resp = json.loads(resp)
        # make sure the map (level in UE) is correct
        # assert resp['dataGlobal']['levelName'] == 'UhmapLargeScale'

        assert len(resp['dataArr']) == len(AgentSettingArray), "Illegal agent initial position. 非法的智能体初始化位置，一部分智能体没有生成."
        return self.parse_response_ob_info(resp)


    def step(self, act):
        """
            step 函数,act中包含了所有agent的决策
        """
        assert len(act) == self.n_agents

        # translate actions to the format recognized by unreal engine
        if self.SubTaskConfig.ActionFormat == 'Single-Digit':
            act_send = [digit2act_dictionary[a] for a in act]
        elif self.SubTaskConfig.ActionFormat == 'Multi-Digit':
            act_send = [decode_action_as_string(a) for a in act]
        elif self.SubTaskConfig.ActionFormat == 'ASCII':            
            act_send = [digitsToStrAction(a) for a in act]
        else:
            act_send = [digitsToStrAction(a) for a in act]

        # simulation engine IO
        resp = json.loads(self.client.send_and_wait_reply(json.dumps({
            'valid': True,
            'DataCmd': 'step',
            'TimeStep': self.t,
            'Actions': None,
            'StringActions': act_send,
        })))

        # get obs for RL, info for script AI
        ob, info = self.parse_response_ob_info(resp)

        # generate reward, get the episode ending infomation
        RewardForAllTeams, WinningResult = self.gen_reward_and_win(resp)
        if WinningResult is not None: 
            info.update(WinningResult)
            assert resp['dataGlobal']['episodeDone']
            done = True
        else:
            done = False

        if resp['dataGlobal']['timeCnt'] >= ScenarioConfig.MaxEpisodeStep:
            assert done

        return (ob, RewardForAllTeams, done, info)  # choose this if RewardAsUnity

    def parse_event(self, event):
        """
            解析环境返回的一些关键事件,
            如智能体阵亡,某队伍胜利等等。
            关键事件需要在ue中进行定义.
            该设计极大地简化了python端奖励的设计流程,
            减小了python端的运算量。
        """
        if not hasattr(self, 'pattern'): self.pattern = re.compile(r'<([^<>]*)>([^<>]*)')
        return {k:v for k,v  in re.findall(self.pattern, event)}

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
                WaterdropWin = False
                WaterdropRank = False
                WaterdropReward = 0
                ShipWin = -1
                ShipRank = -1
                ShipReward = 0
                EndReason = event_parsed['EndReason']
                # According to MISSION\uhmap\SubTasks\UhmapWaterdropConf.py, team 0 is Ship team, team 1 is Waterdrop team
                if EndReason == "ShipNumLessThanTheshold" or EndReason == "Team_0_AllDead":
                    WaterdropWin = True; WaterdropRank = 0; WaterdropReward = 1
                    ShipWin = False; ShipRank = 1; ShipReward = -1
                elif EndReason == "TimeMaxCntReached" or EndReason == "Team_1_AllDead":
                    WaterdropWin = False; WaterdropRank = 1; WaterdropReward = -1
                    ShipWin = True; ShipRank = 0; ShipReward = 1
                else:
                    print('unexpected end reaon:', EndReason)
                    
                WinningResult = {"team_ranking": [ShipRank, WaterdropRank], "end_reason": EndReason}

                reward = [ShipReward, WaterdropReward]
        # print(reward)
        return reward, WinningResult

    def step_skip(self):
        """
            跳过一次决策,无用的函数
        """
        return self.client.send_and_wait_reply(json.dumps({
            'valid': True,
            'DataCmd': 'skip_frame',
        }))

    def find_agent_by_uid(self, uid):
        """
            用uid查找智能体(带缓存加速机制)
        """
        if not hasattr(self, 'uid_to_agent_dict'):
            self.uid_to_agent_dict = {}
            self.uid_to_agent_dict.update({agent.uid:agent for agent in self.agents}) 
            if isinstance(uid, str):
                self.uid_to_agent_dict.update({str(agent.uid):agent for agent in self.agents}) 
        return self.uid_to_agent_dict[uid]



    def parse_response_ob_info(self, resp):
        """
            粗解析智能体的观测,例如把死智能体的位置替换为inf(无穷远),
            将智能体的agentLocation从字典形式转变为更简洁的(x,y,z)tuple形式
        """
        assert resp['valid']
        resp['dataGlobal']['distanceMat'] = np.array(resp['dataGlobal']['distanceMat']['flat_arr']).reshape(self.n_agents,self.n_agents)
        
        if len(resp['dataGlobal']['events'])>0:
            tmp = [kv.split('>') for kv in resp['dataGlobal']['events'][0].split('<') if kv]
            info_parse = {t[0]:t[1] for t in tmp}

        info_dict = resp
        for info in info_dict['dataArr']: 
            alive = info['agentAlive']

            if alive:
                agentLocation = info.pop('agentLocation')
                agentRotation = info.pop('agentRotation')
                agentVelocity = info.pop('agentVelocity')
                agentScale = info.pop('agentScale')
                info['agentLocationArr'] = (agentLocation['x'], agentLocation['y'], agentLocation['z'])
                info['agentVelocityArr'] = (agentVelocity['x'], agentVelocity['y'], agentVelocity['z'])
                info['agentRotationArr'] = (agentRotation['yaw'], agentRotation['pitch'], agentRotation['roll'])
                info['agentScaleArr'] = (agentScale['x'], agentScale['y'], agentScale['z'])
                info.pop('previousAction')
                info.pop('availActions')
                # info.pop('rSVD1')
                info.pop('interaction')
            else:
                inf = float('inf')
                info['agentLocationArr'] = (inf, inf, inf)
                info['agentVelocityArr'] = (inf, inf, inf)
                info['agentRotationArr'] = (inf, inf, inf)

        info = resp['dataArr']
        for i, agent_info in enumerate(info):
            self.agents[i].update_agent_attrs(agent_info)

        self.key_obj = self.extract_key_gameobj(resp)

        # return ob, info
        return self.make_obs(resp), info_dict



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



    def init_ship(self, agent_info, pos_ro):
        agent_class = agent_info['type']
        team = agent_info['team']
        tid = agent_info['tid'] # tid 是智能体在队伍中的编号 
        uid = agent_info['uid'] # uid 是智能体在仿真中的唯一编号
        
        x = -2000
        y = (tid * 1000) # tid 是智能体在队伍中的编号 
        z = 500 # 

        agent_property = copy.deepcopy(self.SubTaskConfig.AgentPropertyDefaults)
        agent_property.update({
                'DebugAgent': False,
                # max drive/fly speed
                'MaxMoveSpeed':  500,
                # also influence object mass, please change it with causion!
                'AgentScale'  : { 'x': 1,  'y': 1, 'z': 1, },
                # team belonging
                'AgentTeam': team,
                # choose ue class to init
                'ClassName': agent_class,
                # custom args
                'RSVD1': '',
                # the rank of agent inside the team
                'IndexInTeam': tid, 
                # the unique identity of this agent in simulation system
                'UID': uid, 
                # show color
                'Color':'(R=0,G=1,B=0,A=1)' if team==0 else '(R=0,G=0,B=1,A=1)',
                # initial location
                'InitLocation': { 'x': x,  'y': y, 'z': z, },
                # initial facing direction et.al.
                'InitRotator': { 'pitch': 0,  'roll': 0, 'yaw': 0, },
        }),
        return agent_property



    def init_waterdrop(self, agent_info, pos_ro):
        agent_class = agent_info['type']
        team = agent_info['team']
        tid = agent_info['tid']
        uid = agent_info['uid']
        
        x = +2000
        y = (tid * 200)
        z = 500 # 

        agent_property = copy.deepcopy(self.SubTaskConfig.AgentPropertyDefaults)
        agent_property.update({
                'DebugAgent': False,
                # max drive/fly speed
                'MaxMoveSpeed':  1000,
                # also influence object mass, please change it with causion!
                'AgentScale'  : { 'x': 1,  'y': 1, 'z': 1, },
                # team belonging
                'AgentTeam': team,
                # choose ue class to init
                'ClassName': agent_class,
                # custom args
                'RSVD1': '-MyCustomArg1=abc -MyCustomArg2=12345',
                # the rank of agent inside the team
                'IndexInTeam': tid, 
                # the unique identity of this agent in simulation system
                'UID': uid, 
                # show color
                'Color':'(R=0,G=1,B=0,A=1)' if team==0 else '(R=0,G=0,B=1,A=1)',
                # initial location
                'InitLocation': { 'x': x,  'y': y, 'z': z, },
                # initial facing direction et.al.
                'InitRotator': { 'pitch': 0,  'roll': 0, 'yaw': 0, },
        }),
        return agent_property
