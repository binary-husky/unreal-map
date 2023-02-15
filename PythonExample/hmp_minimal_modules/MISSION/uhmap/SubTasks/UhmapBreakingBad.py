import json, os, subprocess, time, copy, re, inspect, os
import numpy as np
from UTIL.tensor_ops import my_view, repeat_at, distance_mat_between
from ...common.base_env import RawObsArray
from ..actset_lookup import digit2act_dictionary, AgentPropertyDefaults
from ..actset_lookup import decode_action_as_string, decode_action_as_string
from ..agent import Agent
from ..uhmap_env_wrapper import UhmapEnv, ScenarioConfig

class SubTaskConfig():
    empty = ""

class UhmapBreakingBad(UhmapEnv):
    def __init__(self, rank) -> None:
        super().__init__(rank)
        self.observation_space = self.make_obs(get_shape=True)
        self.SubTaskConfig = SubTaskConfig
        assert os.path.basename(inspect.getfile(SubTaskConfig)) == type(self).__name__+'Conf.py', \
                ('make sure you have imported the correct SubTaskConfig class')
                
    def reset(self):
        super().reset()
        
        self.t = 0

        AgentPropertyDefaults.update({
            'MaxMoveSpeed': 600,
            'AgentScale'  : { 'x': 1,  'y': 1, 'z': 1, },     # also influence object mass, please change it with causion!
            "DodgeProb": 0.0,           # probability of escaping dmg 闪避概率, test ok
            "ExplodeDmg": 20,           # ms explode dmg. test ok
        })

        # 500 is slightly above the ground,
        # but agent will be spawn to ground automatically
        ####################### spawn all ###########################
        AgentSettingArray = []
        agent_uid_cnt = 0
        # "N_AGENT_EACH_TEAM": [10, 10], // update N_AGENT_EACH_TEAM
        for i in range(ScenarioConfig.N_AGENT_EACH_TEAM[0]-1):  # For attacking, drones on the ground
            x = 3254.0
            y = 3891.0 + i *100
            z = 500 
            agent_property = copy.deepcopy(AgentPropertyDefaults)
            agent_property.update({
                    'ClassName': 'RLA_CAR',         # FString ClassName = "";
                    'AgentTeam': 0,                 # int AgentTeam = 0;
                    'IndexInTeam': i,               # int IndexInTeam = 0;
                    'UID': agent_uid_cnt,           # int UID = 0;
                    'MaxMoveSpeed': 600,
                    "ExplodeDmg": 10,
                    "DodgeProb": 0.1,
                    'AgentHp': 100,
                    "WeaponCD": 1,
                    'Color':'(R=0,G=1,B=0,A=1)',
                    'InitLocation': { 'x': x,  'y': y, 'z': z, },
                })
            AgentSettingArray.append(agent_property); agent_uid_cnt += 1


        x = 4000.0
        y = 4000.0
        z = 1000
        agent_property = copy.deepcopy(AgentPropertyDefaults)
        agent_property.update({
                'ClassName': 'RLA_UAV_VIP',         # FString ClassName = "";
                'AgentTeam': 0,                 # int AgentTeam = 0;
                'IndexInTeam': agent_uid_cnt,   # under most situations IndexInTeam=agent_uid_cnt for team 0
                'UID': agent_uid_cnt,           # int UID = 0;
                'MaxMoveSpeed': 1000,
                "DodgeProb": 0.5,
                "ExplodeDmg": 10,
                'AgentHp': 1,
                "WeaponCD": 10000000000,
                'Color':'(R=0,G=1,B=0,A=1)',
                'InitLocation': { 'x': x,  'y': y, 'z': z, },
        })
        AgentSettingArray.append(agent_property); agent_uid_cnt += 1

        # "N_AGENT_EACH_TEAM": [10, 10], // update N_AGENT_EACH_TEAM
        for i in range(ScenarioConfig.N_AGENT_EACH_TEAM[1]):
            x = 0 + 500*(i+1)  *  (-1)**(i+1)
            y = 0
            z = 500
            agent_property = copy.deepcopy(AgentPropertyDefaults)
            agent_property.update({
                    'ClassName': 'RLA_CAR_RED',
                    'AgentTeam': 1,
                    'IndexInTeam': i,
                    'UID': agent_uid_cnt,
                    'MaxMoveSpeed': 700,
                    "DodgeProb": 0.1,
                    'AgentHp':100,
                    "ExplodeDmg": 10,
                    "WeaponCD": 0.5,
                    'Color':'(R=1,G=0,B=0,A=1)',
                    'InitLocation': { 'x': x, 'y': y, 'z': z, },
            })
            AgentSettingArray.append(agent_property); agent_uid_cnt += 1

        # refer to struct.cpp, FParsedDataInput
        resp = self.client.send_and_wait_reply(json.dumps({
            'valid': True,
            'DataCmd': 'reset',
            'AgentSettingArray': AgentSettingArray,  # refer to struct.cpp, FAgentProperty
            'TimeStepMax': ScenarioConfig.MaxEpisodeStep,
            'TimeStep' : 0,
            'Actions': None,
        }))
        resp = json.loads(resp)
        # make sure the map (level in UE) is correct
        assert resp['dataGlobal']['levelName'] == 'UhmapBreakingBad'
        assert len(resp['dataArr']) == len(AgentSettingArray)
        return self.parse_response_ob_info(resp)


    def step(self, act):

        assert len(act) == self.n_agents

        # translate actions to the format recognized by unreal engine
        if ScenarioConfig.ActionFormat == 'Single-Digit':
            act_send = [digit2act_dictionary[a] for a in act]
        elif ScenarioConfig.ActionFormat == 'Multi-Digit':
            act_send = [decode_action_as_string(a) for a in act]

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
        if not hasattr(self, 'pattern'): self.pattern = re.compile(r'<([^<>]*)>([^<>]*)')
        return {k:v for k,v  in re.findall(self.pattern, event)}

    def extract_key_gameobj(self, resp):
        keyObjArr = resp['dataGlobal']['keyObjArr']
        return keyObjArr


    def gen_reward_and_win(self, resp):
        WIN_OR_LOSE_REWARD = 5
        DRAW_REWARD = 2.5
        KILL_REWARD = 0.1
        BE_KILLED_REWARD = 0

        reward = np.array([0.0]*self.n_teams,dtype=float)
        events = resp['dataGlobal']['events']
        WinningResult = None

        # reward according to distance to either of the landmarks
        landmarks_pos3darr = np.array([[
            lm['location']['x'], lm['location']['y'], lm['location']['z']
            ] for lm in resp['dataGlobal']['keyObjArr']])
        agent_pos3darr = np.array([agent.pos3d for agent in self.agents])
        res = distance_mat_between(agent_pos3darr, landmarks_pos3darr)
        penalty = -np.min(res, axis = -1) / 100000
        reward += np.array([sum(penalty[ ScenarioConfig.AGENT_ID_EACH_TEAM[i] ]) for i in range(self.n_teams)])

        # reward according to event (including win or lose event)
        for event in events: 
            event_parsed = self.parse_event(event)
            
            if event_parsed['Event'] == 'Destroyed':
                team = self.find_agent_by_uid(event_parsed['UID']).team
                reward[team]    -= BE_KILLED_REWARD     # this team
                reward[1-team]  += KILL_REWARD          # opp team

            if event_parsed['Event'] == 'EndEpisode':
                # print([a.alive * a.hp for a in self.agents])
                EndReason = event_parsed['EndReason']
                WinTeam = int(event_parsed['WinTeam'])
                if WinTeam<0: # end due to timeout
                    WinTeam = 1

                if WinTeam >= 0:
                    WinningResult = {
                        "team_ranking": [0,1] if WinTeam==0 else [1,0],
                        "end_reason": EndReason
                    }
                    reward[WinTeam] += WIN_OR_LOSE_REWARD
                    reward[1-WinTeam] -= WIN_OR_LOSE_REWARD
                else:
                    WinningResult = {
                        "team_ranking": [-1, -1],
                        "end_reason": EndReason
                    }
                    reward = [-DRAW_REWARD for _ in range(self.n_teams)]
        # print(reward)
        return reward, WinningResult

    def step_skip(self):
        return self.client.send_and_wait_reply(json.dumps({
            'valid': True,
            'DataCmd': 'skip_frame',
        }))


    def find_agent_by_uid(self, uid):
        if not hasattr(self, 'uid_to_agent_dict'):
            self.uid_to_agent_dict = {}
            self.uid_to_agent_dict.update({agent.uid:agent for agent in self.agents}) 
            if isinstance(uid, str):
                self.uid_to_agent_dict.update({str(agent.uid):agent for agent in self.agents}) 
        return self.uid_to_agent_dict[uid]



    def parse_response_ob_info(self, resp):
        assert resp['valid']
        if len(resp['dataGlobal']['events'])>0:
            tmp = [kv.split('>') for kv in resp['dataGlobal']['events'][0].split('<') if kv]
            info_parse = {t[0]:t[1] for t in tmp}
            # print('pass')
        info_dict = resp
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
        pointer = 0
        for i in range(n_bits):
            arr[:, i] = (n_int%2==1).astype(int)
            n_int = n_int / 2
            n_int = n_int.astype(np.int8)
        return arr



    def make_obs(self, resp=None, get_shape=False):
        CORE_DIM = 23
        assert ScenarioConfig.obs_vec_length == CORE_DIM
        if get_shape:
            return CORE_DIM

        # temporary parameters
        OBS_RANGE_PYTHON_SIDE = 1500
        MAX_NUM_OPP_OBS = 5
        MAX_NUM_ALL_OBS = 4
        
        # get and calculate distance array
        pos3d_arr = np.zeros(shape=(self.n_agents, 3), dtype=np.float32)
        for i, agent in enumerate(self.agents): pos3d_arr[i] = agent.pos3d
        # use the distance matrix calculated by unreal engine to accelerate
        # dis_mat = distance_matrix(pos3d_arr)    # dis_mat is a matrix, shape = (n_agent, n_agent)
        dis_mat = np.array(resp['dataGlobal']['distanceMat']['flat_arr']).reshape(self.n_agents,self.n_agents)

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
                self.uid_binary[i]
            )
            obs_arr.append([
                agent.index,
                agent.team,
                agent.alive,
                agent.uid_remote,
            ])
            obs_arr.append(
                agent.pos3d
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
        OBS_ALL_AGENTS = np.zeros(shape=(
            self.n_agents, 
            MAX_NUM_OPP_OBS+MAX_NUM_ALL_OBS, 
            CORE_DIM
            ))

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
                    np.ones(shape=(MAX_NUM_OPP_OBS-len(a2h_feature_sort), CORE_DIM))+np.nan
                ), axis=0)

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
        MAX_OBJ_NUM_ACCEPT = 5
        self.N_Obj = len(self.key_obj)

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
            ]+
            [
                obj['location']['x'], obj['location']['y'], obj['location']['z']  # agent.pos3d
            ]+
            [
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

'''
            obs_arr.append(
                self.uid_binary[i]
            )
            obs_arr.append([
                agent.index,
                agent.team,
                agent.alive,
                agent.uid_remote,
            ])
            obs_arr.append(
                agent.pos3d
            )
            obs_arr.append(
                agent.vel3d
            )
            obs_arr.append([
                agent.hp,
                agent.yaw,
                agent.max_speed,
            ])


'''