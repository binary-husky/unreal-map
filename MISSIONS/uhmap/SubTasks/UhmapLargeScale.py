import json, os, subprocess, time, copy, re
import string
import numpy as np
from UTILS.colorful import print紫, print靛
from UTILS.config_args import ChainVar
from UTILS.tensor_ops import my_view, distance_matrix
from ...common.base_env import BaseEnv
from ..actset_lookup import digit2act_dictionary, AgentPropertyDefaults
from ..agent import Agent
from ..uhmap_env_wrapper import UhmapEnv, ScenarioConfig

DEBUG = True

class raw_obs_array(object):
    raw_obs_size = -1   # shared
    def __init__(self):
        if self.raw_obs_size==-1:
            self.guards_group = []
            self.nosize = True
        else:
            self.guards_group = np.zeros(shape=(self.raw_obs_size), dtype=np.float32)
            self.nosize = False
            self.p = 0

    def append(self, buf):
        if self.nosize:
            self.guards_group.append(buf)
        else:
            L = len(buf)
            self.guards_group[self.p:self.p+L] = buf[:]
            self.p += L

    def get(self):
        if self.nosize:
            self.guards_group = np.concatenate(self.guards_group)
            self.raw_obs_size = len(self.guards_group)
        return self.guards_group
        
    def get_raw_obs_size(self):
        assert self.raw_obs_size > 0
        return self.raw_obs_size

class UhmapLargeScale(UhmapEnv):
    def __init__(self, rank) -> None:
        super().__init__(rank)

    def reset(self):
        self.t = 0

        AgentPropertyDefaults.update({
            'AcceptRLControl': True, 
            'ClassName': 'RLA_CAR_Laser',   # FString ClassName = "";
            'MaxMoveSpeed': 600,
            'AgentHp':100,
        })

        # 500 is slightly above the ground,
        # but agent will be spawn to ground automatically
        ####################### spawn all ###########################
        AgentSettingArray = []
        agent_uid_cnt = 0
        for which_team in range(2):
            n_team_agent = ScenarioConfig.n_team1agent if which_team==0 else ScenarioConfig.n_team2agent
            for i in range(n_team_agent):
                x = 0 + 100*(i - n_team_agent//2)
                y = -3000 + 500* (i%6) + 3000 * which_team
                z = 500
                yaw = 90 if which_team==0 else -90
                assert np.abs(x) < 15000.0 and np.abs(y) < 15000.0
                agent_property = copy.deepcopy(AgentPropertyDefaults)
                agent_property.update({
                        'AgentTeam': which_team,
                        'IndexInTeam': i, 
                        'UID': agent_uid_cnt, 
                        'Color':'(R=0,G=1,B=0,A=1)' if which_team==0 else '(R=0,G=0,B=1,A=1)',
                        'InitLocation': { 'x': x,  'y': y, 'z': z, },
                        'InitRotator': { 'pitch': 0,  'roll': 0, 'yaw': yaw, },
                }),
                AgentSettingArray.append(agent_property); agent_uid_cnt += 1


        # refer to struct.cpp, FParsedDataInput
        resp = self.client.send_and_wait_reply(json.dumps({
            'valid': True,
            'DataCmd': 'reset',
            'NumAgents' : ScenarioConfig.n_team1agent,
            'AgentSettingArray': AgentSettingArray,  # refer to struct.cpp, FAgentProperty
            'TimeStepMax': ScenarioConfig.MaxEpisodeStep,
            'TimeStep' : 0,
            'Actions': None,
        }))
        resp = json.loads(resp)

        # make sure the map (level in UE) is correct
        assert resp['dataGlobal']['rSVD1'] == 'UhmapLargeScale'

        # skip initial frames
        skip_n_init_frame = 1
        for _ in range(skip_n_init_frame):
            resp = self.step_skip()
        resp = json.loads(resp)

        return self.parse_response_ob_info(resp)


    def step(self, act):

        assert len(act) == self.n_agents
        act_send = [digit2act_dictionary[a] for a in act]
        json_to_send = json.dumps({
            'valid': True,
            'DataCmd': 'step',
            'TimeStep': self.t,
            'Actions': None,
            'StringActions': act_send,
        })
        resp = self.client.send_and_wait_reply(json_to_send)
        resp = json.loads(resp)


        ob, info = self.parse_response_ob_info(resp)
        RewardForAllTeams = self.gen_reward(resp)
        done = resp['dataGlobal']['episodeDone']
        if resp['dataGlobal']['timeCnt'] >= ScenarioConfig.MaxEpisodeStep:
            done = True

        return (ob, RewardForAllTeams,  done, info)  # choose this if RewardAsUnity

    def parse_event(self, event):
        if not hasattr(self, 'pattern'): self.pattern = re.compile(r'<([^<>]*)>([^<>]*)')
        return {k:v for k,v  in re.findall(self.pattern, event)}


    def gen_reward(self, resp):
        reward = [0]*self.n_teams
        events = resp['dataGlobal']['events']
        for event in events: 
            event_parsed = self.parse_event(event)
            if event_parsed['Event'] == 'Destroyed':
                team = self.find_agent_by_uid(event_parsed['UID']).team
                reward[team]    -= 1    # this team
        return reward

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



    def parse_response_ob_info(self, response):
        assert response['valid']
        if len(response['dataGlobal']['events'])>0:
            tmp = [kv.split('>') for kv in response['dataGlobal']['events'][0].split('<') if kv]
            info_parse = {t[0]:t[1] for t in tmp}
            # print('pass')
        info_dict = response
        info = response['dataArr']
        for i, agent_info in enumerate(info):
            self.agents[i].update_agent_attrs(agent_info)
        # return ob, info
        return self.make_obs(), info_dict



    def make_obs(self):
        # temporary parameters
        OBS_RANGE_PYTHON_SIDE = 2500
        MAX_NUM_OPP_OBS = 5
        MAX_NUM_ALL_OBS = 5

        # get and calculate distance array
        pos3d_arr = np.zeros(shape=(self.n_agents, 3), dtype=np.float32)
        for i, agent in enumerate(self.agents): pos3d_arr[i] = agent.pos3d
        dis_mat = distance_matrix(pos3d_arr)    # dis_mat is a matrix, shape = (n_agent, n_agent)
        alive_all = np.array([agent.alive for agent in self.agents])
        dis_mat[~alive_all,:] = +np.inf
        dis_mat[:,~alive_all] = +np.inf
        # IN_VISUAL_PYTHON_SIDE = dis_mat < OBS_RANGE_PYTHON_SIDE

        # get team list
        team_belonging = np.array([agent.team for agent in self.agents])

        # gather the obs arr of all known agents
        obs_arr = raw_obs_array()

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


        OBS_ALL_AGENTS = np.zeros(shape=(
            self.n_agents, 
            MAX_NUM_OPP_OBS+MAX_NUM_ALL_OBS, 
            new_obs.shape[-1]
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

            OBS_ALL_AGENTS[i,:] = np.concatenate((self_ally_feature_sort, a2h_feature_sort), axis = 0)

        # # now, new_obs is a matrix of shape (n_agent, core_dim)
        # from UTILS.tensor_ops import repeat_at
        # encoded_obs_all_agent = repeat_at(encoded_obs, insert_dim=0, n_times=self.n_agents)
        return OBS_ALL_AGENTS


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
            arr[:, i] = (n_int%2==1).astype(np.int)
            n_int = n_int / 2
            n_int = n_int.astype(np.int8)
        return arr
