import json, os, subprocess, time, copy
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
        ##################################################
        ################ spawn group 1 ###################
        ##################################################
        AgentSettingArray = []
        agent_uid_cnt = 0
        for i in range(ScenarioConfig.n_team1agent):
            x = 0 + 100*(i - ScenarioConfig.n_team1agent//2)
            y = -3000 + 500* (i%6)
            assert np.abs(x) < 15000.0
            assert np.abs(y) < 15000.0
            # 500 is slightly above the ground (depending the map you have built), 
            # but agent will be spawn to ground automatically
            z = 500 
            agent_property = copy.deepcopy(AgentPropertyDefaults)
            agent_property.update({
                    'ClassName': 'RLA_CAR_Laser',   # FString ClassName = "";
                    'AcceptRLControl': True,    # bool AcceptRLControl = 0;
                    'AgentTeam': 0, # int AgentTeam = 0;
                    'IndexInTeam': i,   # int IndexInTeam = 0;
                    'UID': agent_uid_cnt,   # int UID = 0;
                    'MaxMoveSpeed': 600,
                    'AgentHp':100,
                    'Color':'(R=1,G=1,B=0,A=1)',
                    'InitLocation': { 'x': x,  'y': y, 'z': z, },
            }),
            AgentSettingArray.append(agent_property); agent_uid_cnt += 1

        ##################################################
        ################ spawn group 2 ###################
        ##################################################
        for i in range(ScenarioConfig.n_team2agent):
            x = 0    + 100*(i - ScenarioConfig.n_team2agent//2)
            y = 3000 + 500* (i%6)
            assert np.abs(x) < 15000.0
            assert np.abs(y) < 15000.0
            # 500 is slightly above the ground, but agent will be spawn to ground automatically
            z = 500 
            agent_property = copy.deepcopy(AgentPropertyDefaults)
            agent_property.update({
                    'ClassName': 'RLA_CAR_Laser',
                    'AcceptRLControl': False,
                    'AgentTeam': 1,
                    'IndexInTeam': i,
                    'UID': agent_uid_cnt,
                    'MaxMoveSpeed': 600,
                    'AgentHp':100,
                    'Color':'(R=0,G=0,B=1,A=1)',
                    'InitLocation': { 'x': x, 'y': y, 'z': z, },
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

        # # The agents need to get some 'rest' after spawn (updating the perception)
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
        RewardForAllTeams = 0
        done = resp['dataGlobal']['episodeDone']
        if resp['dataGlobal']['timeCnt'] >= ScenarioConfig.MaxEpisodeStep:
            done = True

        return (ob, RewardForAllTeams,  done, info)  # choose this if RewardAsUnity



    def step_skip(self):
        return self.client.send_and_wait_reply(json.dumps({
            'valid': True,
            'DataCmd': 'skip_frame',
        }))




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
        MAX_NUM_OPP_OBS = 16
        MAX_NUM_ALL_OBS = 16

        # get and calculate distance array
        pos3d_arr = np.zeros(shape=(self.n_agents, 3), dtype=np.float32)
        for i, agent in enumerate(self.agents): pos3d_arr[i] = agent.pos3d
        dis_mat = distance_matrix(pos3d_arr)    # dis_mat is a matrix, shape = (n_agent, n_agent)
        alive_all = [agent.alive for agent in self.agents]
        dis_mat[~alive_all,:] = +np.inf
        dis_mat[:,~alive_all] = +np.inf
        # IN_VISUAL_PYTHON_SIDE = dis_mat < OBS_RANGE_PYTHON_SIDE

        # get team list
        team_belonging = np.zeros([agent.team for agent in self.agents])

        # gather the obs arr of all known agents
        obs_arr = raw_obs_array()
        for i, agent in enumerate(self.agents):
            assert agent.location is not None
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

            ))
        # now arranging the individual obs
        for i, agent in enumerate(self.agents):

            if not agent.alive:
                agent_obs = np.zeros(shape=(self.agent_emb.shape[-1] *vis_n*2,))
                return agent_obss

            dis2all = dis_mat[i, :]
            is_ally = (team_belonging == agent.team)

            dis2ally = dis2all[is_ally]
            dis2opp = dis2all[~is_ally]

        # # now, new_obs is a matrix of shape (n_agent, core_dim)
        # from UTILS.tensor_ops import repeat_at
        # encoded_obs_all_agent = repeat_at(encoded_obs, insert_dim=0, n_times=self.n_agents)
        return encoded_obs_all_agent