import json, os, subprocess, time, copy
import numpy as np
from UTILS.colorful import print紫, print靛
from UTILS.config_args import ChainVar
from ...common.base_env import BaseEnv
from ..actset_lookup import digit2act_dictionary
from ..agent import Agent
from ..uhmap_env_wrapper import UhmapEnv, ScenarioConfig

DEBUG = True


AgentPropertyDefaults = {
    'ClassName': 'RLA_CAR',     # FString ClassName = "";
    'AcceptRLControl': True,    # bool AcceptRLControl = 0;
    'AgentTeam': 0,             # int AgentTeam = 0;
    'IndexInTeam': 0,           # int IndexInTeam = 0;
    'UID': 0,                   # int UID = 0;
    'MaxMoveSpeed': 600,
    'InitLocation': { 'x': 0,  'y': 0, 'z': 0, },
    'InitRotation': { 'x': 0,  'y': 0, 'z': 0, },
    'AgentScale': { 'x': 1,  'y': 1, 'z': 1, },
    'InitVelocity': { 'x': 0,  'y': 0, 'z': 0, },
    'AgentHp':100,
    "WeaponCD": 1,
    "IsTeamReward": True,
    "Type": "",
    'RSVD1':'(R=0,G=1,B=0,A=1)',
    'RSVD2':'',
}

class UhmapLargeScale(UhmapEnv):
    def __init__(self, rank) -> None:
        super().__init__(rank)

    def reset(self):
        self.t = 0
        ##################################################
        ##################################################
        ################ spawn group 1 ###################
        ##################################################
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
                    'RSVD1':'(R=1,G=0,B=0,A=1)',
                    'InitLocation': { 'x': x,  'y': y, 'z': z, },
            }),
            AgentSettingArray.append(agent_property); agent_uid_cnt += 1


        # refer to struct.cpp, FParsedDataInput
        json_to_send = json.dumps({
            'valid': True,
            'DataCmd': 'reset',
            'NumAgents' : ScenarioConfig.n_team1agent,
            'AgentSettingArray': AgentSettingArray,  # refer to struct.cpp, FAgentProperty
            'TimeStepMax': ScenarioConfig.MaxEpisodeStep,
            'TimeStep' : 0,
            'Actions': None,
        })
        resp = self.client.send_and_wait_reply(json_to_send)
        resp = json.loads(resp)

        ##################################################
        ##################################################
        ################ spawn group 2 ###################
        ##################################################
        ##################################################
        AgentSettingArray = []
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
                    'RSVD1':'(R=0,G=0,B=1,A=1)',
                    'InitLocation': { 'x': x, 'y': y, 'z': z, },
                }),
            AgentSettingArray.append(agent_property); agent_uid_cnt += 1

        # refer to struct.cpp, FParsedDataInput
        json_to_send = json.dumps({
            'valid': True,
            'DataCmd': 'reset_spawn',
            'NumAgents' : ScenarioConfig.n_team1agent,
            'AgentSettingArray': AgentSettingArray,  # refer to struct.cpp, FAgentProperty
            'TimeStepMax': ScenarioConfig.MaxEpisodeStep,
            'TimeStep' : 0,
            'Actions': None,
        })
        resp = self.client.send_and_wait_reply(json_to_send)
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

        tic = time.time()
        resp = json.loads(resp)
        toc = time.time()
        print(toc-tic)


        ob, info = self.parse_response_ob_info(resp)
        RewardForAllTeams = 0
        done = resp['dataGlobal']['episodeDone']
        if resp['dataGlobal']['timeCnt'] >= ScenarioConfig.MaxEpisodeStep:
            done = True
        if DEBUG:
            if done:
                print紫(resp['dataGlobal'])
            else:
                print(resp['dataGlobal'])
        return (ob, RewardForAllTeams,  done, info)  # choose this if RewardAsUnity