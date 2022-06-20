import json, os, subprocess, time
import numpy as np
from UTILS.colorful import print紫, print靛
from UTILS.network import TcpClientP2P
from UTILS.config_args import ChainVar
from ...common.base_env import BaseEnv
from ..actset_lookup import digit2act_dictionary, decode_action_as_string
from ..agent import Agent
from ..uhmap_env_wrapper import UhmapEnv, ScenarioConfig

DEBUG = True

class UhmapBreakingBad(UhmapEnv):
    def __init__(self, rank) -> None:
        super().__init__(rank)

    def reset(self):
        self.t = 0
        AgentSettingArray = []
        agent_uid_cnt = 0
        for i in range(ScenarioConfig.n_team1agent-1):
            #x = 2249.0 + 500*i
            #y = 4911.0
            x = 0 + 500*i
            y = 2500
            # 500 is slightly above the ground (depending the map you have built), 
            # but agent will be spawn to ground automatically
            z = 500 
            AgentSettingArray.append(
                {
                    'ClassName': 'RLA_CAR',   # FString ClassName = "";
                    'AcceptRLControl': True,    # bool AcceptRLControl = 0;
                    'AgentTeam': 0, # int AgentTeam = 0;
                    'IndexInTeam': i,   # int IndexInTeam = 0;
                    'UID': agent_uid_cnt,   # int UID = 0;
                    'MaxMoveSpeed': 600,
                    'AgentHp': 100,
                    "WeaponCD": 0.5,
                    'RSVD1':'(R=0,G=1,B=0,A=1)',
                    'InitLocation': { 'x': x,  'y': y, 'z': z, },
                },
            ); agent_uid_cnt += 1

        #x = 2249.0 + 1500
        #y = 4911.0
        x = -1500
        y = 2500
        # 500 is slightly above the ground (depending the map you have built), 
        # but agent will be spawn to ground automatically
        z = 2000
        AgentSettingArray.append(
            {
                'ClassName': 'RLA_UAV',   # FString ClassName = "";
                'AcceptRLControl': True,    # bool AcceptRLControl = 0;
                'AgentTeam': 0, # int AgentTeam = 0;
                'IndexInTeam': i,   # int IndexInTeam = 0;
                'UID': agent_uid_cnt,   # int UID = 0;
                'MaxMoveSpeed': 600,
                'AgentHp':1000,
                "WeaponCD": 10000000,
                'RSVD1':'(R=0,G=1,B=0,A=1)',
                'InitLocation': { 'x': x,  'y': y, 'z': z, },
            },
        ); agent_uid_cnt += 1
        


        for i in range(ScenarioConfig.n_team2agent):
            #x = 766.0 + 500*(i+1)  *  (-1)**(i+1)
            #y = 595.0
            x = 0 + 500*(i+1)  *  (-1)**(i+1)
            y = -3000
            # 500 is slightly above the ground, but agent will be spawn to ground automatically
            z = 500 
            AgentSettingArray.append(
                {
                    'ClassName': 'RLA_CAR_RED',
                    'AcceptRLControl': False,
                    'AgentTeam': 1,
                    'IndexInTeam': i,
                    'UID': agent_uid_cnt,
                    'MaxMoveSpeed': 600,
                    'AgentHp':100,
                    "WeaponCD": 0.5,
                    'RSVD1':'(R=1,G=0,B=0,A=1)',
                    'InitLocation': { 'x': x, 'y': y, 'z': z, },
                },
            ); agent_uid_cnt += 1

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
        return self.parse_response_ob_info(resp)



    def step(self, act):

        assert len(act) == self.n_agents
        act_send = [decode_action_as_string(a) for a in act]
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
        if DEBUG:
            if done:
                print紫(resp['dataGlobal'])
            else:
                print(resp['dataGlobal'])
        return (ob, RewardForAllTeams,  done, info)  # choose this if RewardAsUnity