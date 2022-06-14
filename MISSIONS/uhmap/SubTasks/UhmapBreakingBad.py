import json, os, subprocess, time
import numpy as np
from UTILS.colorful import print紫, print靛
from UTILS.network import TcpClientP2P
from UTILS.config_args import ChainVar
from ...common.base_env import BaseEnv
from ..actset_lookup import digit2act_dictionary
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
            x = -5360.0 + 500*i
            y = 9830.0
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
                    'AgentHp':2000,
                    'InitLocation': { 'x': x,  'y': y, 'z': z, },
                },
            ); agent_uid_cnt += 1

        x = -5360.0 + 1000
        y = 9830.0
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
                'AgentHp':2000,
                'InitLocation': { 'x': x,  'y': y, 'z': z, },
            },
        ); agent_uid_cnt += 1
        


        for i in range(ScenarioConfig.n_team2agent):
            x = -6800.0 + 500*(i+1)  *  (-1)**(i+1)
            y = 6830.0
            # 500 is slightly above the ground, but agent will be spawn to ground automatically
            z = 500 
            AgentSettingArray.append(
                {
                    'ClassName': 'DroneGroundOpp',
                    'AcceptRLControl': False,
                    'AgentTeam': 1,
                    'IndexInTeam': i,
                    'UID': agent_uid_cnt,
                    'MaxMoveSpeed': 600,
                    'AgentHp':29,
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
        act_send = [digit2act_dictionary[a] for a in act] + \
                   ['ActionSet2::N/A;N/A' for _ in range(ScenarioConfig.n_team2agent)]
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