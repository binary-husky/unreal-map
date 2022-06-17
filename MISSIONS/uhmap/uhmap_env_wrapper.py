import json, os, subprocess, time
import numpy as np
from UTILS.colorful import print紫, print靛
from UTILS.network import TcpClientP2P
from UTILS.config_args import ChainVar
from ..common.base_env import BaseEnv
from .actset_lookup import digit2act_dictionary, agent_json2local_attrs
from .actset_lookup import act2digit_dictionary, no_act_placeholder, dictionary_n_actions
from .agent import Agent

# please register this ScenarioConfig into MISSIONS/env_router.py
class ScenarioConfig(object):  
    '''
        ScenarioConfig: This config class will be 'injected' with new settings from JSONC.
        (E.g., override configs with ```python main.py --cfg example.jsonc```)
        (As the name indicated, ChainVars will change WITH vars it 'chained_with' during config injection)
        (please see UTILS.config_args to find out how this advanced trick works out.)
    '''
    n_team1agent = 5
    n_team2agent = 5

    # <Part 1> Needed by the hmp core #
    N_TEAM = 2

    N_AGENT_EACH_TEAM = [n_team1agent, n_team2agent]
    N_AGENT_EACH_TEAM_cv = ChainVar(lambda n1, n2: [n1, n2], chained_with=['n_team1agent', 'n_team2agent'])

    AGENT_ID_EACH_TEAM = [range(0,n_team1agent), range(0,n_team2agent)]
    AGENT_ID_EACH_TEAM_cv = ChainVar(lambda  n1, n2: [range(0,n1),range(0,n2)], chained_with=['n_team1agent', 'n_team2agent'])

    TEAM_NAMES = ['ALGORITHM.None->None',]

    '''
        ## If the length of action array == the number of teams, set ActAsUnity to True
        ## If the length of action array == the number of agents, set ActAsUnity to False
    '''
    ActAsUnity = False

    '''
        ## If the length of reward array == the number of agents, set RewardAsUnity to False
        ## If the length of reward array == 1, set RewardAsUnity to True
    '''
    RewardAsUnity = True

    '''
        ## If the length of obs array == the number of agents, set ObsAsUnity to False
        ## If the length of obs array == the number of teams, set ObsAsUnity to True
    '''
    ObsAsUnity = False

    # <Part 2> Needed by env itself #
    MaxEpisodeStep = 100
    render = False
    TcpAddr = '127.0.0.1'
    UhmapPort = 21051
    SubTaskSelection = 'UhmapBreakingBad'
    
    UhmapServerExe = 'F:/UHMP/Build/WindowsServer/UHMPServer.exe'
    UhmapRenderExe = ''
    TimeDilation = 1.25
    FrameRate = 30
    # must satisfy: (TimeDilation=1.25*n, FrameRate=30*n)
    FrameRate_cv = ChainVar(lambda TimeDilation: int(TimeDilation/1.25*30), chained_with=['TimeDilation'])
    UhmapStartCmd = []
    # <Part 3> Needed by some ALGORITHM #
    StateProvided = False
    AvailActProvided = False
    EntityOriented = False

    n_actions = dictionary_n_actions
    obs_vec_length = 7
    act2digit_dictionary = act2digit_dictionary

DEBUG = False

class UhmapEnvParseHelper:
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
        encoded_obs = np.zeros(shape=(self.n_agents, 10), dtype=np.float32); p=0
        for i, agent in enumerate(self.agents):
            part_1 = np.array([
                agent.index,
                agent.Location['x'],
                agent.Location['y'],
                agent.Location['z'],
                agent.hp,
                agent.weaponCD,
            ])
            length = part_1.shape[0]
            encoded_obs[i,:length] = part_1[:]
        from UTILS.tensor_ops import repeat_at
        encoded_obs_all_agent = repeat_at(encoded_obs, insert_dim=0, n_times=self.n_agents)
        return encoded_obs_all_agent


class UhmapEnv(BaseEnv, UhmapEnvParseHelper):
    def __init__(self, rank) -> None:
        super().__init__(rank)
        self.id = rank
        self.render = ScenarioConfig.render and (self.id==0)
        self.n_agents = ScenarioConfig.n_team1agent + ScenarioConfig.n_team2agent
        self.agents = [Agent(team=0, team_id=i, uid=i                            ) for i in range(ScenarioConfig.n_team1agent)] \
                    + [Agent(team=1, team_id=i, uid=i+ScenarioConfig.n_team1agent) for i in range(ScenarioConfig.n_team2agent)]
        # self.observation_space = ?
        # self.action_space = ?
        if ScenarioConfig.StateProvided:
            # self.observation_space['state_shape'] = ?
            pass
        self.render = ScenarioConfig.render and (rank==0)
        ipport = (ScenarioConfig.TcpAddr, ScenarioConfig.UhmapPort)
        # os.system()
        if (not self.render) and ScenarioConfig.UhmapServerExe != '':
            subprocess.Popen([
                ScenarioConfig.UhmapServerExe,
                '-log', 
                '-TimeDilation=%.4f'%ScenarioConfig.TimeDilation, 
                '-FrameRate=%d'%ScenarioConfig.FrameRate,
                '-LockGameDuringCom=True',
            ])
            # subprocess.Popen(['F:/UHMP/Build/WindowsServer/UHMPServer.exe','-log', '-TimeDilation=10', '-FrameRate=240'])
            # subprocess.Popen(['F:/UHMP/Build/WindowsServer/UHMPServer.exe','-log', '-TimeDilation=20', '-FrameRate=480'])
            print('UHMAP started, wait 10s before continue ...')
            time.sleep(10)
        elif self.render and ScenarioConfig.UhmapRenderExe != '':
            print('UHMAP render client started, wait 10s before continue ...')
            subprocess.Popen([
                ScenarioConfig.UhmapRenderExe,
                '-log', 
                '-TimeDilation=%.4f'%ScenarioConfig.TimeDilation, 
                '-FrameRate=%d'%ScenarioConfig.FrameRate,
                '-LockGameDuringCom=True',
                "-ResX=1280",
                "-ResY=720",
                "-WINDOWED"
            ])
            time.sleep(10)
        elif self.render and ScenarioConfig.UhmapServerExe == '':
            pass
        else:
            print('Cannot start Headless Server Or GUI Server!')
            assert False, 'Cannot start Headless Server Or GUI Server!'

        self.client = TcpClientP2P(ipport, obj='str')
        self.t = 0
        #  run flag https://docs.unrealengine.com/5.0/en-US/unreal-engine-pixel-streaming-reference/
        #  ./UHMP.exe -ResX=1280 -ResY=720 -WINDOWED
        #  ./UHMP.exe -ResX=1280 -ResY=720 -WINDOWED -RenderOffscreen -TimeDilation=1.25 -FrameRate=30
        #  ./UHMP.exe -ResX=1280 -ResY=720 -WINDOWED -TimeDilation=1.25 -FrameRate=30
        #  ./UHMP.exe -ResX=1280 -ResY=720 -WINDOWED -TimeDilation=2.5 -FrameRate=60
        #  ./UHMP.exe -ResX=1280 -ResY=720 -WINDOWED -TimeDilation=3.75 -FrameRate=90 -RenderOffscreen
        #  ./UHMP.exe -ResX=1280 -ResY=720 -WINDOWED -TimeDilation=5 -FrameRate=120 -RenderOffscreen
        #  ./UHMP.exe -ResX=0 -ResY=0 -WINDOWED -TimeDilation=5 -FrameRate=120 -RenderOffscreen


    # override reset function
    def reset(self):
        self.t = 0
        AgentSettingArray = []
        agent_uid_cnt = 0
        for i in range(ScenarioConfig.n_team1agent):
            x = -10990.0 + 500*(i+1)  *  (-1)**(i+1)
            y = -9440.0
            # 500 is slightly above the ground (depending the map you have built), 
            # but agent will be spawn to ground automatically
            z = 500 
            AgentSettingArray.append(
                {
                    'ClassName': 'AgentControllable',   # FString ClassName = "";
                    'AcceptRLControl': True,    # bool AcceptRLControl = 0;
                    'AgentTeam': 0, # int AgentTeam = 0;
                    'IndexInTeam': i,   # int IndexInTeam = 0;
                    'UID': agent_uid_cnt,   # int UID = 0;
                    'MaxMoveSpeed': 600,
                    'AgentHp':100,
                    'RSVD1':'(R=0,G=1,B=0,A=1)',
                    'InitLocation': { 'x': x,  'y': y, 'z': z, },
                },
            )
            agent_uid_cnt += 1

        for i in range(ScenarioConfig.n_team2agent):
            x = -10990.0 + 3000*(i+1)  *  (-1)**(i+1)
            y = -21770.0
            # 500 is slightly above the ground, but agent will be spawn to ground automatically
            z = 500 
            AgentSettingArray.append(
                {
                    'ClassName': 'AgentControllable',
                    'AcceptRLControl': False,
                    'AgentTeam': 1,
                    'IndexInTeam': i,
                    'UID': agent_uid_cnt,
                    'MaxMoveSpeed': 600,
                    'AgentHp':100,
                    'InitLocation': { 'x': x, 'y': y, 'z': z, },
                },
            )
            agent_uid_cnt += 1

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
        if DEBUG:
            print靛(resp['dataGlobal'])
        return self.parse_response_ob_info(resp)


    # override step function
    def step(self, act):

        assert len(act) == self.n_agents
        act_send = [digit2act_dictionary[a] for a in act] + \
                   ['ActionSet2::PatrolMoving;x=-10525.0 Y=-6009.0 Z=380.0' for _ in range(ScenarioConfig.n_team2agent)]
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


# please register this into MISSIONS/env_router.py
def make_uhmap_env(env_id, rank):
    if ScenarioConfig.SubTaskSelection == 'UhmapEnv':
        return UhmapEnv(rank)
    if ScenarioConfig.SubTaskSelection == 'UhmapBreakingBad':
        from .SubTasks.UhmapBreakingBad import UhmapBreakingBad
        return UhmapBreakingBad(rank)
    if ScenarioConfig.SubTaskSelection == 'UhmapLargeScale':
        from .SubTasks.UhmapLargeScale import UhmapLargeScale
        return UhmapLargeScale(rank)
