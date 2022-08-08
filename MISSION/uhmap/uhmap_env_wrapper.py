import json, os, subprocess, time, stat, platform
import numpy as np
from UTIL.colorful import print蓝, print靛, print亮红
from UTIL.network import TcpClientP2PWithCompress, find_free_port_no_repeat
from UTIL.config_args import ChainVar
from ..common.base_env import BaseEnv
from .actset_lookup import binary_friendly, dictionary_n_actions
from .agent import Agent
# please register this ScenarioConfig into MISSION/env_router.py
class ScenarioConfig(object):  
    '''
        ScenarioConfig: This config class will be 'injected' with new settings from JSONC.
        (E.g., override configs with ```python main.py --cfg example.jsonc```)
        (As the name indicated, ChainVars will change WITH vars it 'chained_with' during config injection)
        (please see UTIL.config_args to find out how this advanced trick works out.)
    '''
    n_team1agent = 5
    n_team2agent = 5

    # <Part 1> Needed by the hmp core #
    N_TEAM = 2

    N_AGENT_EACH_TEAM = [n_team1agent, n_team2agent]
    N_AGENT_EACH_TEAM_cv = ChainVar(lambda n1, n2: [n1, n2], chained_with=['n_team1agent', 'n_team2agent'])

    AGENT_ID_EACH_TEAM = [range(0,n_team1agent), range(n_team1agent,n_team1agent+n_team2agent)]
    AGENT_ID_EACH_TEAM_cv = ChainVar(lambda  n1, n2: [range(0,n1),range(n1,n1+n2)], chained_with=['n_team1agent', 'n_team2agent'])

    CanTurnOff = False

    # Hete agents
    HeteAgents = False
    HeteAgentType = []


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

    UnrealLevel = 'UhmapBreakingBad'
    SubTaskSelection = 'UhmapBreakingBad'

    UElink2editor = False
    AutoPortOverride = False

    # this is not going to be precise,
    # the precise step time will be floor(StepGameTime/TimeDilation*FrameRate)*TimeDilation/FrameRate
    StepGameTime = 0.5  

    UhmapServerExe = 'F:/UHMP/Build/WindowsServer/UHMPServer.exe'
    UhmapRenderExe = ''
    TimeDilation = 1.0    # engine calcualtion speed control
    FrameRate = 25.6 # must satisfy: (TimeDilation=1*n, FrameRate=25.6*n)
    FrameRate_cv = ChainVar(lambda TimeDilation: (TimeDilation/1 * 25.6), chained_with=['TimeDilation'])
    UhmapStartCmd = []
    # <Part 3> Needed by some ALGORITHM #
    StateProvided = False
    AvailActProvided = False
    EntityOriented = True

    ActionFormat = 'Multi-Digit'    # 'Single-Digit'

    n_actions = dictionary_n_actions
    obs_vec_length = 38
    ObsBreakBase = 1e4





class UhmapEnvParseHelper:
    def parse_response_ob_info(self, response):
        raise NotImplementedError

    def make_obs(self):
        raise NotImplementedError


class UhmapEnv(BaseEnv, UhmapEnvParseHelper):
    def __init__(self, rank) -> None:
        super().__init__(rank)
        self.id = rank
        self.render = ScenarioConfig.render and (self.id==0)
        self.n_agents = ScenarioConfig.n_team1agent + ScenarioConfig.n_team2agent
        self.n_teams = ScenarioConfig.N_TEAM
        self.agents = [Agent(team=0, team_id=i, uid=i                            ) for i in range(ScenarioConfig.n_team1agent)] \
                    + [Agent(team=1, team_id=i, uid=i+ScenarioConfig.n_team1agent) for i in range(ScenarioConfig.n_team2agent)]
        # self.observation_space = ?
        # self.action_space = ?
        if ScenarioConfig.StateProvided:
            # self.observation_space['state_shape'] = ?
            pass

        # Restart env, this is very fast, can be a failsave if there is memory leaking away on UE side
        self.max_simulation_life = 1024
        
        self.simulation_life = self.max_simulation_life
        # with a lock, we can initialize UE side one by one (not necessary though)
        self.activate_simulation(self.id)

    def __del__(self):
        self.terminate_simulation()

    def activate_simulation(self, rank):
        print('thread %d initializing'%rank)
        self.sim_thread = 'activiting'
        self.render = ScenarioConfig.render #  and (rank==0)
        which_port = ScenarioConfig.UhmapPort
        if ScenarioConfig.AutoPortOverride:
            which_port, release_port_fn = find_free_port_no_repeat()   # port for hmp data exchanging
        ue_visual_port, release_port_fn = find_free_port_no_repeat()    # port for remote visualizing
        print蓝('Port %d will be used by hmp, port %d will be used by UE internally'%(which_port, ue_visual_port))

        ipport = (ScenarioConfig.TcpAddr, which_port)
        # os.system()
        if not ScenarioConfig.UElink2editor:
            assert ScenarioConfig.AutoPortOverride
            # * A Butterfly Effect problem *:
            # UE4 use float (instead of double) for time delta calculation,
            # causing some error calcualtion dt = 1/FrameRate
            # which will be enlarged due to Butterfly Effect
            # therefore we have to make sure that FrameRate = 16,32,64,...
            print('checking ScenarioConfig args problems ...') 
            assert binary_friendly(1/ScenarioConfig.FrameRate), "* A Butterfly Effect problem *"
            assert binary_friendly(ScenarioConfig.TimeDilation/256), "* A Butterfly Effect problem *"
            real_step_time = np.floor(ScenarioConfig.StepGameTime/ScenarioConfig.TimeDilation*ScenarioConfig.FrameRate)*ScenarioConfig.TimeDilation/ScenarioConfig.FrameRate
            # print亮红('Alert, the real Step Game Time will be:', real_step_time) 
            # deal with linux env
            if platform.system()=="Linux":
                # expand '~' path
                ScenarioConfig.UhmapServerExe = os.path.expanduser(ScenarioConfig.UhmapServerExe)
                # give execution permission
                st = os.stat(ScenarioConfig.UhmapServerExe)
                os.chmod(ScenarioConfig.UhmapServerExe, st.st_mode | stat.S_IEXEC)

            if (not self.render) and ScenarioConfig.UhmapServerExe != '':
                # start child process
                self.sim_thread = subprocess.Popen([
                    ScenarioConfig.UhmapServerExe,
                    # '-log', 
                    '-TcpPort=%d'%which_port,   # port for hmp data exchanging
                    '-Port=%d'%ue_visual_port,   # port for remote visualizing
                    '-OpenLevel=%s'%ScenarioConfig.UnrealLevel, 
                    '-TimeDilation=%.8f'%ScenarioConfig.TimeDilation, 
                    '-FrameRate=%.8f'%ScenarioConfig.FrameRate,
                    '-IOInterval=%.8f'%ScenarioConfig.StepGameTime,
                    '-Seed=%d'%int(np.random.rand()*1e5), # 如果已经设定了主线程随机数种子，这里随机出来的数字则是确定的
                    '-DebugMod=False',
                    '-LockGameDuringCom=True',
                ], stdout=subprocess.DEVNULL)
                print('UHMAP (Headless) started ...')
            elif self.render and ScenarioConfig.UhmapRenderExe != '':
                self.sim_thread = subprocess.Popen([
                    ScenarioConfig.UhmapRenderExe,
                    # '-log', 
                    '-TcpPort=%d'%which_port,   # port for hmp data exchanging
                    '-Port=%d'%ue_visual_port,   # port for remote visualizing
                    '-OpenLevel=%s'%ScenarioConfig.UnrealLevel, 
                    '-TimeDilation=%.8f'%ScenarioConfig.TimeDilation, 
                    '-FrameRate=%.8f'%ScenarioConfig.FrameRate,
                    '-IOInterval=%.8f'%ScenarioConfig.StepGameTime,
                    '-Seed=%d'%int(np.random.rand()*1e5), # 如果已经设定了主线程随机数种子，这里随机出来的数字则是确定的
                    '-DebugMod=False',
                    '-LockGameDuringCom=True',
                    "-ResX=1280",
                    "-ResY=720",
                    "-WINDOWED"
                ], stdout=subprocess.DEVNULL)
                print('UHMAP (Render) started ...')
            else:
                print('Cannot start Headless Server Or GUI Server!')
                assert False, 'Cannot start Headless Server Or GUI Server!'
        else:
            print('Trying to link to unreal editor ...')
            assert not ScenarioConfig.AutoPortOverride

        time.sleep(1+np.abs(self.id)/100)
        self.client = TcpClientP2PWithCompress(ipport)
        MAX_RETRY = 100
        for i in range(MAX_RETRY):
            try: 
                self.client.manual_connect()
                print('handshake complete %d'%rank)
                break
            except: 
                if i>50:
                    print('Thread %d: Trying to connect to uhmap simulation. Going to take a while when openning for the first time. Retry %d ...'%(rank, i))
                else:
                    pass
                time.sleep(1)
        # now that port is bind, no need to hold them anymore
        release_port_fn(which_port)
        release_port_fn(ue_visual_port)
        self.t = 0
        print('thread %d initialize complete'%rank)


    def terminate_simulation(self):
        if hasattr(self, 'sim_thread') and self.sim_thread is not None:
            # self.sim_thread.terminate()
            
            self.client.send_dgram_to_target(json.dumps({
                'valid': True,
                'DataCmd': 'end_unreal_engine',
                'NumAgents' : ScenarioConfig.n_team1agent,
                'TimeStepMax': ScenarioConfig.MaxEpisodeStep,
                'TimeStep' : 0,
                'Actions': None,
            }))
            self.client.close()
            self.sim_thread = None
            # json_to_send = json.dumps({
            #     'valid': True,
            #     'DataCmd': 'end_unreal_engine',
            # })
            # self.client.send_dgram_to_target(json_to_send)


    # override reset function
    def reset(self):
        self.simulation_life -= 1
        if self.simulation_life < 0:
            print('restarting simutation')
            self.terminate_simulation()
            self.simulation_life = self.max_simulation_life
            self.activate_simulation(self.id)

    def sleep(self):
        self.simulation_life = -1
        self.terminate_simulation()

    # override step function
    def step(self, act):
        raise NotImplementedError
        # return (ob, RewardForAllTeams,  done, info)  # choose this if RewardAsUnity
        

# please register this into MISSION/env_router.py
def make_uhmap_env(env_id, rank):
    if ScenarioConfig.SubTaskSelection == 'UhmapEnv':
        return UhmapEnv(rank)
    if ScenarioConfig.SubTaskSelection == 'UhmapBreakingBad':
        from .SubTasks.UhmapBreakingBad import UhmapBreakingBad
        return UhmapBreakingBad(rank)
    if ScenarioConfig.SubTaskSelection == 'UhmapLargeScale':
        from .SubTasks.UhmapLargeScale import UhmapLargeScale
        return UhmapLargeScale(rank)
