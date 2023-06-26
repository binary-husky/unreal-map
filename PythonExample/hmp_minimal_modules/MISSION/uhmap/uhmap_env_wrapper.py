import json, os, subprocess, time, stat, platform, importlib
import numpy as np
from UTIL.colorful import print蓝, print靛, print亮红
from UTIL.network import TcpClientP2PWithCompress, find_free_port_no_repeat, get_host_ip
from UTIL.config_args import ChainVar
from config import GlobalConfig
from ..common.base_env import BaseEnv
from .actset_lookup import binary_friendly, dictionary_n_actions
from .agent import Agent

# please register this into MISSION/env_router.py
def make_uhmap_env(env_id, rank):
    if ScenarioConfig.SubTaskSelection == 'UhmapEnv':
        return UhmapEnv(rank)
    else:
        ST = ScenarioConfig.SubTaskSelection
        assert os.path.exists(f'./MISSION/uhmap/SubTasks/{ST}.py'), "Unknown subtask!"
        ST_CLASS = getattr(importlib.import_module(f'.SubTasks.{ST}', package='MISSION.uhmap'), ST)
        return ST_CLASS(rank)

def get_subtask_conf(subtask):
    ST = subtask
    assert os.path.exists(f'./MISSION/uhmap/SubTasks/{ST}Conf.py'), "Configuration not found!"
    ST_CONF_CLASS = getattr(importlib.import_module(f'.SubTasks.{ST}Conf', package='MISSION.uhmap'), 'SubTaskConfig')
    return ST_CONF_CLASS




def usual_id_arrangment(N_AGENT_EACH_TEAM):
    """
        e.g., 
        input [5, 3]
        output [range(0,5), range(5,8)]
    """
    AGENT_ID_EACH_TEAM = []
    p = 0
    for team_agent_num in N_AGENT_EACH_TEAM:
        AGENT_ID_EACH_TEAM.append(range(p, p + team_agent_num))
        p += team_agent_num
    return AGENT_ID_EACH_TEAM

# please register this ScenarioConfig into MISSION/env_router.py
class ScenarioConfig(object):  
    '''
        ScenarioConfig: This config class will be 'injected' with new settings from JSONC.
        (E.g., override configs with ```python main.py --cfg example.jsonc```)
        (As the name indicated, ChainVars will change WITH vars it 'chained_with' during config injection)
        (please see UTIL.config_args to find out how this advanced trick works out.)
    '''
    # <Part 1> Needed by the hmp core #
    N_AGENT_EACH_TEAM = [10, ]
    AGENT_ID_EACH_TEAM = usual_id_arrangment(N_AGENT_EACH_TEAM)
    N_TEAM = len(N_AGENT_EACH_TEAM)

    # chained parameters, will change along with 'N_AGENT_EACH_TEAM'
    AGENT_ID_EACH_TEAM_cv = ChainVar(lambda N_AGENT_EACH_TEAM: usual_id_arrangment(N_AGENT_EACH_TEAM), chained_with=['N_AGENT_EACH_TEAM'])
    N_TEAM_cv = ChainVar(lambda N_AGENT_EACH_TEAM: len(N_AGENT_EACH_TEAM), chained_with=['N_AGENT_EACH_TEAM'])
    
    # algorithm selection
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
    SubTaskConfig = get_subtask_conf(UnrealLevel)
    SubTaskConfig_cv = ChainVar(lambda UnrealLevel:get_subtask_conf(UnrealLevel), chained_with=['SubTaskSelection'])

    UElink2editor = False
    AutoPortOverride = True
    # AutoPortOverride is usually the reverse of UElink2editor
    AutoPortOverride_cv = ChainVar(lambda UElink2editor:(not UElink2editor), chained_with=['UElink2editor'])

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

    ActionFormat = 'ASCII' # 'ASCII'/'Multi-Digit'/'Single-Digit'

    n_actions = dictionary_n_actions

    obs_vec_length = get_subtask_conf(UnrealLevel).obs_vec_length
    obs_vec_length_cv = ChainVar(lambda UnrealLevel:get_subtask_conf(UnrealLevel).obs_vec_length, chained_with=['SubTaskSelection'])

    obs_n_entity = get_subtask_conf(UnrealLevel).obs_n_entity
    obs_n_entity_cv = ChainVar(lambda UnrealLevel:get_subtask_conf(UnrealLevel).obs_n_entity, chained_with=['SubTaskSelection'])

    # # ObsBreakBase = 1e4

    UhmapVersion = '2.3'

    CanTurnOff = False

    # Hete agents
    HeteAgents = False

    # 演示demo类别
    DemoType = "Default" 



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
        self.n_agents = sum(ScenarioConfig.N_AGENT_EACH_TEAM)
        assert self.n_agents == len(ScenarioConfig.SubTaskConfig.agent_list), 'agent number defination error'
        self.n_teams = ScenarioConfig.N_TEAM
        self.sim_thread = None
        self.client = None
        # self.observation_space = ?
        # self.action_space = ?
        if ScenarioConfig.StateProvided:
            # self.observation_space['state_shape'] = ?
            pass

        # Restart env, this is very fast, can be a failsafe if there is memory leaking away on UE side
        self.max_simulation_life = 2048
        
        self.simulation_life = self.max_simulation_life
        # with a lock, we can initialize UE side one by one (not necessary though)

        # wait until thread 0 finish its initialization (to avoid a traffic jam in server memory)
        traffic_light = './TEMP/uhmap_thread_0_init_ok_%s'%GlobalConfig.machine_info['ExpUUID'][:8]
        if rank != 0:
            while not os.path.exists(traffic_light): time.sleep(1)

        self.activate_simulation(self.id, find_port=True)

        # thread 0 finish its initialization, 
        if rank == 0:
            with open(traffic_light, mode='w+') as f: f.write(traffic_light)



    def __del__(self):
        self.terminate_simulation()

    def activate_simulation(self, rank, find_port=True):
        print('thread %d initializing'%rank)
        self.sim_thread = 'activiting'
        
        if find_port:
            self.render = ScenarioConfig.render #  and (rank==0)
            self.hmp_ue_port = ScenarioConfig.UhmapPort
            if ScenarioConfig.AutoPortOverride:
                self.hmp_ue_port, release_port_fn = find_free_port_no_repeat()   # port for hmp data exchanging
            if not ScenarioConfig.UElink2editor:
                self.ue_vis_port, release_port_fn = find_free_port_no_repeat()    # port for remote visualizing
                print蓝('Port %d will be used by hmp, port %d will be used by UE internally'%(self.hmp_ue_port, self.ue_vis_port))
            if (not self.render) and (not ScenarioConfig.UElink2editor):
                print蓝('To visualize on Windows, run "./UHMP.exe -OpenLevel=%s:%d -WINDOWED -TimeDilation=%.8f -FrameRate=%.8f -IOInterval=%.8f -DebugMod=False -LockGameDuringCom=True"'%(
                    get_host_ip(), self.ue_vis_port, ScenarioConfig.TimeDilation, ScenarioConfig.FrameRate, ScenarioConfig.StepGameTime))
            self.ip_port = (ScenarioConfig.TcpAddr, self.hmp_ue_port)
        
        
        # os.system()
        if not ScenarioConfig.UElink2editor:
            assert ScenarioConfig.AutoPortOverride
            # * A Butterfly Effect problem *:
            # UE4 use float (instead of double) for time delta calculation,
            # causing some error calcualtion dt = 1/FrameRate
            # which will be enlarged due to Butterfly Effect
            # therefore we have to make sure that FrameRate = 16,32,64,...
            print('checking ScenarioConfig args problems ...') 
            assert ScenarioConfig.TimeDilation <= 128, "* TimeDilation <= 128 *"
            assert binary_friendly(1/ScenarioConfig.FrameRate), "* A Butterfly Effect problem *"
            assert binary_friendly(ScenarioConfig.TimeDilation/256), "* A Butterfly Effect problem *"
            # real_step_time = 
            #   np.floor(ScenarioConfig.StepGameTime/ScenarioConfig.TimeDilation*ScenarioConfig.FrameRate) 
            #   * ScenarioConfig.TimeDilation / ScenarioConfig.FrameRate
            if not self.render:
                simulation_exe = ScenarioConfig.UhmapServerExe
                assert 'Server' in simulation_exe
            else: 
                simulation_exe = ScenarioConfig.UhmapRenderExe
                assert 'NoEditor' in simulation_exe

            if platform.system()=="Linux":
                if self.render: assert False, "You really want to render on Linux? If so, remove this line."
                if simulation_exe.endswith('.exe'): 
                    simulation_exe = simulation_exe.replace('/Windows', '/Linux')
                    simulation_exe = simulation_exe.replace('.exe','.sh')
                # expand '~' path
                simulation_exe = os.path.expanduser(simulation_exe)
            else:   # Windows
                if simulation_exe.endswith('.sh'): 
                    simulation_exe = simulation_exe.replace('/Linux', '/Windows')
                    simulation_exe = simulation_exe.replace('.sh', '.exe')
                if simulation_exe.startswith('/home'): 
                    simulation_exe = './TEMP' + simulation_exe

            if not os.path.exists(simulation_exe):
                if self.rank == 0:
                    from .auto_download import download_client_binary
                    download_client_binary(desired_path=simulation_exe, desired_version=ScenarioConfig.UhmapVersion, is_render_client=self.render)
                else:
                    while True:
                        time.sleep(60)
                        if os.path.exists(simulation_exe): break

            # give execution permission
            if platform.system()=="Linux":
                st = os.stat(simulation_exe)
                os.chmod(simulation_exe, st.st_mode | stat.S_IEXEC)

            if (not self.render) and simulation_exe != '':
                # start child process
                self.sim_thread = subprocess.Popen([
                    simulation_exe,
                    # '-log', 
                    '-TcpPort=%d'%self.hmp_ue_port,   # port for hmp data exchanging
                    '-Port=%d'%self.ue_vis_port,   # port for remote visualizing
                    '-OpenLevel=%s'%ScenarioConfig.UnrealLevel, 
                    '-TimeDilation=%.8f'%ScenarioConfig.TimeDilation, 
                    '-FrameRate=%.8f'%ScenarioConfig.FrameRate,
                    '-IOInterval=%.8f'%ScenarioConfig.StepGameTime,
                    '-Seed=%d'%int(np.random.rand()*1e5), # 如果已经设定了主线程随机数种子，这里随机出来的数字则是确定的
                    '-DebugMod=False',
                    # '-LLMCSV',
                    '-ABSLOG=%s'%os.path.abspath('./TEMP/uhmap/%s/%d.log'%(GlobalConfig.machine_info['ExpUUID'][:8], rank)),
                    '-Version=%s'%ScenarioConfig.UhmapVersion,
                    '-LockGameDuringCom=True',
                ], stdout=subprocess.DEVNULL)
                print('UHMAP (Headless) started ...')
            elif self.render and simulation_exe != '':
                self.sim_thread = subprocess.Popen([
                    simulation_exe,
                    # '-log', 
                    '-TcpPort=%d'%self.hmp_ue_port,   # port for hmp data exchanging
                    '-Port=%d'%self.ue_vis_port,   # port for remote visualizing
                    '-OpenLevel=%s'%ScenarioConfig.UnrealLevel, 
                    '-TimeDilation=%.8f'%ScenarioConfig.TimeDilation, 
                    '-FrameRate=%.8f'%ScenarioConfig.FrameRate,
                    '-IOInterval=%.8f'%ScenarioConfig.StepGameTime,
                    '-Seed=%d'%int(np.random.rand()*1e5), # 如果已经设定了主线程随机数种子，这里随机出来的数字则是确定的
                    '-DebugMod=False',
                    # '-LLMCSV',
                    '-ABSLOG=%s'%os.path.abspath('./TEMP/uhmap/%s/%d.log'%(GlobalConfig.machine_info['ExpUUID'][:8], rank)),
                    '-Version=%s'%ScenarioConfig.UhmapVersion,
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
        self.client = TcpClientP2PWithCompress(self.ip_port)
        MAX_RETRY = 150
        for i in range(MAX_RETRY):
            try: 
                self.client.manual_connect()
                print('handshake complete %d'%rank)
                break
            except: 
                if i>25:
                    print('Thread %d: Trying to connect to unreal engine. Related library not in memory, going to take some minutes. Retry %d ...'%(rank, i))
                elif i>75:
                    print('Thread %d: Waiting too long, please reduce parallel threads (num_threads), Retry %d ... | 请减小num_threads运行一次, 让动态库载入内存, 然后恢复num_threads即可'%(rank, i))
                elif i >= MAX_RETRY-1:
                    assert False, ('uhmap connection timeout, please reduce parallel threads (num_threads) !')
                time.sleep(1)
        # now that port is bind, no need to hold them anymore
        if find_port:
            if ScenarioConfig.AutoPortOverride: 
                release_port_fn(self.hmp_ue_port)
            if not ScenarioConfig.UElink2editor:
                release_port_fn(self.ue_vis_port)
        self.t = 0
        print('thread %d initialize complete'%rank)


    def terminate_simulation(self):
        if hasattr(self,'sim_thread') and (self.sim_thread is not None) and (self.client is not None):
            # self.sim_thread.terminate()
            # send terminate command to unreal side
            self.client.send_dgram_to_target(json.dumps({
                'valid': True,
                'DataCmd': 'end_unreal_engine',
                'TimeStepMax': ScenarioConfig.MaxEpisodeStep,
                'TimeStep' : 0,
                'Actions': None,
            }))
            self.client.close()
            self.sim_thread = None
            self.client = None


    # override reset function
    def reset(self):
        self.simulation_life -= 1
        if self.simulation_life < 0:
            print('restarting simutation')
            self.terminate_simulation()
            self.simulation_life = self.max_simulation_life
            self.activate_simulation(self.id, find_port=False)

    def sleep(self):
        self.simulation_life = -1
        self.terminate_simulation()

    # override step function
    def step(self, act):
        raise NotImplementedError
        # return (ob, RewardForAllTeams,  done, info)  # choose this if RewardAsUnity
        

