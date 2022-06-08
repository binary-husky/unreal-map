import json
from UTILS.network import TcpClientP2P
from UTILS.config_args import ChainVar

# please register this ScenarioConfig into MISSIONS/env_router.py
class ScenarioConfig(object):  
    '''
        ScenarioConfig: This config class will be 'injected' with new settings from JSONC.
        (E.g., override configs with ```python main.py --cfg example.jsonc```)
        (As the name indicated, ChainVars will change WITH vars it 'chained_with' during config injection)
        (please see UTILS.config_args to find out how this advanced trick works out.)
    '''
    n_team1agent = 5

    # <Part 1> Needed by the hmp core #
    N_TEAM = 1

    N_AGENT_EACH_TEAM = [n_team1agent,]
    N_AGENT_EACH_TEAM_cv = ChainVar(lambda n_team1agent: [n_team1agent,], chained_with=['n_team1agent'])

    AGENT_ID_EACH_TEAM = [range(0,n_team1agent),]
    AGENT_ID_EACH_TEAM_cv = ChainVar(lambda n_team1agent: [range(0,n_team1agent),], chained_with=['n_team1agent'])

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

    # <Part 3> Needed by some ALGORITHM #
    StateProvided = False
    AvailActProvided = False
    EntityOriented = False

    n_actions = 2
    obs_vec_length = 10


class BaseEnv(object):
    def __init__(self, rank) -> None:
        self.observation_space = None
        self.action_space = None
        self.rank = rank

    def step(self, act):
        # obs: a Tensor with shape (n_agent, ...)
        # reward: a Tensor with shape (n_agent, 1) or (n_team, 1)
        # done: a Bool
        # info: a dict
        raise NotImplementedError
        # Warning: if you have only one team and RewardAsUnity, 
        # you must make sure that reward has shape=[n_team=1, 1]
        # e.g. 
        # >> RewardForTheOnlyTeam = +1
        # >> RewardForAllTeams = np.array([RewardForTheOnlyTeam, ])
        # >> return (ob, RewardForAllTeams, done, info)
        return (ob, RewardForAllTeams,  done, info)  # choose this if RewardAsUnity
        return (ob, RewardForAllAgents, done, info)  # choose this if not RewardAsUnity

    def reset(self):
        # obs: a Tensor with shape (n_agent, ...)
        # done: a Bool
        raise NotImplementedError
        return ob, info


class UhmapEnvParseHelper:

    def parse_response_ob_info(self, response):
        ob = None
        info = None
        return ob, info


class UhmapEnv(BaseEnv, UhmapEnvParseHelper):
    def __init__(self, rank) -> None:
        super().__init__(rank)
        self.id = rank
        self.render = ScenarioConfig.render and (self.id==0)
        self.n_agents = ScenarioConfig.n_team1agent
        # self.observation_space = ?
        # self.action_space = ?
        if ScenarioConfig.StateProvided:
            # self.observation_space['state_shape'] = ?
            pass
        if self.render:
            # render init
            pass
        ipport = ('cloud.fuqingxu.top', 21051)
        self.client = TcpClientP2P(ipport, obj='str')
        self.t = 0

    def reset(self):
        self.t = 0
        AgentSettingArray = []
        for i in range(ScenarioConfig.n_team1agent):
            x = 2000*i
            y = 0
            # 500 is slightly above the ground, but agent will be spawn to ground automatically
            z = 500 
            AgentSettingArray.append(
                {
                    'ClassName': 'AgentControllable',
                    'AgentTeam': 0,
                    'IndexInTeam': i,
                    'UID': i,
                    'TimeStep' : self.t,
                    'MaxMoveSpeed': 600,
                    'InitLocation': {
                        'x': x,
                        'y': y,
                        'z': z,
                    },
                },
            )

        json_to_send = json.dumps({
            'valid': True,
            'DataCmd': 'reset',
            'NumAgents' : ScenarioConfig.n_team1agent,
            'AgentSettingArray': AgentSettingArray,
            'TimeStep' : 0,
            'Actions': None,
        })
        resp = self.client.send_and_wait_reply(json_to_send)
        resp = json.loads(resp)
        return self.parse_response_ob_info(resp)

    def step(self, act):
        json_to_send = json.dumps({
            'valid': True,
            'DataCmd': 'step',
            'TimeStep': self.t,
            'Actions': act,
        })
        response = self.client.send_and_wait_reply(json_to_send)
        response = json.loads(response)

        ob = None
        RewardForAllTeams = None
        done = None
        info = None
        return (ob, RewardForAllTeams,  done, info)  # choose this if RewardAsUnity


# please register this into MISSIONS/env_router.py
def make_uhmap_env(env_id, rank):
    return UhmapEnv(rank)