# Task Configuration Core Fields:

## Parameter Internal Relationship
* You may notice some configuration field ends with ```_cv```, they are parameters chained with other parameters. For example, when changing the ```map```, the limit of ```episode_length``` and the number of agents ```N_AGENT_EACH_TEAM``` are implicated and also need to be changed. 
To make it simple, we add ```episode_length_cv``` and ```N_AGENT_EACH_TEAM_cv``` to record this link with lambda function.

* When parameters (e.g. ```map```) that are bind to other parameters are changed, 
the Transparent Parameter Control (TPC) module will scan and parse variables with twin variables that end with ```_cv```, and automatically modify their values. (refer to ./UTIL/config_args.py)

Generally, you can safely ignore them and only pay attention to fields below.

## Fields
|  Field   | Value  | Explaination  | zh Explaination  |
|  ----    | ----   | ----     |  ----  |
| N_TEAM   | ```int```    | the number of agent teams in the tasks, information cannot be shared between different team | 队伍数量，每个队伍被一个ALGORITHM模块控制，队伍之间不可共享信息。大多数任务中，队伍之间是敌对关系  |
| N_AGENT_EACH_TEAM    | ```list (of int)``` | the number of agents in each team | 每个队伍的智能体数量  |
| AGENT_ID_EACH_TEAM   | ```list of list (of int)``` | the ID of agents in each team, double layer list, must agree with N_AGENT_EACH_TEAM! | 每个队伍的智能体的ID，双层列表，必须与N_AGENT_EACH_TEAM对应!  |
| TEAM_NAMES    | ```list (of string)``` | use which ALGORITHM to control each team, fill the path of chosen algorithm and its main class name, e.g.```"ALGORITHM.conc.foundation->ReinforceAlgorithmFoundation"``` | 选择每支队伍的控制算法，填写控制算法主模块的路径和类名|
| RewardAsUnity    | ```bool``` | Shared reward, or each agent has individual reward signal | 每个队伍的智能体共享集体奖励（True），或者每个队伍的智能体都独享个体奖励（False）  |
| ObsAsUnity    | ```bool``` | Agents do not has individual observation, only shared collective observation | 没有个体观测值，整个群体的观测值获取方式如同单智能体问题一样  |
| StateProvided    | ```bool``` | Whether the global state is provided in training. If True, the Algorithm can access both ```obs``` and ```state``` during training  | 是否在训练过程中提供全局state  |




# * How to Introduce a New Mission Environment

### Step 1: Declare Mission Info (how many agents and actions, maximum episode steps et.al.)
- make a folder under ```./MISSION```, e.g. ```./MISSION/uhmap.```
- make a py file, e.g. ```./MISSION/uhmap/uhmap_env_wrapper.py```
- in ```uhmap_env_wrapper.py```, copy and paste following template:

```python
from UTIL.config_args import ChainVar

# please register this ScenarioConfig into MISSION/env_router.py
class ScenarioConfig(object):  
    '''
        ScenarioConfig: This config class will be 'injected' with new settings from JSONC.
        (E.g., override configs with ```python main.py --cfg example.jsonc```)
        (As the name indicated, ChainVars will change WITH vars it 'chained_with' during config injection)
        (please see UTIL.config_args to find out how this advanced trick works out.)
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

```

### Step 2: Writing Environment

- For convenience, please copy and paste ```class BaseEnv(object)``` into your script:
```python
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

```

- Then create a class that inherit from it (```class UhmapEnv(BaseEnv)```):
```python
class UhmapEnv(BaseEnv):
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
```
- Next, it is time to write your own code of ```step()``` and ```reset()``` function.
There is little we can help about that, as it is your custom environment after all.

### Step 3: Write a Function to Initialize the Environment
A empty function getting a instance of environment, it will used in step 4. 
But don'y worry, two lines of code will do:
```python
# please register this into MISSION/env_router.py
def make_uhmap_env(env_id, rank):
    return UhmapEnv(rank)
```

### Step 4: Make Everything Kiss Together
This step will make HMP aware of the existence of this new MISSION.
- Open ```MISSION/env_router.py```
- Add the path of environment's configuration in ```import_path_ref```
``` python
import_path_ref = {
    "uhmap": ("MISSION.uhmap.uhmap_env_wrapper", 'ScenarioConfig'),
}   
```
- Add the path of environment's init function in ```env_init_function_ref```, e.g.:
``` python
env_init_function_ref = {
    "uhmap": ("MISSION.uhmap.uhmap_env_wrapper", "make_uhmap_env"),
}   
```

### Step 5: Write a Config Override to Start Experiment
Create a ```exp.jsonc``` or ```json``` file, 
copy and paste following content, and please pay attention to lines marked with ```***```, they are the most important ones:
```jsonc
{
    // config HMP core
    "config.py->GlobalConfig": {
        "note": "uhmp-dev",
        "env_name": "uhmap", // *** the selection of MISSION
        "env_path": "MISSION.uhmap", // *** confirm the path of env (a fail safe)
        "draw_mode": "Img",
        "num_threads": "1",
        "report_reward_interval": "1",
        "test_interval": "128",
        "test_epoch": "4",
        "device": "cuda",
        "max_n_episode": 500000,
        "fold": "4",
        "backup_files": [
        ]
    },

    // config MISSION
    "MISSION.uhmap.uhmap_env_wrapper.py->ScenarioConfig": {  // *** must kiss with "env_name" and "env_path"
        // remember this? declared in ScenarioConfig class in ./MISSION/math_game/uhmap.py.
        "n_team1agent": 4,
        "n_actions": 10,
        "StateProvided": false,
        "TEAM_NAMES": [
            "ALGORITHM.conc_4hist.foundation->ReinforceAlgorithmFoundation"  // *** select ALGORITHMs
        ]
    },

    // config ALGORITHMs
    "ALGORITHM.conc_4hist.foundation.py->AlgorithmConfig": { // must kiss with "TEAM_NAMES"
        "train_traj_needed": "16",
        "fix_n_sample": "True",
        "n_focus_on": 3,
        "lr": 0.0005,
        "ppo_epoch": 24,
        "gamma_in_reward_forwarding": "True",
        "gamma_in_reward_forwarding_value": 0.95,
        "gamma": 0.99
    }
}
```

At last, run experiment with ```python main.py --cfg ./path-to-exp-json/exp.jsonc```.
