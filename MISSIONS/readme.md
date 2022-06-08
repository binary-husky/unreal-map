# To interface with most algorithm, the task configuration should at least contain following field:

## Parameter Internal Relationship
* You may notice some configuration field ends with ```_cv```, they are parameters chained with other parameters. For example, when changing the ```map```, the limit of ```episode_length``` and the number of agents ```N_AGENT_EACH_TEAM``` are implicated and also need to be changed. 
To make it simple, we add ```episode_length_cv``` and ```N_AGENT_EACH_TEAM_cv``` to record this link with lambda function.

* When parameters (e.g. ```map```) that are bind to other parameters are changed, 
the Transparent Parameter Control (TPC) module will scan and parse variables with twin variables that end with ```_cv```, and automatically modify their values. (refer to ./UTILS/config_args.py)

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

## How to Introduce a New Mission Environment

### Step 1: Declare Mission Info (how many agents and actions, maximum episode steps et.al.)
- make a folder under ```./MISSIONS```, e.g. ```./MISSIONS/uhmap.```
- make a py file, e.g. ```./MISSIONS/uhmap/uhmap_env_wrapper.py```
- in ```uhmap_env_wrapper.py```, copy and paste following template:

```python
from UTILS.config_args import ChainVar
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
    MaxEpisodeStep = 10

    # <Part 3> Needed by some ALGORITHM #
    StateProvided = False
    AvailActProvided = False
    EntityOriented = False

    n_actions = 2
    show_details = False
```

### Step 2: Writing Environment


