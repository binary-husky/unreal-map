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

