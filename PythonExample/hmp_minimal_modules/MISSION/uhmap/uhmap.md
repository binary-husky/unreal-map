# Unreal HMAP (UHMAP) 混合多智能体平台-虚幻仿真模块

## UHMAP 中对虚幻源代码的修改

- (1) 将lz4的接口暴露在外，方便使用
```
F:\UnrealSourceCode\UnrealEngine-4.27.2-release\Engine\Source\Runtime\Core\Public\Compression\lz4.h
新增一行
#define LZ4_DLL_EXPORT 1
```

- (2) 将AIPerception Sight的计算量拉高

```
F:\UnrealSourceCode\UnrealEngine-4.27.2-release\Engine\Source\Runtime\AIModule\Private\Perception\AISense_Sight.cpp
```
修改参数，这两个参数增大，有助于尽早发现进入范围的智能体（源代码中为了运行效率牺牲了实时性，用运行时间和Trace数量加以约束）
```
static const int32 DefaultMaxTracesPerTick = 16;
static const int32 DefaultMinQueriesPerTimeSliceCheck = 40;
```

## Switching MISSION to UHMAP in Json Config 切至虚幻仿真模块
Please use following template:

请使用以下配置文件模板:
```jsonc
{
    // config HMP core
    "config.py->GlobalConfig": {
        "note": "uhmp-dev",
        "env_name": "uhmap", // ***
        "env_path": "MISSION.uhmap", // ***
        "draw_mode": "Img",
        "num_threads": "1",
        // "heartbeat_on": "False",
        "report_reward_interval": "1",
        "test_interval": "128",
        "test_epoch": "4",
        "device": "cuda",
        "max_n_episode": 500000,
        "fold": "1",
        "backup_files": [
        ]
    },

    // config MISSION
    "MISSION.uhmap.uhmap_env_wrapper.py->ScenarioConfig": {  // ***
        "N_AGENT_EACH_TEAM": [3, 2], // update N_AGENT_EACH_TEAM
        "MaxEpisodeStep": 30,
        "n_actions": 10, 
        "StateProvided": false,
        "render": false,
        "SubTaskSelection": "UhmapBreakingBad",
        "UhmapPort": 21051,
        // "UhmapServerExe": "",
        "UhmapRenderExe": "./../../WindowsNoEditor/UHMP.exe",
        "UhmapServerExe": "./../../WindowsServer/UHMPServer.exe",
        "TimeDilation": 1.25, // 时间膨胀系数
        "TEAM_NAMES": [
            "ALGORITHM.script_ai.dummy_uhmap->DummyAlgorithmT1",  // *** select ALGORITHMs
            "ALGORITHM.script_ai.dummy_uhmap->DummyAlgorithmT2"  // *** select ALGORITHMs
        ]
    },

    // config ALGORITHMs
    "ALGORITHM.script_ai.dummy_uhmap.py->DummyAlgConfig": { 
        "reserve": ""
    }
}
```

## Configurations 重要配置参数
path:json配置文件

|  Field   | Value  | Explaination  | zh Explaination  |
|  ----    | ----   | ----     |  ----  |
|  device     | ```str```   | select gpu     |  选择GPU或CPU  |
|  N_AGENT_EACH_TEAM     | ```list of int```   | Agent Num in Each Team     |  各队智能体数量  |
|  MaxEpisodeStep    | ```int```   |   Time Step Limit   |  对战时间步数限制  |
|  n_actions    | ```int```   | ----     |  强化学习预留  |
|  render    | ```bool```   | use render server     |  是否使用渲染  |
|  UhmapPort    | ```int```   |     |  临时，端口选择，后期将改为自动  |
|  UhmapPort    | ```int```   |     |  临时，端口选择，后期将改为自动  |
|  TimeDilation    | ```float```   |     |  时间膨胀，减小实现以实现慢动作，增大可以让CPU燃烧  |
|  TEAM_NAMES    | ```str```   |     |  分别指定一队、二队策略  |


## Unreal Agent Initializing Options 智能体初始化参数
path:```MISSION\uhmap\SubTasks\UhmapBreakingBad.py```
function:```reset```

|  Field   | Value  | Explaination  | zh Explaination  |
|  ----    | ----   | ----     |  ----  |
|  ClassName     | ```str```   | Select Agent Class in unreal engine side     |    |
|  AgentTeam     | ```int```   | team belonging of an agent      | 智能体的队伍归属   |
|  IndexInTeam     | ```int```   | team index of an agent      | 智能体在队伍中的编号   |
|  UID     | ```int```   | index of an agent in the environment     | 智能体在虚幻仿真中的唯一编号   |
|  MaxMoveSpeed     | ```float```   |      | 暂未接入，无效   |
|  AgentHp     | ```int```   |      | 初始生命值   |
|  WeaponCD     | ```float```   |      | 武器cooldown时间，单位秒   |
|  RSVD1     | ```str```   |      | 智能体展现颜色   |
|  InitLocation     | ```dict```   |      | 智能体初始位置   |

## Unit conversion 单位转换

Length unit in the system is 1mm,
e.g. 800 = 800mm = 0.8m

## Algorithm For demonstration
path:```ALGORITHM\script_ai\dummy_uhmap.py```
function:```interact_with_env(override)```

### Argument:
|  Field   | Value  | Explaination  | zh Explaination  |
|  ----    | ----   | ----     |  ----  |
|  ```State_Recall['Latest-Obs']```     |    |  observation array for reinforcement learning     |    |
|  ```State_Recall['ENV-PAUSE']```     |    |  show which thread is paused (refer to [TimeLine](./../../VISUALIZE/md_imgs/timeline.jpg))     |    |
|  ```State_Recall['Current-Obs-Step']```     |    |  show time step index in an episode     |    |
|  ```State_Recall['Latest-Team-Info']```     |    |  interfacing with script-based AIs, including structed agent location, uid, et.al.     |    |
|  ```State_Recall['Test-Flag']'```     |    |  show whether HMP central has recommanded to do a test run for RL     |    |
|  ```'State_Recall['Env-Suffered-Reset']''```     |    |  show whether a thread has be reset and start a new episode     |    |

### Convert Command Format:

#### attack a agent with UID
```python
encode_action_as_digits("SpecificAttacking", "N/A", x=None, y=None, z=None, UID=4, T=None, T_index=None)
```

#### PatrolMoving with coordinate
```python
encode_action_as_digits("PatrolMoving", "N/A", x=444*5, y=444*5, z=379, UID=None, T=None, T_index=None)
```

#### PatrolMoving with direction
```python
encode_action_as_digits("PatrolMoving", "Dir+X+Y", x=None, y=None, z=None, UID=None, T=None, T_index=None) 
encode_action_as_digits("PatrolMoving", "Dir+X-Y", x=None, y=None, z=None, UID=None, T=None, T_index=None) 
encode_action_as_digits("PatrolMoving", "Dir+X", x=None, y=None, z=None, UID=None, T=None, T_index=None) 
```

#### SpecificMoving with coordinate
```python
encode_action_as_digits("SpecificMoving", "N/A", x=444*5, y=444*5, z=379, UID=None, T=None, T_index=None)
```

#### SpecificMoving with direction
```python
encode_action_as_digits("SpecificMoving", "Dir+X+Y", x=None, y=None, z=None, UID=None, T=None, T_index=None) 
encode_action_as_digits("SpecificMoving", "Dir+X-Y", x=None, y=None, z=None, UID=None, T=None, T_index=None) 
encode_action_as_digits("SpecificMoving", "Dir+X", x=None, y=None, z=None, UID=None, T=None, T_index=None) 
```

#### Idle and change guard state
```python
encode_action_as_digits("Idle", "DynamicGuard", x=None, y=None, z=None, UID=None, T=None, T_index=None) 
encode_action_as_digits("Idle", "StaticAlert", x=None, y=None, z=None, UID=None, T=None, T_index=None) 
encode_action_as_digits("Idle", "AggressivePersue", x=None, y=None, z=None, UID=None, T=None, T_index=None) 
```
