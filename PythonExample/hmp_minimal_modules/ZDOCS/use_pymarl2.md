# Using pymarl2 as Algorithm

## step 1: clone hmap code
```
git clone https://github.com/binary-husky/hmp2g.git

cd hmp2g

git submodule update --init

git submodule foreach -q --recursive 'branch="$(git config -f $toplevel/.gitmodules submodule.$name.branch)"; git switch $branch'
```

How to clean temp files when things are not working as expected:
```sh
# switch to master branch
git checkout master --force
# pull lastest code
git pull --force
# clean work directory
git clean -xfd
# clean submodule
git submodule foreach git clean -xfd
```

## step 2: run example
Save following file as ```private_debug.jsonc```: 
``` json
{
    "config.py->GlobalConfig": {
        "note": "RVE-drone2-qmix-fixstate-run1",
        "env_name": "uhmap",
        "env_path": "MISSION.uhmap",
        "draw_mode": "Img",
        "num_threads": 8,
        "report_reward_interval": 256,
        "test_interval": 5120,
        "test_epoch": 256,
        "interested_team": 0,
        "seed": 8529,
        "device": "cuda",
        "max_n_episode": 5000000,
        "fold": 1,
        "backup_files": [
            "ALGORITHM/pymarl2_compat",
            "MISSION/uhmap"
        ]
    },
    "MISSION.uhmap.uhmap_env_wrapper.py->ScenarioConfig": {
        "N_AGENT_EACH_TEAM": [10, 10], // update N_AGENT_EACH_TEAM
        "MaxEpisodeStep": 125,
        "StepGameTime": 0.5,
        "StateProvided": false,
        "render": false,
        "UElink2editor": false,
        "HeteAgents": true,
        "UnrealLevel": "UhmapLargeScale",
        "SubTaskSelection": "UhmapLargeScale",
        "UhmapVersion": "3.5",
        "UhmapRenderExe": "/home/hmp/UnrealHmapBinary/Version3.5/LinuxNoEditor/UHMP.sh",
        "UhmapServerExe": "/home/hmp/UnrealHmapBinary/Version3.5/LinuxServer/UHMPServer.sh",
        "TimeDilation": 64,
        "TEAM_NAMES": [
            "ALGORITHM.pymarl2_compat.pymarl2_compat->PymarlFoundation",
            "ALGORITHM.script_ai.uhmap_ls->DummyAlgorithmLinedAttack"
        ]
    },
    "MISSION.uhmap.SubTasks.UhmapLargeScaleConf.py->SubTaskConfig":{
        "agent_list": [
            { "team":0,  "tid":0,   "type":"RLA_UAV_Support", "init_fn_name":"init_air"      },
            { "team":0,  "tid":1,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":0,  "tid":2,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":0,  "tid":3,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":0,  "tid":4,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":0,  "tid":5,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":0,  "tid":6,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":0,  "tid":7,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":0,  "tid":8,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":0,  "tid":9,   "type":"RLA_UAV_Support", "init_fn_name":"init_air"      },

            { "team":1,  "tid":0,   "type":"RLA_UAV_Support", "init_fn_name":"init_air"      },
            { "team":1,  "tid":1,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":1,  "tid":2,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":1,  "tid":3,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":1,  "tid":4,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":1,  "tid":5,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":1,  "tid":6,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":1,  "tid":7,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":1,  "tid":8,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":1,  "tid":9,   "type":"RLA_UAV_Support", "init_fn_name":"init_air"      },
        ]
    },


    "ALGORITHM.script_ai.uhmap_ls.py->DummyAlgConfig": {
        "reserve": ""
    },
    "ALGORITHM.pymarl2_compat.pymarl2_compat.py->AlgorithmConfig": {
        "use_shell": "mini_shell_uhmap",
        "state_compat": "pad",
        "pymarl_config_injection": {
            "controllers.my_n_controller.py->PymarlAlgorithmConfig": {
                "use_normalization": "True",
                "use_vae": "False"
            },
            "config.py->GlobalConfig": {
                "batch_size": 128,
                "load_checkpoint": "False"
            }
        }
    }
}
```

Then start training with:
```sh
python main.py -c private_debug.jsonc
```