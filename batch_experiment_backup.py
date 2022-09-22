base = """
{
    // --- Part1: config HMP core --- 
    "config.py->GlobalConfig": {
        "note": "z-hete-league",// http://localhost:59547
        "env_name": "uhmap",
        "env_path": "MISSION.uhmap",
        // "heartbeat_on": "False",
        "draw_mode": "Img",
        "num_threads": 32,
        "report_reward_interval": 128,
        "test_interval": 1280,
        "test_epoch": 128,
        "interested_team": 0,
        "seed": 10098,
        "device": "cuda",
        "max_n_episode": 5000000,
        "fold": 1,
        "backup_files": [
            "ALGORITHM/hete_league_onenet_fix",
            "MISSION/uhmap"
        ]
    },


    // --- Part2: config MISSION --- 
    "MISSION.uhmap.uhmap_env_wrapper.py->ScenarioConfig": {
        "n_team1agent": 10,
        "n_team2agent": 10,
        "MaxEpisodeStep": 125,
        "StepGameTime": 0.5,
        "StateProvided": false,
        "render": false, // note: random seed has different impact on renderer and server
        "UElink2editor": false,
        "AutoPortOverride": true,
        "HeteAgents": true,
        "UnrealLevel": "UhmapLargeScale",
        "SubTaskSelection": "UhmapLargeScale",
        // "UhmapRenderExe": "./../../WindowsNoEditor/UHMP.exe",
        // "UhmapServerExe": "./../../WindowsServer/UHMPServer.exe",
        "UhmapRenderExe": "/home/hmp/fuqingxu/UHMP/Build/LinuxNoEditor/UHMP.sh",
        "UhmapServerExe": "/home/hmp/fuqingxu/UHMP/Build/LinuxServer/UHMPServer.sh",
        "TimeDilation": 64, // simulation time speed up, larger is faster
        "TEAM_NAMES": [
            "ALGORITHM.hete_league_onenet_fix.foundation->ReinforceAlgorithmFoundation",
            "ALGORITHM.script_ai.uhmap_ls->DummyAlgorithmLinedAttack",
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






    // --- Part3: config ALGORITHM 1/2 --- 
    "ALGORITHM.script_ai.uhmap_ls.py->DummyAlgConfig": {
        "reserve": ""
    },

    // --- Part3: config ALGORITHM 2/2 --- 
    "ALGORITHM.hete_league_onenet_fix.shell_env.py->ShellEnvConfig": {
        "add_avail_act": true
    },
    "ALGORITHM.hete_league_onenet_fix.foundation.py->AlgorithmConfig": {
        "train_traj_needed": 128,
        "hete_n_alive_frontend": 2,
        "hete_n_net_placeholder":5,
        "hete_same_prob": 0.25,
        "load_checkpoint": false,
        "gamma": 0.99,
        "gamma_in_reward_forwarding": "True",
        "gamma_in_reward_forwarding_value": 0.95,
        "prevent_batchsize_oom": "True",
        "lr": 0.0002,
        "ppo_epoch": 24,
        "policy_resonance": true,
        "hete_exclude_zero_wr": true,
        "debug": false,
        "n_entity_placeholder": 11
    },
    "ALGORITHM.hete_league_onenet_fix.stage_planner.py->PolicyRsnConfig": {
        "resonance_start_at_update": 1,
        "yita_min_prob": 0.05,
        "yita_max": 0.5,
        "cosine_yita_shift": true,
        "cosine_yita_shift_cycle": 1000,
        "yita_inc_per_update": 0.01,
    },
    
}
"""


import commentjson as json
import numpy as np
base_conf = json.loads(base)
n_run = 6
n_run_mode = [
    {
        "addr": "localhost:2266",
        "usr": "hmp",
        "pwd": "hmp"
    },
]*n_run
assert len(n_run_mode)==n_run
conf_override = {
    
    
    "config.py->GlobalConfig-->seed":       
        [
            np.random.randint(0, 10000) for _ in range(n_run)
        ],
    "config.py->GlobalConfig-->device":
        [
            'cuda:4', 
            'cuda:4', 
            
            'cuda:5', 
            'cuda:5', 
            
            'cuda:6', 
            'cuda:6', 
        ],
    "config.py->GlobalConfig-->gpu_party":
        [
            'cuda4_party1', 
            'cuda4_party1', 
            
            'cuda5_party1', 
            'cuda5_party1', 
            
            'cuda6_party1', 
            'cuda6_party1', 
        ],
        
        
    "config.py->GlobalConfig-->note":
        [
            "R-onenetexclude_zero_wr-pr-prob1-T1000-sameprob-0.4-batch128-run1",
            "R-onenetexclude_zero_wr-pr-prob1-T1000-sameprob-0.5-batch128-run1",
            "R-onenetexclude_zero_wr-pr-prob1-T1000-sameprob-0.6-batch128-run1",
            
            "R-onenetexclude_zero_wr-pr-prob1-T500-sameprob-0.4-batch128-run1",
            "R-onenetexclude_zero_wr-pr-prob1-T500-sameprob-0.5-batch128-run1",
            "R-onenetexclude_zero_wr-pr-prob1-T500-sameprob-0.6-batch128-run1",
            
        ],
    "ALGORITHM.hete_league_onenet_fix.stage_planner.py->PolicyRsnConfig-->cosine_yita_shift_cycle":
        [
            1000,
            1000,
            1000,
            
            500,
            500,
            500,
        ],
        
    "ALGORITHM.hete_league_onenet_fix.foundation.py->AlgorithmConfig-->hete_same_prob":
        [
            1.0,
            1.0,
            1.0,
            
            1.0,
            1.0,
            1.0,
        ],

}

if __name__ == '__main__':
    # copy the experiments
    import shutil, os
    shutil.copyfile(__file__, os.path.join(os.path.dirname(__file__), 'batch_experiment_backup.py'))
    # run experiments remotely
    from UTIL.batch_exp import run_batch_exp
    run_batch_exp(n_run, n_run_mode, base_conf, conf_override)