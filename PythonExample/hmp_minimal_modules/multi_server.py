base = """
{
    "config.py->GlobalConfig": {
        "note": "sc-MMM2-conc-db1",               // experiment note, also means the log saving directory
        // "train_time_testing": "False",                      // do not manage train time testing, pymarl env manage the testing itself
        "env_name":"sc2",                                   // starcraft 2
        "env_path":"MISSION.starcraft.sc2_env_wrapper",    // starcraft 2
        // "interested_agent_num":100,                         // only for reward logging, **not needed because sc2 use uniform team reward
        "draw_mode": "Img",                                 // plot curlves as image
        "num_threads": "16",                                 // number of parallel envs
        "report_reward_interval": "128",                      // report the reward averaging x episodes
        "test_interval": "512",                             // begin a test run every x episodes, test run is managed by pymarl side
        "test_epoch": "64",                                 // begin a test run every x episodes, test run is managed by pymarl side
        "device": "cuda",
        "max_n_episode": 1500000,
        "fold": "1",                                        // each linux process handle x parallel envs
        "backup_files":[
        ]
    },

    "MISSION.starcraft.sc2_env_wrapper.py->ScenarioConfig": {
        "map_": "MMM2",
        "sc_version": "2.4.6",
        // "map_": "5m_vs_6m",
        // "SINGLE_TEAM_N_AGENT": 5,
        // "episode_limit": 60,
        // "reward_vec": true,
        "TEAM_NAMES": [
            "ALGORITHM.conc_4hist_scdb.foundation->ReinforceAlgorithmFoundation"
        ]
    },

    "ALGORITHM.conc_4hist_scdb.foundation.py->AlgorithmConfig": {
        "train_traj_needed": "128",
        "n_focus_on": 3,
        "actor_attn_mod": "False",
        "lr": 0.0001,
        "ppo_epoch": 24,
        "load_checkpoint": "False"
    }
}
"""


import commentjson as json
import numpy as np
base_conf = json.loads(base)
n_run = 4
n_run_mode = [
    {
        "addr": "localhost:2266",
        "usr": "hmp",
        "pwd": "hmp"
    },
]*n_run
assert len(n_run_mode)==n_run

sum_note = "MMM2-conc4hist"
conf_override = {

    "config.py->GlobalConfig-->seed":       
        [
            np.random.randint(0, 10000) for _ in range(n_run)
        ],

    "config.py->GlobalConfig-->device":
        [
            'cuda',
            'cuda',
            'cuda',
            'cuda',
        ],

    "config.py->GlobalConfig-->note":
        [
            "n_focus_on_run1_3focus",
            "n_focus_on_run2_3focus",
            "n_focus_on_run1_5focus",
            "n_focus_on_run2_5focus",
        ],

    "ALGORITHM.conc_4hist_scdb.foundation.py->AlgorithmConfig-->n_focus_on":
        [
            3,
            3,
            5,
            5,
        ],
}

if __name__ == '__main__':
    # copy the experiments
    import shutil, os
    shutil.copyfile(__file__, os.path.join(os.path.dirname(__file__), 'batch_experiment_backup.py'))
    # run experiments remotely
    from UTIL.batch_exp import run_batch_exp
    run_batch_exp(sum_note, n_run, n_run_mode, base_conf, conf_override, __file__)
