
class SubTaskConfig():
    agent_list = [
            { "team": 0,    "type": "RLA_UAV_Support",  "init_fn_name": "init_air"   },
            { "team": 0,    "type": "RLA_CAR",          "init_fn_name": "init_ground"},
            { "team": 0,    "type": "RLA_CAR_Laser",    "init_fn_name": "init_ground_tank"},
            { "team": 0,    "type": "RLA_CAR",          "init_fn_name": "init_ground"},
            { "team": 0,    "type": "RLA_CAR_Laser",    "init_fn_name": "init_ground_tank"},
            { "team": 0,    "type": "RLA_CAR",          "init_fn_name": "init_ground"},
            { "team": 0,    "type": "RLA_CAR_Laser",    "init_fn_name": "init_ground_tank"},
            { "team": 0,    "type": "RLA_CAR",          "init_fn_name": "init_ground"},
            { "team": 0,    "type": "RLA_CAR_Laser",    "init_fn_name": "init_ground_tank"},
            { "team": 0,    "type": "RLA_UAV_Support",  "init_fn_name": "init_air"   },
            { "team": 1,    "type": "RLA_UAV_Support",  "init_fn_name": "init_air"   },
            { "team": 1,    "type": "RLA_CAR",          "init_fn_name": "init_ground"},
            { "team": 1,    "type": "RLA_CAR_Laser",    "init_fn_name": "init_ground_tank"},
            { "team": 1,    "type": "RLA_CAR",          "init_fn_name": "init_ground"},
            { "team": 1,    "type": "RLA_CAR_Laser",    "init_fn_name": "init_ground_tank"},
            { "team": 1,    "type": "RLA_CAR",          "init_fn_name": "init_ground"},
            { "team": 1,    "type": "RLA_CAR_Laser",    "init_fn_name": "init_ground_tank"},
            { "team": 1,    "type": "RLA_CAR",          "init_fn_name": "init_ground"},
            { "team": 1,    "type": "RLA_CAR_Laser",    "init_fn_name": "init_ground_tank"},
            { "team": 1,    "type": "RLA_UAV_Support",  "init_fn_name": "init_air"   }
    ]

    obs_vec_length = 23
    obs_n_entity = 11
    ActionFormat = 'Multi-Digit'
