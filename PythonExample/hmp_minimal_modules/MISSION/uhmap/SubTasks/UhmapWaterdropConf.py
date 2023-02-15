
class SubTaskConfig():
    agent_list = [
        { 'team':0, 'type':'Ship',   'init_fn_name':'init_ship',  },    # 0 
        { 'team':0, 'type':'Ship',   'init_fn_name':'init_ship',  },
        { 'team':0, 'type':'Ship',   'init_fn_name':'init_ship',  },
        { 'team':0, 'type':'Ship',   'init_fn_name':'init_ship',  },
        { 'team':0, 'type':'Ship',   'init_fn_name':'init_ship',  },
        { 'team':0, 'type':'Ship',   'init_fn_name':'init_ship',  },
        { 'team':0, 'type':'Ship',   'init_fn_name':'init_ship',  },
        { 'team':0, 'type':'Ship',   'init_fn_name':'init_ship',  },
        { 'team':0, 'type':'Ship',   'init_fn_name':'init_ship',  },
        { 'team':0, 'type':'Ship',   'init_fn_name':'init_ship',  },    # 9
        
        { 'team':1, 'type':'Waterdrop',  'init_fn_name':'init_waterdrop',  }, # 10
        { 'team':1, 'type':'Waterdrop',  'init_fn_name':'init_waterdrop',  }, # 11
    ]


    AgentPropertyDefaults = {
        'ClassName': 'RLA_CAR',     # FString ClassName = "";
        'DebugAgent': False,     
        'AgentTeam': 0,             # int AgentTeam = 0;
        'IndexInTeam': 0,           # int IndexInTeam = 0;
        'UID': 0,                   # int UID = 0;
        'MaxMoveSpeed': 600,        # move speed, test ok
        'InitLocation': { 'x': 0,  'y': 0, 'z': 0, },
        'InitRotation': { 'x': 0,  'y': 0, 'z': 0, },
        'AgentScale'  : { 'x': 1,  'y': 1, 'z': 1, },     # agent size, test ok
        'InitVelocity': { 'x': 0,  'y': 0, 'z': 0, },
        'AgentHp':100,
        "WeaponCD": 1,              # weapon fire rate
        "IsTeamReward": True,
        "Type": "",
        "DodgeProb": 0.8,           # probability of escaping dmg 闪避概率, test ok
        "ExplodeDmg": 25,           # ms explode dmg. test ok
        "FireRange": 1000.0,        # <= 1500
        "GuardRange": 1400.0,       # <= 1500
        "PerceptionRange": 1500.0,       # <= 1500
        'Color':'(R=0,G=1,B=0,A=1)',    # color
        "FireRange": 1000,
        'RSVD1':'',
        'RSVD2':'',
    }

    obs_vec_length = 23
    obs_n_entity = 10
    ActionFormat = 'ASCII'
