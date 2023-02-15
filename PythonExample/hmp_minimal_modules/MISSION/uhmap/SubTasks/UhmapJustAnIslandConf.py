
class SubTaskConfig():
    agent_list = [
        { 'team':0,  'tid':0,  'uid':0,  'n_team_agent':10, 'type':'PlaneAgent',  'init_fn_name':'init_air',  },
        { 'team':0,  'tid':1,  'uid':1,  'n_team_agent':10, 'type':'PlaneAgent',  'init_fn_name':'init_air',  },
        { 'team':0,  'tid':2,  'uid':2,  'n_team_agent':10, 'type':'PlaneAgent',  'init_fn_name':'init_air',  },
        { 'team':0,  'tid':3,  'uid':3,  'n_team_agent':10, 'type':'PlaneAgent',  'init_fn_name':'init_air',  },
        { 'team':0,  'tid':4,  'uid':4,  'n_team_agent':10, 'type':'PlaneAgent',  'init_fn_name':'init_air',  },
        { 'team':0,  'tid':5,  'uid':5,  'n_team_agent':10, 'type':'PlaneAgent',  'init_fn_name':'init_air',  },
        { 'team':0,  'tid':6,  'uid':6,  'n_team_agent':10, 'type':'PlaneAgent',  'init_fn_name':'init_air',  },
        { 'team':0,  'tid':7,  'uid':7,  'n_team_agent':10, 'type':'PlaneAgent',  'init_fn_name':'init_air',  },
        { 'team':0,  'tid':8,  'uid':8,  'n_team_agent':10, 'type':'PlaneAgent',  'init_fn_name':'init_air',  },
        { 'team':0,  'tid':9,  'uid':9,  'n_team_agent':10, 'type':'PlaneAgent',  'init_fn_name':'init_air',  },
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
    obs_n_entity = 11
    ActionFormat = 'ASCII'
