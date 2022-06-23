import numpy as np

dictionary_items = [
    'ActionSet2::Idle;DynamicGuard'     ,
    'ActionSet2::Idle;StaticAlert'      ,
    'ActionSet2::Idle;AggressivePersue' ,
    'ActionSet2::SpecificMoving;Dir+X'  ,
    'ActionSet2::SpecificMoving;Dir+X+Y',
    'ActionSet2::SpecificMoving;Dir+Y'  ,
    'ActionSet2::SpecificMoving;Dir-X+Y',
    'ActionSet2::SpecificMoving;Dir-X'  ,
    'ActionSet2::SpecificMoving;Dir-X-Y',
    'ActionSet2::SpecificMoving;Dir-Y'  ,
    'ActionSet2::SpecificMoving;Dir+X-Y',
    'ActionSet2::SpecificMoving;X=-6800.0 Y=6830.0 Z=397.215454',
    'ActionSet2::PatrolMoving;Dir+X'    ,
    'ActionSet2::PatrolMoving;Dir+X+Y'  ,
    'ActionSet2::PatrolMoving;Dir+Y'    ,
    'ActionSet2::PatrolMoving;Dir-X+Y'  ,
    'ActionSet2::PatrolMoving;Dir-X'    ,
    'ActionSet2::PatrolMoving;Dir-X-Y'  ,
    'ActionSet2::PatrolMoving;Dir-Y'    ,
    'ActionSet2::PatrolMoving;Dir+X-Y'  ,
    'ActionSet2::PatrolMoving;X=-6800.0 Y=6830.0 Z=397.215454',
    'ActionSet2::PatrolMoving;X=0.0 Y=0.0 Z=100.0',
    'ActionSet2::SpecificAttacking;T1-1',
    'ActionSet2::SpecificAttacking;T1-2',
    'ActionSet2::SpecificAttacking;T1-3',
    'ActionSet2::SpecificAttacking;T1-4',
    'ActionSet2::N/A;N/A',
    'ActionSet2::SpecificAttacking;UID-0',
    'ActionSet2::SpecificAttacking;UID-1',
    'ActionSet2::SpecificAttacking;UID-2',
    'ActionSet2::SpecificAttacking;UID-3',
    'ActionSet2::SpecificAttacking;UID-4',
    'ActionSet2::SpecificAttacking;UID-5',
    'ActionSet2::SpecificAttacking;UID-6',
    'ActionSet2::SpecificAttacking;UID-7',
    'ActionSet2::SpecificAttacking;UID-8',
]

dictionary_n_actions = len(dictionary_items)

digit2act_dictionary = {
    i: dictionary_items[i] for i, item in enumerate(dictionary_items)
}

act2digit_dictionary = {
    dictionary_items[i]:i for i, item in enumerate(dictionary_items)
}

no_act_placeholder = 'ActionSet2::N/A;N/A'

agent_json2local_attrs = [
            # json key       ----->    agent key
            ('agentAlive',              'alive'),
            ('agentTeam',               'team'),
            ('indexInTeam',             'index'),
            ('uId',                     'uid_remote'),
            ('acceptRLControl',         'rl_agent'),
            ('maxMoveSpeed',            'max_speed'),
            ('agentLocation',           'location'),
            ('agentRotation',           'rotation'),
            ('agentScale',              'scale3'),
            ('agentVelocity',           'velocity'),
            ('agentHp',                 'hp'),
            ('weaponCD',                'weapon_cd'),
            ('maxEpisodeStep',          'time_limit'),
            ('interaction',             'interaction'), 
            ('type',                    'type'),
]


h_map_center = (-7290.0, 6010.0)
h_grid_size = 400
v_ground = 340
v_grid_size = 1000

x_arr = np.array([h_map_center[0]+v_grid_size*i for i in range(-20, 20)])   # 0~39, 40, 1
y_arr = np.array([h_map_center[1]+v_grid_size*i for i in range(-20, 20)])   # 0~39, 40, 40
z_arr = np.array([v_ground+v_grid_size*i for i in range(4)])                # 0~3,  4,  1600
# offset                                                                    # 0~1,  2,  6400


# output $y \in [1000, 12800]$
def _2digit(main_cmd, x, y, z):
    z_logit = np.argmin(np.abs(z - z_arr))
    x_logit = np.argmin(np.abs(x - x_arr))
    y_logit = np.argmin(np.abs(y - y_arr))
    if main_cmd=='SpecificMoving': cmd_logit = 0
    elif main_cmd=='PatrolMoving': cmd_logit = 1
    ls_mod = [1,40,1600,6400]
    offset = 1000
    x = np.array([x_logit, y_logit, z_logit, cmd_logit])
    print(x)
    y = np.dot(x, ls_mod)+offset
    return y

def _2coordinate(x):
    offset = 1000
    ls_mod = [1,40,1600,6400]
    x = x - offset
    res = []
    for mod in reversed(ls_mod):
        tmp = x // mod
        x = x - tmp*mod
        res.append(tmp)
    res = list(reversed(res))
    x_logit, y_logit, z_logit, cmd_logit = res
    if    cmd_logit == 0 : main_cmd ='SpecificMoving'
    elif  cmd_logit == 1 : main_cmd ='PatrolMoving'  
    x = x_arr[x_logit]
    y = y_arr[y_logit]
    z = z_arr[z_logit]
    print(main_cmd, x, y, z)
    return main_cmd, x, y, z

    

def encode_action_as_digits(main_cmd, sub_cmd, x=None, y=None, z=None, UID=None, T=None, T_index=None):
    main_cmd_encoder = {
        "Idle"                  : 0, 
        "SpecificMoving"        : 1, 
        "PatrolMoving"          : 2, 
        "SpecificAttacking"     : 3, 
        "N/A"                   : 4, 
    }
    sub_cmd_encoder = {
        "DynamicGuard"              : 0, 
        "StaticAlert"               : 1, 
        "AggressivePersue"          : 2, 
        "SpecificAttacking"         : 3, 
        "N/A"                       : 4, 
        'Dir+X'                     : 5,
        'Dir+X+Y'                   : 6,
        'Dir+Y'                     : 7,
        'Dir-X+Y'                   : 8,
        'Dir-X'                     : 9,
        'Dir-X-Y'                   : 10,
        'Dir-Y'                     : 11,
        'Dir+X-Y'                   : 12, 
    }
    return np.array([
        main_cmd_encoder[main_cmd],
        sub_cmd_encoder[sub_cmd],
        x if x is not None                  else np.inf,
        y if y is not None                  else np.inf,
        z if z is not None                  else np.inf,
        UID if UID is not None              else np.inf,
        T if T is not None                  else np.inf,
        T_index if T_index is not None      else np.inf
    ])


def decode_action_as_string(digits):
    main_cmd_decoder = {
        0 :"Idle"                , 
        1 :"SpecificMoving"      , 
        2 :"PatrolMoving"        , 
        3 :"SpecificAttacking"   , 
        4 :"N/A"                 , 
    }
    sub_cmd_decoder = {
        0 :"DynamicGuard"             , 
        1 :"StaticAlert"              , 
        2 :"AggressivePersue"         , 
        3 :"SpecificAttacking"        , 
        4 :"N/A"                      , 
        5 :'Dir+X'                    ,
        6 :'Dir+X+Y'                  ,
        7 :'Dir+Y'                    ,
        8 :'Dir-X+Y'                  ,
        9 :'Dir-X'                    ,
        10:'Dir-X-Y'                  ,
        11:'Dir-Y'                    ,
        12:'Dir+X-Y'                  , 
    }
    main_cmd = main_cmd_decoder[digits[0]]
    sub_cmd = sub_cmd_decoder[digits[1]]
    x       = digits[2] if np.isfinite(digits[2]) else None
    y       = digits[3] if np.isfinite(digits[3]) else None
    z       = digits[4] if np.isfinite(digits[4]) else None
    UID     = digits[5] if np.isfinite(digits[5]) else None
    T       = digits[6] if np.isfinite(digits[6]) else None
    T_index = digits[7] if np.isfinite(digits[7]) else None

    if main_cmd == "Idle":
        res = 'ActionSet2::Idle;%s'%sub_cmd
        assert res in dictionary_items, '指令错误无法解析'
    elif main_cmd == "SpecificMoving":
        if sub_cmd == 'N/A':
            res = 'ActionSet2::SpecificMoving;X=%f Y=%f Z=%f'%(x,y,z)
        else:
            res = 'ActionSet2::SpecificMoving;%s'%sub_cmd
    elif main_cmd == "PatrolMoving":
        if sub_cmd == 'N/A':
            res = 'ActionSet2::PatrolMoving;X=%f Y=%f Z=%f'%(x,y,z)
        else:
            res = 'ActionSet2::PatrolMoving;%s'%sub_cmd
    elif main_cmd == "SpecificAttacking":
        # 'ActionSet2::SpecificAttacking;T1-3',
        # 'ActionSet2::SpecificAttacking;UID-4',
        assert sub_cmd == 'N/A', '指令错误无法解析'
        if UID is not None:
            res = 'ActionSet2::SpecificAttacking;UID-%d'%UID
        else:
            res = 'ActionSet2::SpecificAttacking;T%d-%d'%(T,T_index)
    elif main_cmd == "N/A":
        res = 'ActionSet2::N/A;N/A'
    else:
        print('指令错误无法解析')
    return res


AgentPropertyDefaults = {
    'ClassName': 'RLA_CAR',     # FString ClassName = "";
    'AcceptRLControl': True,    # bool AcceptRLControl = 0;
    'AgentTeam': 0,             # int AgentTeam = 0;
    'IndexInTeam': 0,           # int IndexInTeam = 0;
    'UID': 0,                   # int UID = 0;
    'MaxMoveSpeed': 600,        # move speed, test ok
    'InitLocation': { 'x': 0,  'y': 0, 'z': 0, },
    'InitRotation': { 'x': 0,  'y': 0, 'z': 0, },
    'AgentScale'  : { 'x': .5,  'y': .5, 'z': .5, },     # agent size, test ok
    'InitVelocity': { 'x': 0,  'y': 0, 'z': 0, },
    'AgentHp':100,
    "WeaponCD": 1,              # weapon fire rate
    "IsTeamReward": True,
    "Type": "",
    "DodgeProb": 0.1,           # probability of escaping dmg 闪避概率, test ok
    "ExplodeDmg": 50,           # ms explode dmg. test ok
    "FireRange": 1000.0,        # <= 1500
    "GuardRange": 1400.0,       # <= 1500
    'Color':'(R=0,G=1,B=0,A=1)',    # color
    "FireRange": 1000,
    'RSVD1':'',
    'RSVD2':'',
}

'''



'''